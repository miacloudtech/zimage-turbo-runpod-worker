import base64
import gc
import os
import threading
import time
import traceback
from io import BytesIO

import torch
import runpod
from diffusers import ZImagePipeline

MODEL_ID = "/models/zimage-turbo"

# For your site: allow 1280px on the longest side.
DEFAULT_STEPS = int(os.getenv("AI_Z_STEPS", "8"))
MAX_IMAGE_SIDE = int(os.getenv("AI_MAX_IMAGE_SIDE", "1280"))
REFRESH_WORKER_AFTER_JOB = os.getenv("AI_REFRESH_WORKER_AFTER_JOB", "0").strip().lower() in ("1", "true", "yes")

_pipe = None
_device = None
_generate_lock = threading.Lock()


def log_gpu_memory(label: str) -> None:
    if not torch.cuda.is_available():
        print(f"[GPU] {label}: CUDA not available", flush=True)
        return

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        print(
            f"[GPU] {label}: "
            f"free={free_bytes / 1024**3:.2f}GB "
            f"total={total_bytes / 1024**3:.2f}GB "
            f"allocated={allocated / 1024**3:.2f}GB "
            f"reserved={reserved / 1024**3:.2f}GB",
            flush=True,
        )
    except Exception as exc:
        print(f"[GPU] {label}: could not read memory info: {exc}", flush=True)


def cleanup_cuda(label: str = "cleanup") -> None:
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
        except Exception as exc:
            print(f"[GPU] {label}: CUDA cleanup warning: {exc}", flush=True)
    log_gpu_memory(label)


def image_to_b64(image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def load_pipeline():
    global _pipe, _device

    if _pipe is not None:
        return _pipe, _device

    print("[Cold Start] Loading Z-Image-Turbo...", flush=True)
    start = time.time()

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_gpu_memory("before pipeline load")

    pipe = ZImagePipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=False,
    )

    pipe.to(_device)

    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass

    print(f"[Cold Start] Pipeline loaded in {time.time() - start:.1f}s", flush=True)
    _pipe = pipe
    log_gpu_memory("after pipeline load")

    return _pipe, _device


def normalise_dimension(value, default_value: int) -> int:
    try:
        value = int(value)
    except Exception:
        value = default_value

    value = max(512, min(MAX_IMAGE_SIDE, value))
    return max(512, value // 32 * 32)


def normalise_steps(value) -> int:
    try:
        steps = int(value)
    except Exception:
        steps = DEFAULT_STEPS
    return max(4, min(12, steps))


def get_steps_from_input(job_input: dict) -> int:
    # Your PHP currently sends num_inference_steps. Keep steps too for compatibility.
    if "num_inference_steps" in job_input:
        return normalise_steps(job_input.get("num_inference_steps"))
    if "steps" in job_input:
        return normalise_steps(job_input.get("steps"))
    return normalise_steps(DEFAULT_STEPS)


def generate_one(prompt: str, width: int, height: int, seed=None, steps=None):
    pipe, device = load_pipeline()
    steps = normalise_steps(steps)

    if seed is None or str(seed).strip() == "" or str(seed).strip() == "-1":
        generator = None
        actual_seed = None
    else:
        actual_seed = int(seed)
        generator_device = "cuda" if device.type == "cuda" else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(actual_seed)

    start = time.time()
    image = None
    output = None

    log_gpu_memory("before generation")

    try:
        with torch.inference_mode():
            output = pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=0.0,
                generator=generator,
            )

        image = output.images[0]

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        wall_time = time.time() - start

        return {
            "image_b64": image_to_b64(image),
            "wall_time_s": round(wall_time, 2),
            "seed": actual_seed,
            "width": width,
            "height": height,
            "steps": steps,
        }
    finally:
        try:
            del image
        except Exception:
            pass
        try:
            del output
        except Exception:
            pass
        cleanup_cuda("after generation cleanup")


def handler(job):
    # This only serialises work inside one worker process. Concurrency should be
    # handled by increasing RunPod Max Workers, not by running multiple jobs on one GPU.
    with _generate_lock:
        try:
            job_input = job.get("input", {})

            prompt = str(job_input.get("prompt", "")).strip()
            if not prompt:
                return {"error": "Missing required field: prompt", "refresh_worker": REFRESH_WORKER_AFTER_JOB}

            width = normalise_dimension(job_input.get("width", 1024), 1024)
            height = normalise_dimension(job_input.get("height", 1024), 1024)
            seed = job_input.get("seed", None)
            steps = get_steps_from_input(job_input)

            print(
                f"[Job] width={width} height={height} steps={steps} "
                f"max_side={MAX_IMAGE_SIDE} refresh_worker={REFRESH_WORKER_AFTER_JOB}",
                flush=True,
            )

            result = generate_one(prompt=prompt, width=width, height=height, seed=seed, steps=steps)

            if REFRESH_WORKER_AFTER_JOB:
                result["refresh_worker"] = True

            return result

        except torch.cuda.OutOfMemoryError as exc:
            traceback.print_exc()
            cleanup_cuda("after CUDA OOM")
            return {
                "error": "cuda_out_of_memory",
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "refresh_worker": True,
            }

        except Exception as exc:
            traceback.print_exc()
            cleanup_cuda("after general exception")
            return {
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "refresh_worker": REFRESH_WORKER_AFTER_JOB,
            }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
