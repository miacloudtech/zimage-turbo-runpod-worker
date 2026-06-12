import base64
import os
import time
import traceback
from io import BytesIO

import torch
import runpod
from diffusers import ZImagePipeline

MODEL_ID = "/models/zimage-turbo"

_pipe = None
_device = None


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

    pipe = ZImagePipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=False,
    )

    pipe.to(_device)

    # Optional: only enable later if stable
    # try:
    #     pipe.transformer.set_attention_backend("flash")
    #     print("[Cold Start] Flash attention enabled.", flush=True)
    # except Exception as exc:
    #     print(f"[Cold Start] Flash attention not enabled: {exc}", flush=True)

    print(f"[Cold Start] Pipeline loaded in {time.time() - start:.1f}s", flush=True)

    _pipe = pipe
    return _pipe, _device


def generate_one(prompt: str, width: int, height: int, seed=None):
    pipe, device = load_pipeline()

    if seed is None or str(seed).strip() == "" or str(seed).strip() == "-1":
        generator = None
        actual_seed = None
    else:
        actual_seed = int(seed)
        generator = torch.Generator(device="cuda").manual_seed(actual_seed)

    start = time.time()

    image = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=9,   # Z-Image-Turbo example setting
        guidance_scale=0.0,      # Turbo guidance should be 0
        generator=generator,
    ).images[0]

    wall_time = time.time() - start

    return {
        "image_b64": image_to_b64(image),
        "wall_time_s": round(wall_time, 2),
        "seed": actual_seed,
        "width": width,
        "height": height,
    }


def handler(job):
    try:
        job_input = job.get("input", {})

        prompt = str(job_input.get("prompt", "")).strip()
        if not prompt:
            return {"error": "Missing required field: prompt"}

        width = int(job_input.get("width", 1024))
        height = int(job_input.get("height", 1024))
        seed = job_input.get("seed", None)

        # Keep first test simple and safe
        width = max(512, min(1536, width // 32 * 32))
        height = max(512, min(1536, height // 32 * 32))

        result = generate_one(
            prompt=prompt,
            width=width,
            height=height,
            seed=seed,
        )

        return result

    except Exception as exc:
        traceback.print_exc()
        return {
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
