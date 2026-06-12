"""
RunPod serverless handler for Sana 0.6B — text prompt → image generation.

Architecture:
  - DC-AE (32x compression) + Gemma 2 2B text encoder + linear attention DiT
  - 0.6B parameters, ~1.5GB weights in fp16
  - Runs on any GPU with >=4GB VRAM (e.g., RTX A2000 6GB)
  - ~1s for a 1024×1024 image at 18 steps
  - Diffusers-native SanaPipeline (from_pretrained)

Environment (set by RunPod template):
  - RUNPOD_POD_ID       — auto
  - RUNPOD_AI_API_KEY   — auto
  - (No HF_TOKEN needed — Sana is Apache 2.0, no gated access)

Input schema (via RunPod serverless job):
  {
    "input": {
      "prompt": "a cyberpunk cat",           // REQUIRED — text prompt
      "negative_prompt": "",                  // optional — negative prompt
      "height": 1024,                         // optional — image height (512-4096)
      "width": 1024,                          // optional — image width (512-4096)
      "num_inference_steps": 18,              // optional — fewer = faster (4-step possible)
      "guidance_scale": 5.0,                  // optional — CFG scale
      "pag_guidance_scale": 2.0,              // optional — PAG scale
      "seed": null                            // optional — random seed (null = random)
    }
  }

Output:
  {
    "image_b64": "<base64-encoded PNG>",
    "prompt": "a cyberpunk cat",
    "seed": 42,
    "wall_time_s": 0.8
  }
"""

import base64
import os
import random
import time
import traceback
from io import BytesIO

# ── Environment setup ─────────────────────────────────────────────────────────
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from diffusers import SanaPipeline

# Disable flash/mem-efficient SDPA — avoids "no kernel image" CUDA errors
# that occur when the GPU compute capability isn't in the precompiled kernels.
# Gemma 2 text encoder uses SDPA; math backend works on all CUDA GPUs.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# ── Model path (baked into image at BUILD TIME) ──────────────────────────────
MODEL_ID = "/models/sana"

# ── Global pipeline (loaded once, reused across jobs) ─────────────────────────
_pipe = None
_device = None


def load_pipeline():
    """Load Sana 0.6B pipeline once and cache globally."""
    global _pipe, _device
    if _pipe is not None:
        return _pipe, _device

    print("[Cold Start] Loading Sana 0.6B pipeline...", flush=True)
    t0 = time.time()

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print(f"  Device: {_device}, dtype: {dtype}", flush=True)

    # Load pipeline from HuggingFace (weights NOT in image — downloaded on cold start)
    # SanaPipeline uses DC-AE + Gemma 2 text encoder + linear attention DiT
    pipe = SanaPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        variant="fp16",
    )

    # Move fully to GPU — cpu_offload causes issues in serverless containers
  pipe = pipe.to(_device)
  
  # Avoid black output from fp16 VAE/DC-AE decoding.
  # Keep the main model fast, but decode the final image in safer precision.
  try:
      if hasattr(pipe, "vae") and pipe.vae is not None:
          pipe.vae.to(dtype=torch.float32)
          print("[Cold Start] VAE moved to float32 to avoid black image output.", flush=True)
  except Exception as exc:
      print(f"[Cold Start] Could not move VAE to float32: {exc}", flush=True)
  
  print(f"[Cold Start] Pipeline ready in {time.time() - t0:.1f}s", flush=True)
  
  _pipe = pipe
  return _pipe, _device


def image_to_b64(image) -> str:
    """Convert PIL Image to base64 PNG string."""
    buf = BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def run_inference(
    prompt: str,
    negative_prompt: str = "",
    height: int = 1024,
    width: int = 1024,
    num_inference_steps: int = 18,
    guidance_scale: float = 5.0,
    pag_guidance_scale: float = 2.0,
    seed: int | None = None,
) -> tuple:
    """
    Run Sana 0.6B inference.
    Returns (PIL Image, actual_seed, wall_time_s).
    """
    pipe, device = load_pipeline()

    # Set seed
    if seed is None:
        seed = random.randint(0, 2**31 - 1)
    generator = torch.Generator(device=device).manual_seed(seed)

    print(f"[Inference] Generating: prompt='{prompt[:80]}'", flush=True)
    print(
        f"  size={width}x{height}, steps={num_inference_steps}, "
        f"cfg={guidance_scale}, pag={pag_guidance_scale}, seed={seed}",
        flush=True,
    )

    t_start = time.time()

    # Run inference.
    # Do not wrap the entire pipeline in fp16 autocast because that can force
    # the decoder/VAE path back into fp16 and cause black output.
    with torch.inference_mode():
        pipe_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt or "",
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=1,
            generator=generator,
            use_resolution_binning=True,
        )
    
        # pag_guidance_scale is only supported by SanaPAGPipeline, not the
        # base SanaPipeline. Pass it only if the loaded pipeline accepts it.
        try:
            import inspect
    
            if "pag_guidance_scale" in inspect.signature(pipe.__call__).parameters:
                pipe_kwargs["pag_guidance_scale"] = pag_guidance_scale
        except (TypeError, ValueError):
            pass
    
        images = pipe(**pipe_kwargs).images

    wall_time = time.time() - t_start
    print(f"[Done] Generation took {wall_time:.1f}s", flush=True)

    return images[0], seed, wall_time


# ═══════════════════════════════════════════════════════════════════════════════
# RunPod Serverless Handler
# ═══════════════════════════════════════════════════════════════════════════════


def handler(job):
    """
    RunPod serverless handler: text prompt → base64 PNG.

    Called once per job. The pipeline stays loaded across jobs (global).
    """
    job_input = job.get("input", {})
    prompt = job_input.get("prompt", "")

    if not prompt:
        return {"error": "Missing required field: prompt"}

    negative_prompt = str(job_input.get("negative_prompt", ""))
    height = int(job_input.get("height", 1024))
    width = int(job_input.get("width", 1024))
    num_inference_steps = int(job_input.get("num_inference_steps", 18))
    guidance_scale = float(job_input.get("guidance_scale", 5.0))
    pag_guidance_scale = float(job_input.get("pag_guidance_scale", 2.0))
    seed_raw = job_input.get("seed", None)

    if seed_raw is None or seed_raw == "" or str(seed_raw).strip() == "-1":
        seed = None
    else:
        seed = int(seed_raw)

    # Validate dimensions (Sana supports 512-4096 in 32px multiples)
    height = max(512, min(4096, height // 32 * 32))
    width = max(512, min(4096, width // 32 * 32))

    # Validate steps (4-step possible with Sana's efficient attention)
    num_inference_steps = max(1, min(50, num_inference_steps))

    try:
        # Run inference
        image, actual_seed, wall_time = run_inference(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            pag_guidance_scale=pag_guidance_scale,
            seed=seed,
        )

        # Encode as base64 PNG
        print("[Worker] Encoding image as base64 PNG...", flush=True)
        image_b64 = image_to_b64(image)

        return {
            "image_b64": image_b64,
            "prompt": prompt,
            "seed": actual_seed,
            "wall_time_s": round(wall_time, 1),
            "width": width,
            "height": height,
        }

    except Exception as exc:
        traceback.print_exc()
        return {
            "error": f"Sana inference failed: {str(exc)}",
            "traceback": traceback.format_exc(),
        }


# ── Entrypoint ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
