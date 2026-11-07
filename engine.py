"""
DreamCanvas engine: text-to-image generation with automatic quality scoring.

  * Stable Diffusion 1.5 in half precision, with attention slicing and VAE tiling,
    so it fits on a free Colab T4 GPU. Falls back to CPU when no GPU is present.
  * CLIP scoring: measures how well each image matches the prompt.
  * Best-of-N: generates several candidates and keeps the one CLIP rates highest.
  * Style presets and negative prompts for better-looking output.
  * Reproducible seeds.
  * The built-in safety checker stays on.
  * Demo mode (GENAI_DEMO=1) that needs no torch or GPU, for working on the website.
"""

from __future__ import annotations

import base64
import hashlib
import io
import math
import os
import random
import threading
import time
from dataclasses import dataclass, field

from PIL import Image, ImageDraw, ImageFilter

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

MODELS = {
    # Good quality, supports negative prompts. ~25 steps.
    "sd15": {
        "repo": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "label": "Stable Diffusion 1.5",
        "default_steps": 25,
        "default_guidance": 7.5,
        "max_steps": 50,
    },
    # Very fast (1-4 steps), great for live demos. No negative prompt / guidance.
    "turbo": {
        "repo": "stabilityai/sd-turbo",
        "label": "SD-Turbo (fast)",
        "default_steps": 2,
        "default_guidance": 0.0,
        "max_steps": 6,
    },
}

CLIP_REPO = "openai/clip-vit-base-patch32"

STYLES = {
    "none": {"label": "No style", "suffix": "", "negative": ""},
    "photo": {
        "label": "Photorealistic",
        "suffix": "professional photograph, natural lighting, sharp focus, highly detailed, 8k",
        "negative": "cartoon, painting, illustration, drawing, cgi",
    },
    "cinematic": {
        "label": "Cinematic",
        "suffix": "cinematic film still, dramatic lighting, shallow depth of field, color graded, epic composition",
        "negative": "flat lighting, amateur",
    },
    "anime": {
        "label": "Anime",
        "suffix": "anime artwork, vibrant colors, clean line art, studio quality, detailed background",
        "negative": "photo, realistic, 3d render",
    },
    "digital-art": {
        "label": "Digital art",
        "suffix": "digital painting, concept art, artstation trending, highly detailed, vivid colors",
        "negative": "photo, blurry",
    },
    "watercolor": {
        "label": "Watercolor",
        "suffix": "watercolor painting, soft washes, paper texture, delicate brush strokes",
        "negative": "photo, 3d render, harsh lines",
    },
    "3d": {
        "label": "3D render",
        "suffix": "3d render, octane render, soft global illumination, physically based materials",
        "negative": "flat, 2d, sketch",
    },
    "pixel": {
        "label": "Pixel art",
        "suffix": "pixel art, 16-bit, retro game style, crisp pixels",
        "negative": "blurry, smooth gradients, photo",
    },
}

BASE_NEGATIVE = "lowres, blurry, worst quality, jpeg artifacts, deformed, distorted, watermark, signature"


@dataclass
class GenRequest:
    prompt: str
    negative_prompt: str = ""
    style: str = "none"
    steps: int | None = None
    guidance: float | None = None
    width: int = 512
    height: int = 512
    seed: int | None = None
    candidates: int = 1


@dataclass
class GenResult:
    images: list = field(default_factory=list)  # [{png_b64, seed, clip_score, flagged}]
    ignored_text: str = ""  # words past the model's 77-token limit
    best_index: int = 0
    final_prompt: str = ""
    final_negative: str = ""
    elapsed: float = 0.0
    model: str = ""
    demo: bool = False


def build_prompts(req: GenRequest, supports_negative: bool) -> tuple[str, str]:
    style = STYLES.get(req.style, STYLES["none"])
    prompt = req.prompt.strip()
    if style["suffix"]:
        # The short style name goes first (SD only reads ~75 tokens, so the end may be cut),
        # the longer quality wording goes last.
        head, _, tail = style["suffix"].partition(", ")
        prompt = f"{head}, {prompt}" + (f", {tail}" if tail else "")
    negative = ""
    if supports_negative:
        parts = [req.negative_prompt.strip(), style["negative"], BASE_NEGATIVE]
        negative = ", ".join(p for p in parts if p)
    return prompt, negative


def to_b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------------------------------- #
# Real engine (Stable Diffusion + CLIP)
# --------------------------------------------------------------------------- #

class DiffusionEngine:
    demo = False

    def __init__(self, model_key: str = "sd15"):
        self.model_key = model_key if model_key in MODELS else "sd15"
        self.cfg = MODELS[self.model_key]
        self.status = "loading"
        self.error = None
        self.device = "detecting"  # set to cuda/cpu once torch has loaded
        self._lock = threading.Lock()
        self.pipe = None
        self.clip_model = None
        self.clip_proc = None

    # Loading is slow (downloads ~4 GB the first time) so it runs in a thread.
    def load(self):
        try:
            import torch
            from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler
            from transformers import CLIPModel, CLIPProcessor

            self.torch = torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if self.device == "cuda" else torch.float32

            pipe = AutoPipelineForText2Image.from_pretrained(
                self.cfg["repo"],
                torch_dtype=dtype,
                variant="fp16" if self.device == "cuda" else None,
            )
            if self.model_key == "sd15":
                # DPM++ reaches good quality in ~25 steps instead of 50.
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipe.scheduler.config, use_karras_sigmas=True
                )
            pipe = pipe.to(self.device)
            # Memory optimisations
            pipe.enable_attention_slicing()
            pipe.vae.enable_tiling()
            pipe.set_progress_bar_config(disable=True)
            self.pipe = pipe

            self.clip_model = CLIPModel.from_pretrained(CLIP_REPO).to(self.device).eval()
            self.clip_proc = CLIPProcessor.from_pretrained(CLIP_REPO)
            self.status = "ready"
        except Exception as e:  # surfaced through /api/health
            self.status = "error"
            self.error = f"{type(e).__name__}: {e}"

    def clip_scores(self, prompt: str, images: list[Image.Image]) -> list[float]:
        """CLIP score = 100 * cosine(text, image), clipped at 0. ~25-35 is a good match for SD."""
        torch = self.torch
        with torch.no_grad():
            inputs = self.clip_proc(
                text=[prompt], images=images, return_tensors="pt", padding=True, truncation=True
            ).to(self.device)
            out = self.clip_model(**inputs)
            img = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
            txt = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
            sims = (img @ txt.T).squeeze(-1)
        return [round(max(float(s) * 100, 0.0), 2) for s in sims]

    def generate(self, req: GenRequest) -> GenResult:
        if self.status != "ready":
            raise RuntimeError(f"Model not ready (status: {self.status})")
        torch = self.torch
        supports_negative = self.model_key != "turbo"
        prompt, negative = build_prompts(req, supports_negative)
        steps = req.steps or self.cfg["default_steps"]
        guidance = self.cfg["default_guidance"] if req.guidance is None else req.guidance
        if not supports_negative:
            guidance = 0.0
        base_seed = req.seed if req.seed is not None else random.randint(0, 2**31 - 1)
        seeds = [base_seed + i for i in range(req.candidates)]

        t0 = time.time()
        with self._lock:  # one generation at a time on a single GPU
            images, flagged = [], []
            # One candidate at a time keeps memory low enough for a free T4 GPU.
            for s in seeds:
                kwargs = dict(
                    prompt=prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    width=req.width,
                    height=req.height,
                    generator=torch.Generator(device=self.device).manual_seed(s),
                )
                if supports_negative:
                    kwargs["negative_prompt"] = negative
                out = self.pipe(**kwargs)
                images.append(out.images[0])
                nsfw = getattr(out, "nsfw_content_detected", None)
                flagged.append(bool(nsfw[0]) if nsfw else False)
            # Score against the user's own words (not the style suffix).
            scores = self.clip_scores(req.prompt, images)

        for i, f in enumerate(flagged):
            if f:
                scores[i] = 0.0
        best = max(range(len(images)), key=lambda i: scores[i])
        ignored = ""
        tok = getattr(self.pipe, "tokenizer", None)
        if tok is not None:
            ids = tok(prompt, truncation=False).input_ids
            limit = tok.model_max_length  # 77 incl. start/end tokens
            if len(ids) > limit:
                ignored = tok.decode(ids[limit - 1:-1]).strip()
        return GenResult(
            ignored_text=ignored,
            images=[
                {"png_b64": to_b64_png(im), "seed": s, "clip_score": sc, "flagged": bool(f)}
                for im, s, sc, f in zip(images, seeds, scores, flagged)
            ],
            best_index=best,
            final_prompt=prompt,
            final_negative=negative,
            elapsed=round(time.time() - t0, 2),
            model=self.cfg["label"],
        )

    def info(self) -> dict:
        return {
            "status": self.status,
            "error": self.error,
            "demo": False,
            "model": self.cfg["label"],
            "model_key": self.model_key,
            "device": self.device,
            "supports_negative": self.model_key != "turbo",
            "default_steps": self.cfg["default_steps"],
            "default_guidance": self.cfg["default_guidance"],
            "max_steps": self.cfg["max_steps"],
        }


# --------------------------------------------------------------------------- #
# Demo engine: no ML libraries, draws a deterministic abstract image.
# Lets you build / test the website on any laptop.
# --------------------------------------------------------------------------- #

class DemoEngine:
    demo = True
    status = "ready"

    def load(self):
        pass

    @staticmethod
    def _palette(seed: int):
        rnd = random.Random(seed)
        h = rnd.random()
        cols = []
        for k in range(4):
            hh = (h + k * 0.17 + rnd.uniform(-0.05, 0.05)) % 1.0
            r, g, b = _hsv(hh, 0.55 + 0.3 * rnd.random(), 0.55 + 0.4 * rnd.random())
            cols.append((r, g, b))
        return cols

    def _draw(self, text: str, seed: int, w: int, h: int) -> Image.Image:
        rnd = random.Random(seed)
        c = self._palette(seed)
        img = Image.new("RGB", (w, h))
        px = img.load()
        for y in range(h):
            t = y / (h - 1)
            for x in range(0, w):
                u = x / (w - 1)
                m = 0.5 + 0.5 * math.sin(u * 3 + t * 2 + seed % 7)
                top = [c[0][i] * (1 - u) + c[1][i] * u for i in range(3)]
                bot = [c[2][i] * (1 - m) + c[3][i] * m for i in range(3)]
                px[x, y] = tuple(int(top[i] * (1 - t) + bot[i] * t) for i in range(3))
        d = ImageDraw.Draw(img, "RGBA")
        for _ in range(14):
            r = rnd.randint(w // 14, w // 4)
            x, y = rnd.randint(0, w), rnd.randint(0, h)
            col = c[rnd.randint(0, 3)]
            d.ellipse([x - r, y - r, x + r, y + r], fill=(*col, rnd.randint(40, 110)))
        img = img.filter(ImageFilter.GaussianBlur(radius=w / 60))
        d = ImageDraw.Draw(img, "RGBA")
        d.rectangle([0, h - 34, w, h], fill=(0, 0, 0, 120))
        d.text((10, h - 25), ("DEMO · " + text)[:70], fill=(255, 255, 255, 230))
        return img

    def generate(self, req: GenRequest) -> GenResult:
        prompt, negative = build_prompts(req, True)
        base_seed = req.seed if req.seed is not None else random.randint(0, 2**31 - 1)
        t0 = time.time()
        imgs = []
        for i in range(req.candidates):
            s = base_seed + i
            key = int(hashlib.md5(f"{req.prompt}|{req.style}|{s}".encode()).hexdigest()[:8], 16)
            im = self._draw(req.prompt, key, req.width, req.height)
            imgs.append({"png_b64": to_b64_png(im), "seed": s, "clip_score": None, "flagged": False})
        time.sleep(0.6)  # feel like a real request
        return GenResult(
            images=imgs, best_index=0, final_prompt=prompt, final_negative=negative,
            elapsed=round(time.time() - t0, 2), model="Demo (no model loaded)", demo=True,
        )

    def info(self) -> dict:
        return {
            "status": "ready", "error": None, "demo": True, "model": "Demo (no model loaded)",
            "model_key": "demo", "device": "none", "supports_negative": True,
            "default_steps": 25, "default_guidance": 7.5, "max_steps": 50,
        }


def _hsv(h, s, v):
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def create_engine():
    if os.environ.get("GENAI_DEMO", "0") == "1":
        return DemoEngine()
    return DiffusionEngine(os.environ.get("GENAI_MODEL", "sd15"))
