"""
ComfyUI Custom Nodes for LLaDA 2.0-Uni
Supports: Text-to-Image, Image Understanding, Image Editing, Decode, Unload

Features:
- Attention backend selection: Flash Attention, SDPA, SageAttention
- CPU offload mode for limited VRAM
- Auto VRAM cleanup after inference
- Manual unload node
"""

import torch
import numpy as np
from PIL import Image
from typing import Optional
import os
import shutil
import subprocess


# ═══════════════════════════════════════════════════════════════
#  LLaDA Loader
# ═══════════════════════════════════════════════════════════════

class LLaDALoader:
    """Load LLaDA 2.0-Uni model with attention & offload options."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "attention": (["flash_attn", "sdpa"],),
                "dtype": (["bf16", "fp8"],),
                "offload": ("BOOLEAN", {"default": False}),
                "device": (["cuda", "cpu"],),
            }
        }

    RETURN_TYPES = ("LLaDA_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "LLaDA 2.0-Uni"
    DESCRIPTION = "Load LLaDA 2.0-Uni from ComfyUI/models/llada2uni with optional auto-download."

    @staticmethod
    def _get_fixed_model_path() -> str:
        # nodes.py -> custom node dir -> custom_nodes -> ComfyUI root
        comfyui_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        return os.path.join(comfyui_root, "models", "llada2uni")

    @staticmethod
    def _has_model_files(model_path: str) -> bool:
        if not os.path.isdir(model_path):
            return False
        return len(os.listdir(model_path)) > 0

    @staticmethod
    def _download_model_if_missing(model_path: str):
        if LLaDALoader._has_model_files(model_path):
            return

        os.makedirs(model_path, exist_ok=True)
        hf_cmd = shutil.which("hf")
        if hf_cmd is None:
            raise RuntimeError(
                "LLaDA model folder is missing and `hf` CLI is not installed. "
                "Install Hugging Face CLI, then run: "
                f"hf download inclusionAI/LLaDA2.0-Uni --local-dir \"{model_path}\""
            )

        cmd = [hf_cmd, "download", "inclusionAI/LLaDA2.0-Uni", "--local-dir", model_path]
        print(f"[LLaDA] Model not found. Downloading to: {model_path}")
        subprocess.run(cmd, check=True)
        print("[LLaDA] Model download complete.")

    def load(self, attention, dtype, offload, device):
        if not isinstance(device, str) or not device:
            device = "cuda"
        from .model_manager import load_llm
        model_path = self._get_fixed_model_path()
        self._download_model_if_missing(model_path)
        model, tokenizer = load_llm(model_path, device=device,
                                     attention=attention, offload=offload, dtype=dtype)
        return ({
            "model_path": model_path,
            "device": device,
            "attention": attention,
            "dtype": dtype,
            "offload": offload,
        },)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")


# ═══════════════════════════════════════════════════════════════
#  Text-to-Image
# ═══════════════════════════════════════════════════════════════

class LLaDATextToImage:
    """Generate image tokens from text prompt."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("LLaDA_MODEL",),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 32}),
                "cfg_scale": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),
            },
            "optional": {
                "mode": (["standard", "thinking"],),
                "thinking_steps": ("INT", {"default": 32, "min": 1, "max": 64}),
                "thinking_length": ("INT", {"default": 4096, "min": 512, "max": 8192, "step": 512}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**32 - 1}),
                "use_sprint": ("BOOLEAN", {"default": False}),
                "sprint_block_length": ("INT", {"default": 32, "min": 1, "max": 128}),
                "sprint_keep_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "unload_after": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("LLaDA_TOKENS", "STRING",)
    RETURN_NAMES = ("tokens", "thinking_output",)
    FUNCTION = "generate"
    CATEGORY = "LLaDA 2.0-Uni"
    DESCRIPTION = "Generate VQ image tokens from text prompt."

    def generate(self, model, prompt, width, height, steps, cfg_scale,
                 mode="standard", thinking_steps=32, thinking_length=4096,
                 seed=-1, use_sprint=False, sprint_block_length=32,
                 sprint_keep_ratio=0.5, unload_after=True):

        if seed >= 0:
            torch.manual_seed(seed)

        from .model_manager import load_llm, unload_llm
        llm, tokenizer = load_llm(model["model_path"], model["device"],
                                   model.get("attention", "flash_attn"),
                                   model.get("offload", False),
                                   model.get("dtype", "bf16"))

        kwargs = {
            "prompt": prompt,
            "image_h": height,
            "image_w": width,
            "steps": steps,
            "cfg_scale": cfg_scale,
        }

        if mode == "thinking":
            kwargs.update({
                "mode": "thinking",
                "thinking_steps": thinking_steps,
                "thinking_gen_length": thinking_length,
            })

        if use_sprint:
            kwargs.update({
                "use_sprint": True,
                "block_length": sprint_block_length,
                "keep_ratio": sprint_keep_ratio,
                "cache_warmup_steps": 1,
            })

        result = llm.generate_image(**kwargs)
        thinking_text = result.get("thinking", "")

        token_data = {
            "token_ids": result["token_ids"],
            "h": result["h"],
            "w": result["w"],
            "model_path": model["model_path"],
        }

        # ── Auto VRAM cleanup ──
        if unload_after:
            unload_llm()

        return (token_data, thinking_text,)


# ═══════════════════════════════════════════════════════════════
#  Image Understanding (VQA)
# ═══════════════════════════════════════════════════════════════

class LLaDAImageUnderstanding:
    """Understand/describe an image using VQA."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("LLaDA_MODEL",),
                "image": ("IMAGE",),
                "question": ("STRING", {"multiline": True, "default": "Describe this image in detail."}),
            },
            "optional": {
                "gen_steps": ("INT", {"default": 32, "min": 1, "max": 64}),
                "gen_length": ("INT", {"default": 2048, "min": 256, "max": 8192, "step": 256}),
                "use_sprint": ("BOOLEAN", {"default": False}),
                "sprint_threshold": ("FLOAT", {"default": 0.93, "min": 0.5, "max": 0.99, "step": 0.01}),
                "unload_after": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "understand"
    CATEGORY = "LLaDA 2.0-Uni"
    DESCRIPTION = "Ask questions about an image (VQA)."

    def understand(self, model, image, question, gen_steps=32, gen_length=2048,
                   use_sprint=False, sprint_threshold=0.93, unload_after=True):

        from .model_manager import load_llm, get_image_tokenizer, unload_llm
        llm, _ = load_llm(model["model_path"], model["device"],
                           model.get("attention", "flash_attn"),
                           model.get("offload", False),
                           model.get("dtype", "bf16"))

        # ComfyUI tensor (B,H,W,C) → PIL
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_np)

        # Encode image
        image_tokenizer = get_image_tokenizer(model["model_path"], model["device"])
        from decoder.smart_img_process import smart_resize_images
        pil_image = smart_resize_images([pil_image])[0]
        info = image_tokenizer.encode_with_info(pil_image)
        image_tokens = [x + llm.config.image_token_offset for x in info["token_ids"]]
        _, h, w = info["grid_thw"]

        kwargs = {
            "image_tokens": image_tokens,
            "image_h": h, "image_w": w,
            "question": question,
            "steps": gen_steps,
            "gen_length": gen_length,
        }
        if use_sprint:
            kwargs.update({
                "use_sprint": True,
                "threshold": sprint_threshold,
                "keep_ratio": 0.5,
                "cache_warmup_steps": 1,
                "image_keep_ratio": 1.0,
                "text_keep_ratio": 1.0,
            })

        response = llm.understand_image(**kwargs)

        if unload_after:
            unload_llm()

        return (response,)


# ═══════════════════════════════════════════════════════════════
#  Image Editing
# ═══════════════════════════════════════════════════════════════

class LLaDAImageEditing:
    """Edit an image based on a text instruction."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("LLaDA_MODEL",),
                "image": ("IMAGE",),
                "instruction": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "steps": ("INT", {"default": 8, "min": 1, "max": 32}),
                "cfg_text_scale": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "cfg_image_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**32 - 1}),
                "unload_after": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("LLaDA_TOKENS",)
    RETURN_NAMES = ("tokens",)
    FUNCTION = "edit"
    CATEGORY = "LLaDA 2.0-Uni"
    DESCRIPTION = "Edit an image with a text instruction. Connect output to Decode node."

    def edit(self, model, image, instruction, steps=8, cfg_text_scale=4.0,
             cfg_image_scale=0.0, seed=-1, unload_after=True):

        if seed >= 0:
            torch.manual_seed(seed)

        from .model_manager import load_llm, get_image_tokenizer, unload_llm
        llm, _ = load_llm(model["model_path"], model["device"],
                           model.get("attention", "flash_attn"),
                           model.get("offload", False),
                           model.get("dtype", "bf16"))

        # ComfyUI tensor → PIL
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_np)

        # Encode source image
        image_tokenizer = get_image_tokenizer(model["model_path"], model["device"])
        from decoder.utils import generate_crop_size_list, var_center_crop
        crop_size_list = generate_crop_size_list((512 // 32) ** 2, 32)
        pil_image = var_center_crop(pil_image, crop_size_list=crop_size_list)
        info = image_tokenizer.encode_with_info(pil_image)
        image_tokens = [x + llm.config.image_token_offset for x in info["token_ids"]]
        _, h, w = info["grid_thw"]

        result = llm.edit_image(
            image_tokens, h, w, instruction,
            steps=steps,
            cfg_text_scale=cfg_text_scale,
            cfg_image_scale=cfg_image_scale,
        )

        token_data = {
            "token_ids": result["token_ids"],
            "h": result["h"],
            "w": result["w"],
            "model_path": model["model_path"],
        }

        if unload_after:
            unload_llm()

        return (token_data,)


# ═══════════════════════════════════════════════════════════════
#  Token Decoder
# ═══════════════════════════════════════════════════════════════

class LLaDAImageDecode:
    """Decode VQ tokens from LLaDA into a pixel image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tokens": ("LLaDA_TOKENS",),
            },
            "optional": {
                "decode_mode": (["normal", "decoder-turbo"],),
                "decoder_steps": ("INT", {"default": 50, "min": 1, "max": 100}),
                "resolution_multiplier": ("INT", {"default": 2, "min": 1, "max": 4}),
                "unload_after": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "decode"
    CATEGORY = "LLaDA 2.0-Uni"
    DESCRIPTION = "Decode VQ tokens to pixel image. Use decoder-turbo for ~10× speed."

    def decode(self, tokens, decode_mode="decoder-turbo", decoder_steps=50,
               resolution_multiplier=2, unload_after=True):

        from .model_manager import decode_tokens, unload_all

        num_steps = 8 if decode_mode == "decoder-turbo" else decoder_steps

        pil_image = decode_tokens(
            tokens["token_ids"],
            tokens["h"],
            tokens["w"],
            tokens["model_path"],
            device="cuda",
            num_steps=num_steps,
            decode_mode=decode_mode,
            resolution_multiplier=resolution_multiplier,
        )

        # PIL → ComfyUI tensor (1, H, W, 3) float32 [0,1]
        img_np = np.array(pil_image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_np).unsqueeze(0)

        # ── Auto VRAM cleanup after full pipeline ──
        if unload_after:
            unload_all()

        return (tensor,)


# ═══════════════════════════════════════════════════════════════
#  Manual Unload
# ═══════════════════════════════════════════════════════════════

class LLaDAUnloadModel:
    """Manually unload all LLaDA 2.0-Uni components and free VRAM."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {},
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "unload"
    CATEGORY = "LLaDA 2.0-Uni"
    DESCRIPTION = "Unload all LLaDA 2.0-Uni models from VRAM."

    def unload(self):
        from .model_manager import unload_all
        unload_all()
        print("[LLaDA] ✅ All models unloaded, VRAM freed")
        return ()
