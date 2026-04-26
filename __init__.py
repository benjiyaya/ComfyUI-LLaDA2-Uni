"""
ComfyUI Custom Nodes for LLaDA 2.0-Uni
Unified multimodal: Text-to-Image, Image Understanding (VQA), Image Editing
"""

from .nodes import (
    LLaDALoader,
    LLaDATextToImage,
    LLaDAImageUnderstanding,
    LLaDAImageEditing,
    LLaDAImageDecode,
    LLaDAUnloadModel,
)

NODE_CLASS_MAPPINGS = {
    "LLaDALoader": LLaDALoader,
    "LLaDATextToImage": LLaDATextToImage,
    "LLaDAImageUnderstanding": LLaDAImageUnderstanding,
    "LLaDAImageEditing": LLaDAImageEditing,
    "LLaDAImageDecode": LLaDAImageDecode,
    "LLaDAUnloadModel": LLaDAUnloadModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLaDALoader": "LLaDA 2.0-Uni Loader",
    "LLaDATextToImage": "LLaDA 2.0-Uni Text-to-Image",
    "LLaDAImageUnderstanding": "LLaDA 2.0-Uni Image Understanding",
    "LLaDAImageEditing": "LLaDA 2.0-Uni Image Editing",
    "LLaDAImageDecode": "LLaDA 2.0-Uni Token Decoder",
    "LLaDAUnloadModel": "LLaDA 2.0-Uni Unload Model",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
