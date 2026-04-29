# ComfyUI-LLaDA2-Uni

ComfyUI custom nodes for LLaDA 2.0-Uni: text-to-image, image understanding (VQA), and image editing.

## Official Upstream Links

- Original LLaDA 2.0-Uni GitHub: https://github.com/inclusionAI/LLaDA2.0-Uni
- Official model weights: https://huggingface.co/inclusionAI/LLaDA2.0-Uni
- This ComfyUI custom node repo: https://github.com/benjiyaya/ComfyUI-LLaDA2-Uni
- Model weights FP8 : https://huggingface.co/benjiaiplayground/LLaDA2.0-Uni-FP8

## Nodes

| Node | Description |
|---|---|
| **LLaDA 2.0-Uni Loader** | Load model with attention backend (Flash/SDPA) and CPU offload |
| **LLaDA 2.0-Uni Text-to-Image** | Generate images from text prompts (supports thinking mode + SPRINT) |
| **LLaDA 2.0-Uni Image Understanding** | VQA: ask questions about images |
| **LLaDA 2.0-Uni Image Editing** | Edit images with text instructions |
| **LLaDA 2.0-Uni Token Decoder** | Decode VQ tokens to pixel images (normal or turbo mode) |
| **LLaDA 2.0-Uni Unload Model** | Manually free all VRAM |

## Model Path Behavior

- Loader model path is fixed to: `<ComfyUI>/models/llada2uni`
- Users do not need to set a model path in the node.
- If the folder is missing/empty, loader auto-runs:

```bash
hf download inclusionAI/LLaDA2.0-Uni --local-dir <ComfyUI>/models/llada2uni
```

## Installation Guide

1. Install this custom node:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/benjiyaya/ComfyUI-LLaDA2-Uni
cd ComfyUI-LLaDA2-Uni
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Make sure `hf` CLI is available (used for auto-download if model folder is missing):

```bash
pip install -U "huggingface_hub[cli]"
```

4. (Optional) Install acceleration backends:

```bash
pip install flash-attn --no-build-isolation
```

## Requirements Notes

- Python 3.10 recommended
- PyTorch: newer versions are supported; it does not need to be exactly 2.4/CUDA 12.4
- `flash-attn` is optional but recommended for best speed

## Workflow Connection Guide

### A) Text-to-Image

`LLaDA Loader` -> `LLaDA Text-to-Image` -> `LLaDA Token Decoder` -> `Preview/Save Image`

### B) Image Understanding

`LLaDA Loader` + `Load Image` -> `LLaDA Image Understanding` -> `Text Output`

### C) Image Editing

`LLaDA Loader` + `Load Image` -> `LLaDA Image Editing` -> `LLaDA Token Decoder` -> `Preview/Save Image`

### Optional Memory Cleanup

- Keep `unload_after` enabled on inference nodes.
- Use `LLaDA Unload Model` node anytime to manually free VRAM.

## License

Apache 2.0.
