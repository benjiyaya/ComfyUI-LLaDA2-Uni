# ComfyUI-LLaDA2-Uni

ComfyUI custom nodes for LLaDA 2.0-Uni: text-to-image, image understanding (VQA), and image editing.

## Official Upstream Links

- Original LLaDA 2.0-Uni GitHub: https://github.com/inclusionAI/LLaDA2.0-Uni
- Official model weights: https://huggingface.co/inclusionAI/LLaDA2.0-Uni
- This ComfyUI custom node repo: https://github.com/benjiyaya/ComfyUI-LLaDA2-Uni

## Nodes

| Node | Description |
|---|---|
| **LLaDA 2.0-Uni Loader** | Load model with attention backend (Flash/SDPA/Sage) and CPU offload |
| **LLaDA 2.0-Uni Text-to-Image** | Generate images from text prompts (supports thinking mode + SPRINT) |
| **LLaDA 2.0-Uni Image Understanding** | VQA: ask questions about images |
| **LLaDA 2.0-Uni Image Editing** | Edit images with text instructions |
| **LLaDA 2.0-Uni Token Decoder** | Decode VQ tokens to pixel images (normal or turbo mode) |
| **LLaDA 2.0-Uni Unload Model** | Manually free all VRAM |


<img width="1093" height="630" alt="Screenshot 2026-04-29 023237" src="https://github.com/user-attachments/assets/be44a6d3-a0d9-4237-b2c8-60698b8bdc0a" />

<img width="2019" height="1058" alt="Screenshot 2026-04-29 033552" src="https://github.com/user-attachments/assets/ccb6a046-63e0-462a-8754-737f04e6fc86" />


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
pip install sageattention
```

## Requirements Notes

- Python 3.10 recommended
- PyTorch: newer versions are supported; it does not need to be exactly 2.4/CUDA 12.4
- `flash-attn` is optional but recommended for best speed
- `sageattention` is optional for `sage_attn` mode

## Workflow Connection Guide

### A) Text-to-Image

`LLaDA Loader` -> `LLaDA Text-to-Image` -> `LLaDA Token Decoder` -> `Preview/Save Image`

### B) Image Understanding

`LLaDA Loader` + `Load Image` -> `LLaDA Image Understanding` -> `Text Output`

### C) Image Editing

`LLaDA Loader` + `Load Image` -> `LLaDA Image Editing` -> `LLaDA Token Decoder` -> `Preview/Save Image`

### Optional Memory Cleanup

- Keep `unload_after` enabled on inference nodes (default recommended).
- Use `LLaDA Unload Model` node anytime to manually free VRAM.

## License

Apache 2.0.
