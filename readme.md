# GUSUQ WebUI – Gradio-based Unified Simple UI for Qwen-image with Nunchaku

## Offloading Option

"opt_policy" parameter in `config/opt_pol.yaml` allows you to choose different offloading strategies to optimize VRAM usage based on your hardware capabilities:

- low_vram: sequentially offloads both the text encoder and DiT to CPU when not in use. Minimal VRAM usage. It works even with < 16GB VRAM.
- mid_vram: model-wise offloading both the text encoder and DiT to CPU when not in use. Balances VRAM usage and performance. 16GB VRAM required.
- high_vram: Offloads the text encoder to CPU when not in use. 18GB VRAM required. If you cannot run with high_vram, try mid_vram or low_vram.