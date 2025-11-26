# GUSUQ WebUI – Gradio-based Unified Simple UI for Qwen-image with Nunchaku

## Installation

1. Install system dependencies:
    - Install git, CUDA
    - Install python and build tools
        - Linux (Ubuntu) / WSL:
            ```bash
            sudo apt update
            sudo apt install build-essential python3-dev # python 3.12 for Ubuntu 24.04
            # sudo apt install build-essential python3.13 python3.13-venv python3.13-dev # other python versions (example: python 3.13). Deadsnakes PPA may be needed.
            ```

2. Clone this repository:

    ```bash
    git clone https://github.com/Ai-tensa/GUSUQ-WebUI.git
    cd GUSUQ-WebUI
    python3 -m venv {venv_name} # Create a virtual environment (optional. other methods like conda are also fine)
    source {venv_name}/bin/activate # Activate the virtual environment (Linux/Mac)
    .\{venv_name}\Scripts\activate # Activate the virtual environment (Windows)
    ```
3. Install the required dependencies:
    1. Install pytorch and nunchaku according to your CUDA version and Python version from [pytorch.org](https://pytorch.org/get-started/locally/) and [nunchaku releases](https://github.com/nunchaku-tech/nunchaku). I tested pytorch 2.8.0 and nunchaku 1.0.2, but you can try other versions as well. 

        Note:
        If you use Linux/WSL, python 3.12 and CUDA 12.8, you can use `requirements.txt` with just uncomment (skip this step).

        Example for CUDA 12.6 and Python 3.13:
        ```bash
        pip install torch==2.8.0+cu126 torchvision==0.23.0+cu126 torchaudio==2.8.0 --extra-index-url https://download.pytorch.org/whl/cu126
        ```
        Then install nunchaku for Linux and Python 3.13:
        ```bash
        pip install https://github.com/nunchaku-tech/nunchaku/releases/download/v1.0.2/nunchaku-1.0.2+torch2.8-cp313-cp313-linux_x86_64.whl
        ```
    2. Install other dependencies
        ```bash
        pip install -r requirements.txt
        ```
    3. AutoAWQ installation (Linux only?):
        ```bash
        pip install autoawq==0.2.9
        ```
        Windows installation guide: [AutoAWQ Windows Installation Guide](https://github.com/casper-hansen/AutoAWQ/issues/704) (but I haven't tested it yet)
    4. Desable AutoAWQ of Qwen-VL-7B-Instruct model in `constants.py` if you cannot install AutoAWQ:
        ```python
        # TEXT_ENCODER_ID = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ" # 7GB model with AutoAWQ)
        TEXT_ENCODER_ID = "Qwen/Qwen2.5-VL-7B-Instruct" # 14GB, but "low_vram" offloading works well even less than 16GB VRAM
        ```
    5. Launch the web UI:
        ```bash
        python webui.py (--user-config-yaml path/to/your_config.yaml ...) # RTX 40xx orl older GPUs
        # python webui.py  --vit-models-yaml config/vit_models_sample_50xx.yaml (--user-config-yaml path/to/your_config.yaml ...) # for RTX 50xx (Blackwell) GPUs
        (python webui.py --help for more options)
        ```


## Offloading Option

"opt_policy" parameter in `config/opt_pol.yaml` allows you to choose different offloading strategies to optimize VRAM usage based on your hardware capabilities:

- low_vram: sequentially offloads both the text encoder and DiT to CPU when not in use. Minimal VRAM usage. It works even with < 16GB VRAM.
- mid_vram: model-wise offloading both the text encoder and DiT to CPU when not in use. Balances VRAM usage and performance. 16GB VRAM required.
- high_vram: Offloads the text encoder to CPU when not in use. 18GB VRAM required. If you cannot run with high_vram, try mid_vram or low_vram.