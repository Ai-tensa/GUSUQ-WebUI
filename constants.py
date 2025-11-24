from typing import Any
import math
from diffusers import FlowMatchEulerDiscreteScheduler
from pathlib import Path
import yaml

SAMPLERS = {
    "FlowMatchEuler": FlowMatchEulerDiscreteScheduler,  # FlowMatch Euler is only supported sampler for Qwen-Image at the moment
}


# From https://github.com/ModelTC/Qwen-Image-Lightning/blob/342260e8f5468d2f24d084ce04f55e101007118b/generate_with_diffusers.py#L82C9-L97C10
FLOWMATCH_CFG = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}


TEXT_ENCODER_ID = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
DEFAULT_TEXT_ENCODER_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_QWEN_IMAGE_ID = "Qwen/Qwen-Image"
DEFAULT_QWEN_IMAGE_EDIT_ID = "Qwen/Qwen-Image-Edit-2509"


def load_vit_model_table(yaml_path: Path) -> dict[str, dict[str, Any]]:
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f) or []
    table = {}
    for item in raw:
        name = item["name"]
        table[name] = {
            "path": item["path_or_url"],
            "edit": bool(item.get("edit", False)),
        }
    return table
