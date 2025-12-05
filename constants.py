import math
from diffusers import FlowMatchEulerDiscreteScheduler

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


BASE_QWEN_IMAGE_ID = "Qwen/Qwen-Image"
BASE_QWEN_IMAGE_EDIT_ID = "Qwen/Qwen-Image-Edit-2509"