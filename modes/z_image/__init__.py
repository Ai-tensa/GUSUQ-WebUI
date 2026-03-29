from PIL import Image

from .pipeline import get_pipeline_z_image


cfg_param_key = "guidance_scale"


def get_pipeline(*args, **kwargs):
	return get_pipeline_z_image(*args, **kwargs)


def build_params(gen_mode: str, base_params: dict, extra: dict) -> dict:
	if gen_mode == "t2i":
		return base_params
	if gen_mode == "i2i":
		return {**base_params, "image": extra["image"], "strength": extra["strength"]}
	if gen_mode == "inpaint":
		control_image = extra.get("control_image") or extra["image"]
		return {
			**base_params,
			"image": extra["image"],
			"mask_image": extra["mask_image"],
			"control_image": control_image,
		}
	raise RuntimeError(f"Unsupported gen_mode: {gen_mode}")


def prepare_i2i_inputs(
	input_image,
	width: int,
	height: int,
	resize_before_i2i: bool,
	strength: float,
	consistency_strength: float,
	is_edit_model: bool,
	notify,
) -> dict:
	if consistency_strength != 0.0:
		notify("Consistency strength is not supported in the current mode. Ignoring it.")
		consistency_strength = 0.0
	if resize_before_i2i:
		input_image = input_image.resize((width, height), Image.Resampling.LANCZOS)
	return {
		"input_image": input_image,
		"resize_before_i2i": resize_before_i2i,
		"strength": strength,
		"consistency_strength": consistency_strength,
	}


def prepare_inpaint_inputs(
	input_image,
	mask_image,
	width: int,
	height: int,
	strength: float,
	consistency_strength: float,
	is_edit_model: bool,
	notify,
) -> dict:
	if consistency_strength != 0.0:
		notify("Consistency strength is not supported in the current mode. Ignoring it.")
		consistency_strength = 0.0
	if strength != 1.0:
		notify("Denoising strength is not supported in the current mode. Ignoring it.")
	return {
		"input_image": input_image,
		"mask_image": mask_image,
		"strength": strength,
		"consistency_strength": consistency_strength,
		"control_image": input_image,
	}


__all__ = [
	"cfg_param_key",
	"get_pipeline",
	"build_params",
	"prepare_i2i_inputs",
	"prepare_inpaint_inputs",
]
