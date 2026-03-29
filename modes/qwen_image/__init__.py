from PIL import Image

from .pipeline import get_pipeline_qwen_image


cfg_param_key = "true_cfg_scale"


def get_pipeline(*args, **kwargs):
	return get_pipeline_qwen_image(*args, **kwargs)


def is_edit_model(vit_model_table: dict, vit_key: str) -> bool:
	return bool(vit_model_table.get(vit_key, {}).get("edit", False))


def build_params(gen_mode: str, base_params: dict, extra: dict) -> dict:
	if gen_mode == "t2i":
		return base_params
	if gen_mode == "i2i":
		params = {**base_params, "image": extra["image"]}
		if extra["is_edit_model"]:
			params["consistency_strength"] = extra["consistency_strength"]
		else:
			params["strength"] = extra["strength"]
		return params
	if gen_mode == "inpaint":
		params = {
			**base_params,
			"image": extra["image"],
			"mask_image": extra["mask_image"],
			"strength": extra["strength"],
		}
		if extra["is_edit_model"]:
			params["consistency_strength"] = extra["consistency_strength"]
		return params
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
	if is_edit_model:
		if strength != 1.0:
			notify("Strength is not used for Edit models. Ignoring it.")
		if not resize_before_i2i and consistency_strength != 0.0:
			notify(
				"For consistency strength, input image must be resized. Setting consistency strength to 0.0."
			)
			consistency_strength = 0.0
	else:
		if not resize_before_i2i:
			notify("Normal i2i models always resize input. Strength is set to 1.0.")
			resize_before_i2i = True
			strength = 1.0
		if consistency_strength != 0.0:
			notify("Consistency strength is only supported for Edit models. Ignoring it.")

	if resize_before_i2i and is_edit_model:  # Normal i2i models resize in their pipe
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
	if consistency_strength != 0.0 and not is_edit_model:
		notify("Consistency strength is only supported for Edit models. Ignoring it.")
		consistency_strength = 0.0
	if is_edit_model:  # Normal inpaint models resize in their pipe
		input_image = input_image.resize((width, height), Image.Resampling.LANCZOS)
		mask_image = mask_image.resize((width, height), Image.Resampling.LANCZOS)
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
	"is_edit_model",
	"build_params",
	"prepare_i2i_inputs",
	"prepare_inpaint_inputs",
]
