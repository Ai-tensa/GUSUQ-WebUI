import torch
from diffusers.models import AutoencoderKLQwenImage
from qwen_image_pipelines import (
    QwenImagePipeline,
    QwenImageImg2ImgPipeline,
    QwenImageInpaintPipeline,
    QwenImageEditPlusPipeline,
    QwenImageEditPlusInpaintPipeline,
)
from constants import (
    SAMPLERS,
    FLOWMATCH_CFG,
    BASE_QWEN_IMAGE_ID,
    BASE_QWEN_IMAGE_EDIT_ID,
    SAME_AS_IMAGE_GENERATION_PIPELINE,
)
from utils import release_memory_resources


def _is_compatible_qwen_vae(vae) -> bool:
    return isinstance(vae, AutoencoderKLQwenImage)


def _load_nunchaku_transformer(pm, model_key: str):
    try:
        from nunchaku.models.transformers.transformer_qwenimage import (
            NunchakuQwenImageTransformer2DModel,
        )
    except ImportError as e:
        raise RuntimeError(
            "nunchaku is required to load Qwen Image ViT checkpoints."
        ) from e

    if pm.transformer is not None:
        if model_key != pm.current_vit:
            print(
                "Please unload previous transformer model first before loading a new one."
            )
            raise RuntimeError("Previous transformer model not unloaded.")
        return

    if model_key is None:
        model_key = list(pm.vit_model_table.keys())[0]

    pm.current_vit = model_key
    pm.transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
        pm.vit_model_table[model_key]["path"],
        low_cpu_mem_usage=True,
    )
    release_memory_resources()


def _resolve_base_model_key(pm, base_model_key: str = None) -> str:
    if isinstance(base_model_key, str) and base_model_key not in ("", "Default", "None"):
        return base_model_key

    for name, cfg in pm.base_pipeline_table.items():
        if cfg.get("arch") == "Qwen-Image":
            return name
    return BASE_QWEN_IMAGE_ID


def get_pipeline_qwen_image(
    pm,
    vit_model_key: str,
    sampler_name: str,
    pipe_mode: str = "t2i",
    vlm_model_key: str = None,
    base_model_key: str = None,
    patch_encode_prompt_fn=None,
    override_device_property_fn=None,
):
    use_nunchaku_vit = vit_model_key != SAME_AS_IMAGE_GENERATION_PIPELINE
    resolved_base_model_key = _resolve_base_model_key(pm, base_model_key)
    base_cfg = pm.base_pipeline_table.get(resolved_base_model_key, {})
    if use_nunchaku_vit:
        is_edit_model = bool(pm.vit_model_table.get(vit_model_key, {}).get("edit", False))
    else:
        is_edit_model = bool(base_cfg.get("edit", False))
    image_base_id = base_cfg.get("image_repo", BASE_QWEN_IMAGE_ID)
    edit_base_id = base_cfg.get("edit_repo", BASE_QWEN_IMAGE_EDIT_ID)
    # First load
    if pm.pipes == {}:
        del_vlm = vlm_model_key != pm.current_vlm
        del_vit = (
            vit_model_key != pm.current_vit
            or resolved_base_model_key != pm.current_base_model
        )
        del_vae = (
            resolved_base_model_key != pm.current_base_model
            or (pm.vae is not None and not _is_compatible_qwen_vae(pm.vae))
        )
        if del_vlm or del_vit or del_vae:
            pm.clear_pipelines(del_vlm=del_vlm, del_vit=del_vit, del_vae=del_vae)
            release_memory_resources()
        pm._load_vlm(vlm_model_key)
        if use_nunchaku_vit:
            _load_nunchaku_transformer(pm, vit_model_key)
        else:
            pm.current_vit = SAME_AS_IMAGE_GENERATION_PIPELINE
        scheduler = SAMPLERS[sampler_name].from_config(FLOWMATCH_CFG)
        params = {
            "pretrained_model_name_or_path": image_base_id,
            "text_encoder": pm.text_encoder,
            "tokenizer": pm.tokenizer,
            "scheduler": scheduler,
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
        }
        if use_nunchaku_vit:
            params["transformer"] = pm.transformer
        if pm.vae is None:
            pm.pipes["t2i"] = QwenImagePipeline.from_pretrained(**params)
            pm.vae = pm.pipes["t2i"].vae
        else:
            params["vae"] = pm.vae
            pm.pipes["t2i"] = QwenImagePipeline.from_pretrained(**params)
        pm.current_vit = vit_model_key
        pm.current_base_model = resolved_base_model_key
        pm.pipes["i2i"] = QwenImageImg2ImgPipeline.from_pretrained(
            image_base_id, **pm.pipes["t2i"].components
        )
        pm.pipes["i2i_edit"] = QwenImageEditPlusPipeline.from_pretrained(
            edit_base_id,
            vision_processor=pm.vision_processor,
            **pm.pipes["t2i"].components,
        )
        pm.pipes["inpaint"] = QwenImageInpaintPipeline.from_pretrained(
            image_base_id, **pm.pipes["t2i"].components
        )
        pm.pipes["inpaint_edit"] = QwenImageEditPlusInpaintPipeline.from_pretrained(
            edit_base_id, **pm.pipes["i2i_edit"].components
        )

        opt_policy = pm.opt_pol_cfg.get("opt_policy", None)
        if patch_encode_prompt_fn is not None:
            for _pipe in pm.pipes.values():
                patch_encode_prompt_fn(_pipe, opt_policy)
        if pm.opt_pol_cfg.get("enable_vae_slicing", True):
            pm.vae.enable_slicing()
        if pm.opt_pol_cfg.get("enable_vae_tiling", False):
            pm.vae.enable_tiling(
                tile_sample_min_height=pm.opt_pol_cfg.get(
                    "vae_tile_sample_min_height", None
                ),
                tile_sample_min_width=pm.opt_pol_cfg.get(
                    "vae_tile_sample_min_width", None
                ),
                tile_sample_stride_height=pm.opt_pol_cfg.get(
                    "vae_tile_sample_stride_height", None
                ),
                tile_sample_stride_width=pm.opt_pol_cfg.get(
                    "vae_tile_sample_stride_width", None
                ),
            )

        if opt_policy == "no_offload":
            pm.pipes["i2i_edit"].to("cuda")
            print("No offloading applied.")
        elif opt_policy == "high_vram":
            pm.transformer.to("cuda")
            pm.vae.to("cuda")
            if override_device_property_fn is not None:
                for _pipe in pm.pipes.values():
                    override_device_property_fn(_pipe)
            print("Enabled high vram setting for offloading.")
        elif opt_policy == "mid_vram":
            pm.pipes["t2i"].enable_model_cpu_offload()
            print("Enabled medium vram setting for offloading.")
        elif opt_policy == "low_vram":
            if use_nunchaku_vit:
                pm.transformer.set_offload(
                    True, use_pin_memory=False, num_blocks_on_gpu=1
                )
                pm.pipes["t2i"]._exclude_from_cpu_offload.append("transformer")
            pm.pipes["t2i"].enable_sequential_cpu_offload()
            print("Enabled low vram setting for offloading.")
        else:
            print(f"Unknown opt_policy: {opt_policy}")
            print("Available options: high_vram | mid_vram | low_vram")
        pm.is_set_te_offload = True

        release_memory_resources()
        if pipe_mode == "t2i":
            return pm.pipes["t2i"]
        elif pipe_mode == "i2i":
            return pm.pipes["i2i_edit"] if is_edit_model else pm.pipes["i2i"]
        else:
            return pm.pipes["inpaint_edit"] if is_edit_model else pm.pipes["inpaint"]

    # Model switch
    if (
        vlm_model_key != pm.current_vlm
        or vit_model_key != pm.current_vit
        or resolved_base_model_key != pm.current_base_model
    ):
        del_vlm = vlm_model_key != pm.current_vlm
        del_vit = (
            vit_model_key != pm.current_vit
            or resolved_base_model_key != pm.current_base_model
        )
        del_vae = (
            resolved_base_model_key != pm.current_base_model
            or (pm.vae is not None and not _is_compatible_qwen_vae(pm.vae))
        )
        pm.clear_pipelines(del_vlm=del_vlm, del_vit=del_vit, del_vae=del_vae)

        return pm.get_pipeline(
            "Qwen Image",
            vit_model_key,
            sampler_name,
            pipe_mode,
            vlm_model_key=vlm_model_key,
            base_model_key=base_model_key,
        )

    # Sampler switch
    if pm.pipes["t2i"].scheduler.__class__ is not SAMPLERS[sampler_name]:
        pm.pipes["t2i"].scheduler = SAMPLERS[sampler_name].from_config(FLOWMATCH_CFG)
        pm.pipes["i2i"].scheduler = pm.pipes["t2i"].scheduler
        pm.pipes["i2i_edit"].scheduler = pm.pipes["t2i"].scheduler
        pm.pipes["inpaint"].scheduler = pm.pipes["t2i"].scheduler
        pm.pipes["inpaint_edit"].scheduler = pm.pipes["t2i"].scheduler
        release_memory_resources()
    if pipe_mode == "t2i":
        return pm.pipes["t2i"]
    elif pipe_mode == "i2i":
        return pm.pipes["i2i_edit"] if is_edit_model else pm.pipes["i2i"]
    else:
        return pm.pipes["inpaint_edit"] if is_edit_model else pm.pipes["inpaint"]
