import torch
import gradio as gr
from types import MethodType
from diffusers.pipelines.z_image import (
    ZImagePipeline,
    ZImageImg2ImgPipeline,
)
from transformers import Qwen3VLProcessor
from constants import (
    SAMPLERS,
    FLOWMATCH_CFG,
    BASE_ZIMAGE_ID,
    BASE_QWEN3_VL_ID,
    SAME_AS_IMAGE_GENERATION_PIPELINE,
)
from utils import release_memory_resources


def _resolve_base_model_key(pm, base_model_key: str = None) -> str:
    if isinstance(base_model_key, str) and base_model_key not in ("", "Default", "None"):
        return base_model_key

    for name, cfg in pm.base_pipeline_table.items():
        if cfg.get("arch") == "Z-Image":
            return name
    return BASE_ZIMAGE_ID


def _load_nunchaku_transformer(pm, model_key: str):
    try:
        from nunchaku.models.transformers.transformer_zimage import (
            NunchakuZImageRopeHook,
            NunchakuZImageTransformer2DModel,
        )
    except ImportError as e:
        raise RuntimeError(
            "nunchaku is required to load Z-Image ViT checkpoints."
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
    pm.transformer = NunchakuZImageTransformer2DModel.from_pretrained(
        pm.vit_model_table[model_key]["path"],
    )

    if not getattr(pm.transformer, "_gusuq_forward_compat_patched", False):
        def _forward_compat(
            self,
            x,
            t,
            cap_feats,
            return_dict: bool = True,
            controlnet_block_samples=None,
            siglip_feats=None,
            image_noise_mask=None,
            patch_size: int = 2,
            f_patch_size: int = 1,
        ):
            rope_hook = NunchakuZImageRopeHook()
            self.register_rope_hook(rope_hook)
            try:
                return super(type(self), self).forward(
                    x,
                    t,
                    cap_feats,
                    return_dict=return_dict,
                    controlnet_block_samples=controlnet_block_samples,
                    siglip_feats=siglip_feats,
                    image_noise_mask=image_noise_mask,
                    patch_size=patch_size,
                    f_patch_size=f_patch_size,
                )
            finally:
                self.unregister_rope_hook()
                del rope_hook

        pm.transformer.forward = MethodType(_forward_compat, pm.transformer)
        pm.transformer._gusuq_forward_compat_patched = True

    release_memory_resources()


def get_pipeline_z_image(
    pm,
    vit_model_key: str,
    sampler_name: str,
    pipe_mode: str = "t2i",
    vlm_model_key: str = None,
    base_model_key: str = None,
    patch_encode_prompt_fn=None,
    override_device_property_fn=None,
):
    if pipe_mode not in ("t2i", "i2i"):
        gr.Error(f"Unsupported pipe_mode '{pipe_mode}' for Z-Image pipeline.")
        raise RuntimeError("Invalid pipe_mode for Z-Image pipeline.")

    use_nunchaku_vit = vit_model_key != SAME_AS_IMAGE_GENERATION_PIPELINE
    resolved_base_model_key = _resolve_base_model_key(pm, base_model_key)
    base_cfg = pm.base_pipeline_table.get(resolved_base_model_key, {})
    pipeline_id = base_cfg.get("pipeline_id", resolved_base_model_key)
    # First load
    if pm.pipes == {}:
        params = {
            "pretrained_model_name_or_path": pipeline_id,
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
        }
        if use_nunchaku_vit:
            _load_nunchaku_transformer(pm, vit_model_key)
            params["transformer"] = pm.transformer
        else:
            pm.current_vit = SAME_AS_IMAGE_GENERATION_PIPELINE

        if vlm_model_key != SAME_AS_IMAGE_GENERATION_PIPELINE:
            pm._load_vlm(vlm_model_key)
            params["text_encoder"] = pm.text_encoder
            params["tokenizer"] = pm.tokenizer
        else:
            pm.current_vlm = vlm_model_key
        pm.pipes["t2i"] = ZImagePipeline.from_pretrained(**params)
        pm.pipes["i2i"] = ZImageImg2ImgPipeline.from_pretrained(
            pipeline_id,
            **pm.pipes["t2i"].components,
        )
        pm.text_encoder = pm.pipes["t2i"].text_encoder
        pm.tokenizer = pm.pipes["t2i"].tokenizer
        pm.transformer = pm.pipes["t2i"].transformer
        pm.current_vit = vit_model_key
        pm.current_base_model = resolved_base_model_key
        if getattr(pm.pipes["t2i"], "vision_processor", None) is None:
            pm.vision_processor = Qwen3VLProcessor.from_pretrained(
                BASE_QWEN3_VL_ID,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
        else:
            pm.vision_processor = pm.pipes["t2i"].vision_processor

        pm.vae = pm.pipes["t2i"].vae

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
            pm.pipes["t2i"].to("cuda")
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
            pm.pipes["t2i"].enable_sequential_cpu_offload()
            print("Enabled low vram setting for offloading.")
        else:
            print(f"Unknown opt_policy: {opt_policy}")
            print("Available options: high_vram | mid_vram | low_vram")
        pm.is_set_te_offload = True

        release_memory_resources()
        return pm.pipes[pipe_mode]

    # Model switch
    if (
        vlm_model_key != pm.current_vlm
        or vit_model_key != pm.current_vit
        or resolved_base_model_key != pm.current_base_model
    ):
        del_vlm = vlm_model_key != pm.current_vlm
        del_vit = vit_model_key != pm.current_vit
        del_vae = resolved_base_model_key != pm.current_base_model
        pm.clear_pipelines(del_vlm=del_vlm, del_vit=del_vit or del_vae, del_vae=del_vae)

        return pm.get_pipeline(
            "Z-Image",
            vit_model_key,
            sampler_name,
            pipe_mode,
            vlm_model_key=vlm_model_key,
            base_model_key=resolved_base_model_key,
        )

    # Sampler switch
    if pm.pipes["t2i"].scheduler.__class__ is not SAMPLERS[sampler_name]:
        scheduler = SAMPLERS[sampler_name].from_config(FLOWMATCH_CFG)
        pm.pipes["t2i"].scheduler = scheduler
        pm.pipes["i2i"].scheduler = scheduler

    return pm.pipes[pipe_mode]
