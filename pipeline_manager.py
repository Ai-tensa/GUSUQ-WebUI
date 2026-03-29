from importlib import import_module
import torch
from types import MethodType
from transformers import (
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
    Qwen3VLForConditionalGeneration,
    Qwen3VLProcessor,
    Qwen3Model,
)
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    DiffusionPipeline,
)
from constants import (
    SAMPLERS,
    FLOWMATCH_CFG,
    BASE_QWEN3_VL_ID,
    SAME_AS_IMAGE_GENERATION_PIPELINE,
)
from modes import ModeAdapter, load_mode_adapters
from utils import release_memory_resources


def build_scheduler(name: str, base_cfg):
    if base_cfg is None:
        return FlowMatchEulerDiscreteScheduler.from_config(FLOWMATCH_CFG)
    return SAMPLERS[name].from_config(FLOWMATCH_CFG)


def _override_device_property(pipe):
    if getattr(pipe.__class__, "_device_overridden", False):
        return

    _orig_fget = type(pipe).device.fget

    def _vit_first(self):
        # priority to transformer (ViT)
        return (
            self.transformer.device
            if hasattr(self, "transformer")
            else _orig_fget(self)
        )

    pipe.__class__.device = property(_vit_first)
    pipe.__class__._device_overridden = True


def patch_encode_prompt(pipe, opt_policy):
    original_encode = pipe.encode_prompt.__func__

    def encode_prompt_cast(self, *args, **kwargs):
        # high VRAM setting moves the model; others only match dtype without moving
        tgt_dtype = self.transformer.dtype
        if opt_policy == "high_vram":
            # Move to GPU
            tgt_dev = self.transformer.device
            self.text_encoder.to(tgt_dev, non_blocking=True)
            # Encode with correct device/dtype
            kwargs["device"] = tgt_dev
            embeds, mask = original_encode(self, *args, **kwargs)
            embeds = embeds.to(device=tgt_dev, dtype=tgt_dtype, non_blocking=True)
            if mask is not None and mask.device != tgt_dev:
                mask = mask.to(tgt_dev, non_blocking=True)

            # Move back to CPU
            self.text_encoder.to("cpu", non_blocking=True)
            torch.cuda.empty_cache()
            return embeds, mask
        else:
            embeds, mask = original_encode(self, *args, **kwargs)
            if not isinstance(embeds, torch.Tensor):
                return embeds, mask
            embeds = embeds.to(dtype=tgt_dtype)
            return embeds, mask

    pipe.encode_prompt = MethodType(encode_prompt_cast, pipe)


class PipelineManager:
    def __init__(
        self,
        opt_pol_cfg: dict,
        vlm_model_table: dict,
        vit_model_table: dict,
        base_pipeline_table: dict,
        mode_config: dict,
    ):
        self.pipes: dict[str, DiffusionPipeline] = {}
        self.current_arch_mode = None
        self.current_vlm = None
        self.current_vit = None
        self.current_base_model = None
        self.text_encoder = None
        self.tokenizer = None
        self.vision_processor = None
        self.transformer = None
        self.vae = None
        self.opt_pol_cfg = opt_pol_cfg
        self.vlm_model_table = vlm_model_table
        self.vit_model_table = vit_model_table
        self.base_pipeline_table = base_pipeline_table
        self.mode_config = mode_config
        self.mode_adapters: dict[str, ModeAdapter] = load_mode_adapters(mode_config)
        self.is_set_te_offload = False

    def get_mode_adapter(self, arch_mode: str) -> ModeAdapter:
        adapter = self.mode_adapters.get(arch_mode)
        if adapter is None:
            raise RuntimeError(f"Unsupported arch_mode: {arch_mode}")
        return adapter

    def get_pipeline(
        self,
        arch_mode: str,
        vit_model_key: str,
        sampler_name: str,
        pipe_mode: str = "t2i",
        vlm_model_key: str = None,
        base_model_key: str = None,
    ):
        self._switch_arch_mode(arch_mode)
        adapter = self.get_mode_adapter(arch_mode)
        return adapter.get_pipeline(
            self,
            vit_model_key,
            sampler_name,
            pipe_mode,
            vlm_model_key=vlm_model_key,
            base_model_key=base_model_key,
            patch_encode_prompt_fn=patch_encode_prompt,
            override_device_property_fn=_override_device_property,
        )

    def get_vlm(self, arch_mode: str, model_key: str = None, base_model_key: str = None):
        self._switch_arch_mode(arch_mode)
        if model_key == SAME_AS_IMAGE_GENERATION_PIPELINE:
            if self.text_encoder is not None and model_key != self.current_vlm:
                self.clear_pipelines(del_vlm=True)

            if self.text_encoder is None:
                cfg = self.mode_config.get(arch_mode, {})
                base_choices = cfg.get("base_pipelines") or [None]
                resolved_base_model = (
                    base_model_key
                    if base_model_key not in (None, "Default", "None")
                    else base_choices[0]
                )
                self.get_pipeline(
                    arch_mode,
                    SAME_AS_IMAGE_GENERATION_PIPELINE,
                    list(SAMPLERS.keys())[0],
                    pipe_mode="t2i",
                    vlm_model_key=model_key,
                    base_model_key=resolved_base_model,
                )

            self.current_vlm = model_key
        elif self.text_encoder is not None and (model_key != self.current_vlm):
            self.clear_pipelines(del_vlm=True)

        if self.text_encoder is None:
            self._load_vlm(model_key)
        release_memory_resources()
        if isinstance(self.text_encoder, Qwen3Model):
            text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
                BASE_QWEN3_VL_ID,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            text_encoder.model.text_model = self.text_encoder
            self.vision_processor = Qwen3VLProcessor.from_pretrained(
                BASE_QWEN3_VL_ID,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            release_memory_resources()
            text_encoder.to("cuda")
            return text_encoder, self.tokenizer, self.vision_processor
        else:
            return self.text_encoder, self.tokenizer, self.vision_processor

    def clear_pipelines(self, del_vlm: bool = False, del_vit: bool = False, del_vae: bool = False):
        self.pipes.clear()
        self.current_base_model = None
        if del_vlm:
            del self.text_encoder
            self.text_encoder = None
            del self.tokenizer
            self.tokenizer = None
            del self.vision_processor
            self.vision_processor = None
            self.current_vlm = None
            self.is_set_te_offload = False
        if del_vit:
            del self.transformer
            self.transformer = None
            self.current_vit = None
        if del_vae:
            del self.vae
            self.vae = None
        release_memory_resources()

    def _load_vlm(self, model_key: str = None):
        if self.text_encoder is not None:
            if model_key != self.current_vlm:
                print(
                    "Please unload previous VLM model first before loading a new one."
                )
                raise RuntimeError("Previous VLM model not unloaded.")
            return
        if model_key is None:
            model_key = list(self.vlm_model_table.keys())[0]
        self.current_vlm = model_key
        cfg = self.vlm_model_table[model_key]
        te_cls = (
            _import_class_from_string(cfg["model_class"])
            or Qwen2_5_VLForConditionalGeneration
        )
        tk_cls = _import_class_from_string(cfg["tokenizer_class"]) or AutoTokenizer
        vp_cls = (
            _import_class_from_string(cfg["processor_class"]) or Qwen2_5_VLProcessor
        )
        self.text_encoder = te_cls.from_pretrained(
            cfg["id"],
            torch_dtype=getattr(torch, cfg.get("dtype", "bfloat16")),
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.tokenizer = tk_cls.from_pretrained(cfg["id"], use_fast=False)
        self.vision_processor = vp_cls.from_pretrained(
            cfg["id"], low_cpu_mem_usage=True, trust_remote_code=True
        )

        opt_policy = self.opt_pol_cfg.get("opt_policy", None)
        if opt_policy == "no_offload":
            self.text_encoder.to("cuda")

        release_memory_resources()

    def _switch_arch_mode(self, arch_mode: str):
        if arch_mode == self.current_arch_mode:
            return
        cfg = self.mode_config.get(arch_mode, {})
        if not cfg.get("image_generation", True):
            print("Switched to non-image-generation mode, clearing pipelines.")
            self.clear_pipelines(del_vit=True, del_vae=True)
            self.current_arch_mode = arch_mode
            return

        if self.text_encoder is not None:
            class_name = cfg.get(
                "model_class", None
            )  # example: transformers.Qwen2_5_VLForConditionalGeneration
            class_name = class_name.split(".")[-1] if class_name is not None else None
            if class_name is not None and not isinstance(
                self.text_encoder, type(class_name)
            ):
                print(
                    f"Cleared VLM model due to vlm model class {class_name} mismatch (current: {type(self.text_encoder)})."
                )
                self.clear_pipelines(del_vlm=True, del_vae=True)

        self.current_arch_mode = arch_mode


def _import_class_from_string(class_path: str):
    try:
        module_path, class_name = class_path.rsplit(".", 1)
        module = import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        print(f"Error importing {class_path}: {e}")
        return None
