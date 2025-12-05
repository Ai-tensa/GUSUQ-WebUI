from importlib import import_module
import torch
from types import MethodType
from transformers import (
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
)
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    DiffusionPipeline,
)
from nunchaku.models.transformers.transformer_qwenimage import (
    NunchakuQwenImageTransformer2DModel,
)
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
)
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
            embeds = embeds.to(dtype=tgt_dtype)
            return embeds, mask

    pipe.encode_prompt = MethodType(encode_prompt_cast, pipe)


class PipelineManager:
    def __init__(
        self,
        opt_pol_cfg: dict,
        vlm_model_table: dict,
        vit_model_table: dict,
        mode_config: dict,
    ):
        self.pipes: dict[str, DiffusionPipeline] = {}
        self.current_arch_mode = None
        self.current_vlm = None
        self.current_vit = None
        self.text_encoder = None
        self.tokenizer = None
        self.vision_processor = None
        self.transformer = None
        self.vae = None
        self.opt_pol_cfg = opt_pol_cfg
        self.vlm_model_table = vlm_model_table
        self.vit_model_table = vit_model_table
        self.mode_config = mode_config
        self.is_set_te_offload = False

    def get_pipeline(
        self,
        arch_mode: str,
        vit_model_key: str,
        sampler_name: str,
        pipe_mode: str = "t2i",
        vlm_model_key: str = None,
    ):
        self._switch_arch_mode(arch_mode)
        is_edit_model = self.vit_model_table[vit_model_key]["edit"]
        # First load
        if self.pipes == {}:
            del_vlm = vlm_model_key != self.current_vlm
            del_vit = vit_model_key != self.current_vit
            if del_vlm or del_vit:
                self.clear_pipelines(del_vlm=del_vlm, del_vit=del_vit)
                release_memory_resources()
            self._load_vlm(vlm_model_key)
            self._load_transformer(vit_model_key)
            scheduler = SAMPLERS[sampler_name].from_config(FLOWMATCH_CFG)
            if self.vae is None:
                self.pipes["t2i"] = QwenImagePipeline.from_pretrained(
                    BASE_QWEN_IMAGE_ID,
                    text_encoder=self.text_encoder,
                    tokenizer=self.tokenizer,
                    transformer=self.transformer,
                    scheduler=scheduler,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                )
                self.vae = self.pipes["t2i"].vae
            else:
                self.pipes["t2i"] = QwenImagePipeline.from_pretrained(
                    BASE_QWEN_IMAGE_ID,
                    text_encoder=self.text_encoder,
                    tokenizer=self.tokenizer,
                    transformer=self.transformer,
                    scheduler=scheduler,
                    vae=self.vae,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                )
            self.pipes["i2i"] = QwenImageImg2ImgPipeline.from_pretrained(
                BASE_QWEN_IMAGE_ID, **self.pipes["t2i"].components
            )
            self.pipes["i2i_edit"] = QwenImageEditPlusPipeline.from_pretrained(
                BASE_QWEN_IMAGE_EDIT_ID,
                vision_processor=self.vision_processor,
                **self.pipes["t2i"].components,
            )
            self.pipes["inpaint"] = QwenImageInpaintPipeline.from_pretrained(
                BASE_QWEN_IMAGE_ID, **self.pipes["t2i"].components
            )
            self.pipes["inpaint_edit"] = (
                QwenImageEditPlusInpaintPipeline.from_pretrained(
                    BASE_QWEN_IMAGE_EDIT_ID, **self.pipes["i2i_edit"].components
                )
            )

            opt_policy = self.opt_pol_cfg.get("opt_policy", None)
            for _pipe in self.pipes.values():
                patch_encode_prompt(_pipe, opt_policy)
            if self.opt_pol_cfg.get("enable_vae_slicing", True):
                self.vae.enable_slicing()
            if self.opt_pol_cfg.get("enable_vae_tiling", False):
                self.vae.enable_tiling(
                    tile_sample_min_height=self.opt_pol_cfg.get(
                        "vae_tile_sample_min_height", None
                    ),
                    tile_sample_min_width=self.opt_pol_cfg.get(
                        "vae_tile_sample_min_width", None
                    ),
                    tile_sample_stride_height=self.opt_pol_cfg.get(
                        "vae_tile_sample_stride_height", None
                    ),
                    tile_sample_stride_width=self.opt_pol_cfg.get(
                        "vae_tile_sample_stride_width", None
                    ),
                )

            if opt_policy == "no_offload":
                self.pipes["i2i_edit"].to("cuda")
                print("No offloading applied.")
            elif opt_policy == "high_vram":
                self.transformer.to("cuda")
                self.vae.to("cuda")
                for _pipe in self.pipes.values():
                    _override_device_property(_pipe)
                print("Enabled high vram setting for offloading.")
            elif opt_policy == "mid_vram":
                self.pipes["t2i"].enable_model_cpu_offload()
                print("Enabled medium vram setting for offloading.")
            elif opt_policy == "low_vram":
                self.transformer.set_offload(
                    True, use_pin_memory=False, num_blocks_on_gpu=1
                )
                self.pipes["t2i"]._exclude_from_cpu_offload.append("transformer")
                self.pipes["t2i"].enable_sequential_cpu_offload()
                print("Enabled low vram setting for offloading.")
            else:
                print(f"Unknown opt_policy: {opt_policy}")
                print("Available options: high_vram | mid_vram | low_vram")
            self.is_set_te_offload = True

            release_memory_resources()
            if pipe_mode == "t2i":
                return self.pipes["t2i"]
            elif pipe_mode == "i2i":
                return self.pipes["i2i_edit"] if is_edit_model else self.pipes["i2i"]
            else:
                return (
                    self.pipes["inpaint_edit"]
                    if is_edit_model
                    else self.pipes["inpaint"]
                )

        # Model switch
        if vlm_model_key != self.current_vlm or vit_model_key != self.current_vit:
            del_vlm = vlm_model_key != self.current_vlm
            del_vit = vit_model_key != self.current_vit
            self.clear_pipelines(del_vlm=del_vlm, del_vit=del_vit)

            return self.get_pipeline(
                vit_model_key, sampler_name, pipe_mode, vlm_model_key=vlm_model_key
            )

        # Sampler switch
        if self.pipes["t2i"].scheduler.__class__ is not SAMPLERS.get(
            sampler_name, FlowMatchEulerDiscreteScheduler
        ):
            self.pipes["t2i"].scheduler = SAMPLERS[sampler_name].from_config(
                FLOWMATCH_CFG
            )
            self.pipes["i2i"].scheduler = self.pipes["t2i"].scheduler
            self.pipes["i2i_edit"].scheduler = self.pipes["t2i"].scheduler
            self.pipes["inpaint"].scheduler = self.pipes["t2i"].scheduler
            self.pipes["inpaint_edit"].scheduler = self.pipes["t2i"].scheduler
        if pipe_mode == "t2i":
            return self.pipes["t2i"]
        elif pipe_mode == "i2i":
            return self.pipes["i2i_edit"] if is_edit_model else self.pipes["i2i"]
        else:
            return (
                self.pipes["inpaint_edit"] if is_edit_model else self.pipes["inpaint"]
            )

    def get_vlm(self, arch_mode: str, model_key: str = None):
        self._switch_arch_mode(arch_mode)
        if self.text_encoder is not None and (model_key != self.current_vlm):
            self.clear_pipelines(del_vlm=True)

        self._load_vlm(model_key)
        release_memory_resources()
        return self.text_encoder, self.tokenizer, self.vision_processor

    def clear_pipelines(self, del_vlm: bool = False, del_vit: bool = False):
        self.pipes.clear()
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

    def _load_transformer(self, model_key: str):
        if self.transformer is not None:
            if model_key != self.current_vit:
                print(
                    "Please unload previous transformer model first before loading a new one."
                )
                raise RuntimeError("Previous transformer model not unloaded.")
            return
        if model_key is None:
            model_key = list(self.vit_model_table.keys())[0]
        self.current_vit = model_key
        self.transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
            self.vit_model_table[model_key]["path"],
            low_cpu_mem_usage=True,
        )
        release_memory_resources()

    def _switch_arch_mode(self, arch_mode: str):
        if arch_mode == self.current_arch_mode:
            return

        if arch_mode == "VLM Only":
            print("Switched to VLM Only mode, clearing pipelines.")
            self.clear_pipelines(del_vit=True)
            self.current_arch_mode = arch_mode
            return

        cfg = self.mode_config.get(arch_mode, {})

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
                self.clear_pipelines(del_vlm=True)

        self.current_arch_mode = arch_mode


def _import_class_from_string(class_path: str):
    try:
        module_path, class_name = class_path.rsplit(".", 1)
        module = import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        print(f"Error importing {class_path}: {e}")
        return None
