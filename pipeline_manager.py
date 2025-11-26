import torch
from types import MethodType
from transformers import (
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
)
from diffusers import (
    QwenImagePipeline, QwenImageImg2ImgPipeline,
    QwenImageInpaintPipeline,
    FlowMatchEulerDiscreteScheduler,
    DiffusionPipeline,
)
from nunchaku.models.transformers.transformer_qwenimage import (
    NunchakuQwenImageTransformer2DModel
)
from qwen_image_pipelines import QwenImageEditPlusPipeline, QwenImageEditPlusInpaintPipeline
from constants import (
    SAMPLERS,
    FLOWMATCH_CFG,
    TEXT_ENCODER_ID,
    DEFAULT_TEXT_ENCODER_ID,
    DEFAULT_QWEN_IMAGE_ID,
    DEFAULT_QWEN_IMAGE_EDIT_ID,
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
        return self.transformer.device if hasattr(self, "transformer") else _orig_fget(self)

    pipe.__class__.device = property(_vit_first)
    pipe.__class__._device_overridden = True

def patch_encode_prompt(pipe, opt_policy):
    original_encode = pipe.encode_prompt.__func__

    def encode_prompt_cast(self, *args, **kwargs):
        # high VRAM setting moves the model; others only match dtype without moving
        tgt_dtype = self.transformer.dtype
        if opt_policy == "high_vram":
            # Move to GPU
            tgt_dev  = self.transformer.device
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

class PipelineManager():
    def __init__(self, opt_pol_cfg: dict, vit_model_table: dict):
        self.pipes: dict[str, DiffusionPipeline] = {}
        self.current_model = None
        self.text_encoder = None
        self.tokenizer = None
        self.vision_processor = None
        self.transformer = None
        self.vae = None
        self.opt_pol_cfg = opt_pol_cfg
        self.vit_model_table = vit_model_table

    def get_pipeline(self, model_key: str, sampler_name: str, mode: str = "t2i"):
        is_edit_model = self.vit_model_table[model_key]["edit"]
        # First load
        if self.pipes == {}:
            self._load_vlm()
            self._load_transformer(model_key)
            scheduler   = SAMPLERS[sampler_name].from_config(FLOWMATCH_CFG)
            if self.vae is None:
                self.pipes["t2i"] = QwenImagePipeline.from_pretrained(
                    DEFAULT_QWEN_IMAGE_ID,
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
                    DEFAULT_QWEN_IMAGE_ID,
                    text_encoder=self.text_encoder,
                    tokenizer=self.tokenizer,
                    transformer=self.transformer,
                    scheduler=scheduler,
                    vae=self.vae,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                )
            self.pipes["i2i"] = QwenImageImg2ImgPipeline.from_pretrained(
                DEFAULT_QWEN_IMAGE_ID, **self.pipes["t2i"].components)
            self.pipes["i2i_edit"] = QwenImageEditPlusPipeline.from_pretrained(
                DEFAULT_QWEN_IMAGE_EDIT_ID, vision_processor=self.vision_processor, **self.pipes["t2i"].components)
            self.pipes["inpaint"] = QwenImageInpaintPipeline.from_pretrained(
                DEFAULT_QWEN_IMAGE_ID, **self.pipes["t2i"].components)
            self.pipes["inpaint_edit"] = QwenImageEditPlusInpaintPipeline.from_pretrained(
                DEFAULT_QWEN_IMAGE_EDIT_ID, **self.pipes["i2i_edit"].components)
            
            opt_policy = self.opt_pol_cfg.get("opt_policy", None)
            for _pipe in self.pipes.values():
                patch_encode_prompt(_pipe, opt_policy)
            if self.opt_pol_cfg.get("enable_vae_slicing", True):
                self.vae.enable_slicing()
            if self.opt_pol_cfg.get("enable_vae_tiling", False):
                self.vae.enable_tiling()

            if opt_policy == "high_vram":
                self.transformer.to("cuda")
                self.vae.to("cuda")
                for _pipe in self.pipes.values():
                    _override_device_property(_pipe)
                print("Enabled high vram setting for offloading.")
            elif opt_policy == "mid_vram":
                self.pipes["t2i"].enable_model_cpu_offload()
                print("Enabled medium vram setting for offloading.")
            elif opt_policy == "low_vram":
                self.transformer.set_offload(True, use_pin_memory=False, num_blocks_on_gpu=1)
                self.pipes["t2i"]._exclude_from_cpu_offload.append("transformer")
                self.pipes["t2i"].enable_sequential_cpu_offload()
                print("Enabled low vram setting for offloading.")
            else:
                print(f"Unknown opt_policy: {opt_policy}")
                print("Available options: high_vram | mid_vram | low_vram")
            self.current_model = model_key
            if mode == "t2i":
                return self.pipes["t2i"]
            elif mode == "i2i":
                return self.pipes["i2i_edit"] if is_edit_model else self.pipes["i2i"]
            else:
                return self.pipes["inpaint_edit"] if is_edit_model else self.pipes["inpaint"]

        # Model switch
        if model_key != self.current_model:
            self.pipes.clear()
            del self.transformer
            self.transformer = None
            release_memory_resources()

            return self.get_pipeline(model_key, sampler_name, mode)

        # Sampler switch
        if self.pipes["t2i"].scheduler.__class__ is not SAMPLERS.get(sampler_name, FlowMatchEulerDiscreteScheduler):
            self.pipes["t2i"].scheduler = SAMPLERS[sampler_name].from_config(FLOWMATCH_CFG)
            self.pipes["i2i"].scheduler = self.pipes["t2i"].scheduler
            self.pipes["i2i_edit"].scheduler = self.pipes["t2i"].scheduler
            self.pipes["inpaint"].scheduler = self.pipes["t2i"].scheduler
            self.pipes["inpaint_edit"].scheduler = self.pipes["t2i"].scheduler
        if mode == "t2i":
            return self.pipes["t2i"]
        elif mode == "i2i":
            return self.pipes["i2i_edit"] if is_edit_model else self.pipes["i2i"]
        else:
            return self.pipes["inpaint_edit"] if is_edit_model else self.pipes["inpaint"]

    def _load_vlm(self):
        if self.text_encoder is None or self.tokenizer is None or self.vision_processor is None:
            self.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                TEXT_ENCODER_ID,
                torch_dtype=torch.float16 if "AWQ" in TEXT_ENCODER_ID else torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(TEXT_ENCODER_ID, use_fast=False)
            self.vision_processor = Qwen2_5_VLProcessor.from_pretrained(DEFAULT_TEXT_ENCODER_ID)
            release_memory_resources()

    def _load_transformer(self, model_key: str):
        self.transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
            self.vit_model_table[model_key]["path"],
            low_cpu_mem_usage=True,
        )
        release_memory_resources()