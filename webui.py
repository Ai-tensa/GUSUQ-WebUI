import argparse
import torch
import gradio as gr
import math
import random
from tqdm.auto import tqdm
from datetime import datetime
from pathlib import Path
import json
import yaml
from functools import partial
from huggingface_hub import hf_hub_download
from PIL import Image, PngImagePlugin, ImageChops
from time import perf_counter
from pipeline_manager import PipelineManager
from constants import SAMPLERS
from constants import SAME_AS_IMAGE_GENERATION_PIPELINE
from config_loader import (
    ensure_yaml_from_sample,
    load_base_pipeline_table,
    load_yaml_config,
    load_vlm_model_table,
    load_vit_model_table,
    filter_models,
)
from cap import build_caption_prompt, vl_generate
from metadata import (
    apply_config_to_t2i,
    apply_config_to_i2i,
    apply_config_to_inpaint,
    extract_meta,
    show_meta,
    sync_meta,
)
from utils import release_memory_resources, rss_mb, round32

# ── global state ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="GUSUQ WebUI -- Gradio-based Unified Simple UI for Qwen-image with Nunchaku"
)
parser.add_argument(
    "--user-config-yaml",
    type=Path,
    default=Path("config/user_config.yaml"),
    help="Path to user configuration YAML",
)
parser.add_argument(
    "--mode-config-yaml",
    type=Path,
    default=Path("config/mode_config.yaml"),
    help="Path to mode configuration YAML",
)
parser.add_argument(
    "--opt-pol-yaml",
    type=Path,
    default=Path("config/opt_pol.yaml"),
    help="Path to optimization-policy YAML",
)
parser.add_argument(
    "--vlm-models-yaml",
    type=Path,
    default=Path("config/vlm_models.yaml"),
    help="Path to VLM models YAML",
)
parser.add_argument(
    "--vit-models-yaml",
    type=Path,
    default=Path("config/vit_models.yaml"),
    help="Path to ViT models YAML",
)
parser.add_argument(
    "--base-pipelines-yaml",
    type=Path,
    default=Path("config/base_pipelines.yaml"),
    help="Path to base pipeline config YAML",
)
parser.add_argument(
    "--adapters-yaml",
    type=Path,
    default=Path("config/adapters.yaml"),
    help="Path to adapters YAML",
)
parser.add_argument(
    "--port", type=int, default=None, help="Port number for the Gradio server"
)
parser.add_argument(
    "--server_name",
    type=str,
    default=None,
    help="Server name or IP address for the Gradio server.",
)
parser.add_argument(
    "--listen", action="store_true", help="Whether to listen on all interfaces"
)
args = parser.parse_args()

CONFIG_DIR = Path("config")


def _supports_fp4_vit_models() -> bool:
    """Detect whether the primary visible CUDA device should use FP4/NVFP4 ViT samples."""
    if not torch.cuda.is_available():
        return False

    try:
        capability = torch.cuda.get_device_capability(torch.cuda.current_device())
    except Exception:
        return False

    # NVIDIA's current support matrices show FP4 support on compute capability
    # 10.0, 11.0, and 12.0, so using the FP4 sample for 10.x-and-later devices
    # keeps future architectures on the FP4 path as well.
    return capability >= (10, 0)


def _ensure_config_inputs() -> None:
    ensure_yaml_from_sample(
        args.user_config_yaml,
        CONFIG_DIR / "user_config_sample.yaml",
        "user",
    )
    ensure_yaml_from_sample(
        args.mode_config_yaml,
        CONFIG_DIR / "mode_config_sample.yaml",
        "mode",
    )
    ensure_yaml_from_sample(
        args.opt_pol_yaml,
        CONFIG_DIR / "opt_pol_sample.yaml",
        "optimization policy",
    )
    ensure_yaml_from_sample(
        args.vlm_models_yaml,
        CONFIG_DIR / "vlm_models_sample.yaml",
        "VLM models",
    )
    vit_sample = (
        CONFIG_DIR / "vit_models_sample_50xx.yaml"
        if _supports_fp4_vit_models()
        else CONFIG_DIR / "vit_models_sample_old.yaml"
    )
    ensure_yaml_from_sample(args.vit_models_yaml, vit_sample, "ViT models")
    ensure_yaml_from_sample(
        args.base_pipelines_yaml,
        CONFIG_DIR / "base_pipelines_sample.yaml",
        "base pipeline",
    )
    ensure_yaml_from_sample(
        args.adapters_yaml,
        CONFIG_DIR / "adapters_sample.yaml",
        "adapters",
    )


_ensure_config_inputs()

with open(args.user_config_yaml, "r") as f:
    config = yaml.safe_load(f)
mode_config = load_yaml_config(args.mode_config_yaml)
mode_list = list(mode_config.keys())
if not mode_list:
    raise RuntimeError("mode_config must define at least one mode.")
default_mode = config.get("default_mode", mode_list[0])
default_enable_adapters = mode_config.get(default_mode, {}).get("enable_adapters", [])
default_enable_lora = "LoRA" in default_enable_adapters

with open(args.opt_pol_yaml, "r") as f:
    opt_pol_cfg = yaml.safe_load(f)
vlm_model_table = load_vlm_model_table(args.vlm_models_yaml)
vit_model_table = load_vit_model_table(args.vit_models_yaml)
base_pipeline_table = load_base_pipeline_table(args.base_pipelines_yaml)

vit_model_list = list(vit_model_table.keys())


def _get_base_pipeline_choices(mode_name: str) -> list[str]:
    cfg = mode_config.get(mode_name, {})
    allowed_base_arch = cfg.get("allowed_base_arch", "All")
    if allowed_base_arch is not None:
        base_pipelines = filter_models(base_pipeline_table, allowed_base_arch, "All")
        if base_pipelines:
            return base_pipelines
    return ["None"] if not cfg.get("image_generation", True) else ["Default"]


def _get_vit_choices(mode_name: str) -> list[str]:
    cfg = mode_config.get(mode_name, {})
    allowed_vit_arch = cfg.get("allowed_vit_arch", "All")
    if allowed_vit_arch is None:
        return [SAME_AS_IMAGE_GENERATION_PIPELINE] if cfg.get("image_generation", True) else ["None"]
    vit_choices = filter_models(vit_model_table, allowed_vit_arch, "All")
    if cfg.get("allow_same_as_image_generation_pipeline", False):
        return [SAME_AS_IMAGE_GENERATION_PIPELINE] + vit_choices
    return vit_choices


def _get_vlm_choices(mode_name: str) -> list[str]:
    cfg = mode_config.get(mode_name, {})
    return filter_models(
        vlm_model_table,
        allowed_arch=cfg.get("allowed_vlm_arch", "All"),
        allowed_variants=cfg.get("allowed_vlm_variants", "All"),
    )


def _get_sampler_choices(mode_name: str) -> list[str]:
    cfg = mode_config.get(mode_name, {})
    return ["None"] if not cfg.get("image_generation", True) else list(SAMPLERS.keys())


base_pipeline_list = _get_base_pipeline_choices(default_mode)
vit_model_list = _get_vit_choices(default_mode)
vlm_model_list = _get_vlm_choices(default_mode)
sampler_list = _get_sampler_choices(default_mode)

default_base_pipeline = config.get("default_base_pipeline", base_pipeline_list[0])
if default_base_pipeline not in base_pipeline_list:
    default_base_pipeline = base_pipeline_list[0]

default_vlm = config.get("default_vlm_model", vlm_model_list[0])
if default_vlm not in vlm_model_list:
    default_vlm = vlm_model_list[0]

default_vit = config.get("default_vit_model", vit_model_list[0])
if default_vit not in vit_model_list:
    default_vit = vit_model_list[0]

default_sampler = config.get("default_sampler", sampler_list[0])
if default_sampler not in sampler_list:
    default_sampler = sampler_list[0]

adapters_config = load_yaml_config(args.adapters_yaml)


def _adapter_mode_keys(mode_name: str) -> list[str]:
    keys = [mode_name]
    if " " in mode_name:
        keys.append(mode_name.replace(" ", "-"))
    if "-" in mode_name:
        keys.append(mode_name.replace("-", " "))
    seen = set()
    ordered = []
    for key in keys:
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


def _get_adapter_table(mode_name: str) -> dict:
    for key in _adapter_mode_keys(mode_name):
        table = adapters_config.get(key)
        if isinstance(table, dict):
            return table
    return {}


def _reload_lora_list(mode_name: str, lora_t2i: str, lora_i2i: str, lora_inp: str):
    global adapters_config

    try:
        adapters_config = load_yaml_config(args.adapters_yaml)
    except Exception as e:
        raise gr.Error(f"Failed to reload adapters config: {e}")

    cfg = mode_config.get(mode_name, {})
    enable_lora = "LoRA" in cfg.get("enable_adapters", [])
    lora_list = list(_get_adapter_table(mode_name).get("LoRA", {}).keys()) if enable_lora else []

    def _upd(current_name: str):
        value = current_name if current_name in lora_list else (lora_list[0] if lora_list else None)
        return gr.update(choices=lora_list, value=value)

    return _upd(lora_t2i), _upd(lora_i2i), _upd(lora_inp)


default_adapter_table = _get_adapter_table(default_mode)
default_lora_table = default_adapter_table.get("LoRA", {}) if default_enable_lora else {}
default_lora_list = list(default_lora_table.keys()) if default_enable_lora else []

BASE_OUTPUT_DIR = Path(config.get("output_dir", "outputs"))

port = args.port or config.get("server_port", 7860)
server_name = args.server_name or config.get("server_name", "127.0.0.1")
if args.listen or config.get("listen", False):
    server_name = "0.0.0.0"

torch.backends.cuda.matmul.allow_tf32 = True
pm = PipelineManager(
    opt_pol_cfg,
    vlm_model_table,
    vit_model_table,
    base_pipeline_table,
    mode_config,
)

# ── helpers ───────────────────────────────────────────────────────────────
EDIT_TRAIN_PIXELS = 1024 * 1024  # 1,048,576 px


def _replace_prompts(find, repl, use_pos_neg, pos_txt, neg_txt):
    use_pos = "Positive" in use_pos_neg
    use_neg = "Negative" in use_pos_neg
    new_pos = pos_txt.replace(find, repl) if use_pos and find else pos_txt
    new_neg = neg_txt.replace(find, repl) if use_neg and find else neg_txt
    return gr.update(value=new_pos), gr.update(value=new_neg)


def _swap_dims(w, h):
    if isinstance(w, (int, float)) and isinstance(h, (int, float)):
        return int(h), int(w)
    return gr.update(), gr.update()


def _import_dims(img):
    if img is None:
        return gr.update(), gr.update()
    if isinstance(img, dict):
        img = img["background"].convert("RGB")
    w, h = img.size
    return round32(w), round32(h)


def rescale_dims(w, h, model_name):
    if "edit" not in str(model_name).lower():
        gr.Info("Rescaling is only available for Edit models.")
        return gr.update(), gr.update()
    return _rescale_dims(w, h)


def _rescale_dims(w, h):
    r = float(w) / float(h)
    new_h = math.sqrt(EDIT_TRAIN_PIXELS / r)
    new_w = new_h * r
    new_h = round32(new_h)
    new_w = round32(new_w)
    return int(new_w), int(new_h)


def _extract_mask(ev):
    if isinstance(ev, dict):
        if ev["layers"]:
            mask = ev["layers"][-1].convert("L").point(lambda x: 255 if x > 0 else 0)
        else:
            diff = ImageChops.difference(
                ev["composite"].convert("RGB"), ev["background"].convert("RGB")
            ).convert("L")
            mask = diff.point(lambda x: 255 if x > 10 else 0)
        return mask.convert("RGB")
    raise ValueError("Unexpected editor value")


def _extract_seed(meta):
    if isinstance(meta, dict) and "seed" in meta:
        return int(meta["seed"])
    return gr.update()


def _apply_mode(mode_name, base_pipe_dd, vlm_dd, vit_dd, sampler, lora_t2i, lora_i2i, lora_inp):
    cfg = mode_config.get(mode_name, {})
    adapter_table = _get_adapter_table(mode_name)
    keep_tabs = cfg.get("keep_tabs", [])
    enable_adapters = cfg.get("enable_adapters", [])
    enable_lora = "LoRA" in enable_adapters
    lora_list = list(adapter_table.get("LoRA", {}).keys()) if enable_lora else []

    # Tab visibility updates
    t2i_tab_upd = gr.update(visible="t2i_tab" in keep_tabs)
    i2i_tab_upd = gr.update(visible="i2i_tab" in keep_tabs)
    inp_tab_upd = gr.update(visible="inpaint_tab" in keep_tabs)
    vlm_tab_upd = gr.update(visible="vlm_tab" in keep_tabs)
    png_tab_upd = gr.update(visible="png_info_tab" in keep_tabs)
    tab_upds = [
        t2i_tab_upd,
        i2i_tab_upd,
        inp_tab_upd,
        vlm_tab_upd,
        png_tab_upd,
    ]

    # Meta accordion updates
    meta_acc_t2i_upd = gr.update(visible=enable_lora)
    meta_acc_i2i_upd = gr.update(visible=enable_lora)
    meta_acc_inp_upd = gr.update(visible=enable_lora)
    meta_acc_upds = [meta_acc_t2i_upd, meta_acc_i2i_upd, meta_acc_inp_upd]

    # Dropdown updates
    new_base_pipeline_list = _get_base_pipeline_choices(mode_name)
    new_vlm_list = _get_vlm_choices(mode_name)
    new_vit_list = _get_vit_choices(mode_name)
    new_sampler_list = _get_sampler_choices(mode_name)

    new_base_pipeline = (
        base_pipe_dd
        if base_pipe_dd in new_base_pipeline_list
        else new_base_pipeline_list[0]
    )
    new_vlm = vlm_dd if vlm_dd in new_vlm_list else new_vlm_list[0]
    new_vit = vit_dd if vit_dd in new_vit_list else new_vit_list[0]
    new_sampler = sampler if sampler in new_sampler_list else new_sampler_list[0]
    base_pipe_upd = gr.update(
        choices=new_base_pipeline_list,
        value=new_base_pipeline,
        interactive=cfg.get("image_generation", True),
    )
    vlm_upd = gr.update(choices=new_vlm_list, value=new_vlm)
    vit_upd = gr.update(choices=new_vit_list, value=new_vit)
    sampler_upd = gr.update(choices=new_sampler_list, value=new_sampler)
    dd_upds = [base_pipe_upd, vlm_upd, vit_upd, sampler_upd]

    # LoRA name dropdown updates
    def _lora_upd(current_name: str):
        value = current_name if current_name in lora_list else (lora_list[0] if lora_list else None)
        return gr.update(choices=lora_list, value=value)

    lora_name_t2i_upd = _lora_upd(lora_t2i)
    lora_name_i2i_upd = _lora_upd(lora_i2i)
    lora_name_inp_upd = _lora_upd(lora_inp)
    lora_upds = [lora_name_t2i_upd, lora_name_i2i_upd, lora_name_inp_upd]

    return tab_upds + dd_upds + meta_acc_upds + lora_upds


def _is_edit_model_for_mode(arch_mode: str, vit: str, base_model: str) -> bool:
    if vit == SAME_AS_IMAGE_GENERATION_PIPELINE:
        return bool(base_pipeline_table.get(base_model, {}).get("edit", False))
    adapter = pm.get_mode_adapter(arch_mode)
    return bool(adapter.is_edit_model(vit_model_table, vit))


def _base_generation_params(
    arch_mode,
    prompt,
    negative,
    cfg,
    steps,
    width,
    height,
    bsz,
    gen_list,
):
    adapter = pm.get_mode_adapter(arch_mode)
    cfg_key = adapter.cfg_param_key
    return {
        "prompt": prompt,
        "negative_prompt": negative,
        "num_inference_steps": steps,
        "width": width,
        "height": height,
        "num_images_per_prompt": bsz,
        "generator": gen_list,
        cfg_key: cfg,
    }

def _parse_meta_prompt(meta_prompt: str) -> dict:
    raw = (meta_prompt or "").strip()
    if not raw:
        return {}

    try:
        obj = yaml.safe_load(raw)
    except Exception:
        raise gr.Error("Meta prompt must be valid YAML.")

    if not isinstance(obj, dict):
        raise gr.Error("Meta prompt must be a YAML mapping.")

    loras = obj.get("LoRA")
    if loras is not None and not isinstance(loras, dict):
        raise gr.Error("Meta prompt: LoRA must be a mapping of name: strength.")

    return obj


def _add_lora_to_meta(meta_prompt: str, lora_name: str, strength: float) -> str:
    if not lora_name:
        return meta_prompt or ""

    obj = _parse_meta_prompt(meta_prompt)
    obj.setdefault("LoRA", {})
    obj["LoRA"][lora_name] = strength
    return yaml.safe_dump(obj, sort_keys=False, allow_unicode=True)


def _is_nunchaku_lora_backend(pipe) -> bool:
    transformer = getattr(pipe, "transformer", None)
    return all(
        hasattr(transformer, method)
        for method in ("update_lora_params", "set_lora_strength", "reset_lora")
    )


def _resolve_nunchaku_lora_source(params: dict):
    src = params.get("pretrained_model_name_or_path_or_dict")
    if isinstance(src, dict):
        return src
    if not isinstance(src, str) or not src:
        raise gr.Error("Nunchaku LoRA requires a valid path or state dict.")

    if Path(src).exists():
        return src

    weight_name = params.get("weight_name")
    if weight_name:
        return hf_hub_download(repo_id=src, filename=weight_name)

    raise gr.Error(
        "Nunchaku LoRA requires a local safetensors path, or a Hugging Face repo plus weight_name."
    )


def _apply_nunchaku_loras(pipe, lora_params: dict):
    transformer = pipe.transformer
    requested_signature = tuple(
        sorted((name, float(params["strength"])) for name, params in lora_params.items())
    )
    current_signature = getattr(transformer, "_gusuq_nunchaku_lora_signature", ())

    if not lora_params:
        if current_signature:
            transformer.reset_lora()
            transformer._gusuq_nunchaku_lora_signature = ()
        return

    if current_signature == requested_signature:
        return

    if current_signature:
        transformer.reset_lora()
        transformer._gusuq_nunchaku_lora_signature = ()

    sources = []
    strengths = []
    for params in lora_params.values():
        sources.append(_resolve_nunchaku_lora_source(params))
        strengths.append(float(params["strength"]))

    transformer.update_lora_params(sources, strengths)
    transformer._gusuq_nunchaku_lora_signature = requested_signature


def _apply_diffusers_loras(pipe, lora_params: dict):
    requested_signature = tuple(
        sorted((name, float(params["strength"])) for name, params in lora_params.items())
    )
    current_signature = getattr(pipe, "_gusuq_diffusers_lora_signature", ())
    loaded_names = pipe.get_list_adapters().get("transformer", [])

    if not lora_params:
        if loaded_names or current_signature:
            pipe.unload_lora_weights()
            pipe._gusuq_diffusers_lora_signature = ()
        return

    if current_signature == requested_signature:
        return

    if loaded_names or current_signature:
        pipe.unload_lora_weights()
        pipe._gusuq_diffusers_lora_signature = ()

    adapter_names = []
    adapter_weights = []
    for name, params in lora_params.items():
        load_params = dict(params)
        adapter_name = load_params.pop("adapter_name")
        adapter_names.append(adapter_name)
        adapter_weights.append(float(load_params.pop("strength")))
        load_params["adapter_name"] = adapter_name
        pipe.load_lora_weights(**load_params)

    pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)
    pipe._gusuq_diffusers_lora_signature = requested_signature


def _apply_requested_loras(pipe, arch_mode: str, meta_prompt: str):
    enable_adapters = mode_config.get(arch_mode, {}).get("enable_adapters", [])
    if not enable_adapters:
        if _is_nunchaku_lora_backend(pipe):
            _apply_nunchaku_loras(pipe, {})
        elif hasattr(pipe, "unload_lora_weights"):
            _apply_diffusers_loras(pipe, {})
        return

    adapters_requests = _parse_meta_prompt(meta_prompt)
    lora_requests = adapters_requests.get("LoRA", {}) if "LoRA" in enable_adapters else {}
    lora_table = _get_adapter_table(arch_mode).get("LoRA", {})
    lora_params = {}
    for name, strength in lora_requests.items():
        if name not in lora_table:
            raise gr.Error(f"LoRA '{name}' not found for mode '{arch_mode}'.")
        params = dict(lora_table[name])
        params["adapter_name"] = name
        params["strength"] = float(strength)
        lora_params[name] = params

    if _is_nunchaku_lora_backend(pipe):
        _apply_nunchaku_loras(pipe, lora_params)
    else:
        _apply_diffusers_loras(pipe, lora_params)


def _build_params(
    arch_mode,
    gen_mode,
    prompt,
    negative,
    cfg,
    steps,
    width,
    height,
    bsz,
    gen_list,
    **extra,
):
    base_params = _base_generation_params(
        arch_mode,
        prompt,
        negative,
        cfg,
        steps,
        width,
        height,
        bsz,
        gen_list,
    )
    adapter = pm.get_mode_adapter(arch_mode)
    return adapter.build_params(gen_mode, base_params, extra)


def _generator_device_for(pipe):
    if pm.opt_pol_cfg.get("opt_policy") == "low_vram":
        return pipe._execution_device
    return pipe.transformer.device


def generate_t2i(
    arch_mode,
    base_model,
    vlm,
    vit,
    prompt,
    negative,
    meta_prompt,
    cfg,
    steps,
    width,
    height,
    bsz,
    bcnt,
    sampler,
    seed,
):
    start_time = perf_counter()
    base_seed = random.randint(0, 2**32 - 1) if seed == -1 else int(seed)
    negative = negative if negative.strip() != "" else None

    print("RSS before get pipe:", rss_mb(), "MB")
    pipe = pm.get_pipeline(
        arch_mode,
        vit,
        sampler,
        vlm_model_key=vlm,
        pipe_mode="t2i",
        base_model_key=base_model,
    )
    print("RSS after get pipe :", rss_mb(), "MB")
    gens = [
        torch.Generator(device=_generator_device_for(pipe)).manual_seed(base_seed + i)
        for i in range(bsz * bcnt)
    ]

    out_dir = (
        BASE_OUTPUT_DIR / "t2i"
        if config.get("save_subdir_by_mode", True)
        else BASE_OUTPUT_DIR
    )
    if config.get("save_subdir_by_date", True):
        date_str = datetime.now().strftime("%Y%m%d")
        out_dir = out_dir / date_str
    out_dir.mkdir(exist_ok=True, parents=True)
    images, meta_list = [], []
    _apply_requested_loras(pipe, arch_mode, meta_prompt)

    for i in tqdm(range(bcnt), desc="Batches"):
        params = _build_params(
            arch_mode,
            "t2i",
            prompt,
            negative,
            cfg,
            steps,
            width,
            height,
            bsz,
            gens[i * bsz : (i + 1) * bsz],
        )
        result = pipe(**params).images
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        for j, img in enumerate(result):
            s = base_seed + i * bsz + j
            meta = dict(
                ts=ts,
                model=vit,
                prompt=prompt,
                negative=negative,
                meta_prompt=meta_prompt,
                sampler=sampler,
                steps=steps,
                cfg=cfg,
                width=width,
                height=height,
                seed=s,
            )
            meta_list.append(meta)
            fname = f"{ts}_{s}.png"
            fpath = out_dir / fname
            info = PngImagePlugin.PngInfo()
            info.add_text("parameters", json.dumps(meta))
            img.save(fpath, "PNG", pnginfo=info, optimize=True)
            images.append(str(fpath))

    release_memory_resources()
    elapsed = perf_counter() - start_time
    status = f"Generated {bsz * bcnt} images in {elapsed:.1f} seconds."
    return images, meta_list, status


def generate_i2i(
    arch_mode,
    base_model,
    vlm,
    vit,
    input_image,
    enable_ref1,
    ref_image1,
    enable_ref2,
    ref_image2,
    enable_ref3,
    ref_image3,
    prompt,
    negative,
    meta_prompt,
    cfg,
    resize_before_i2i,
    strength,
    steps,
    width,
    height,
    bsz,
    bcnt,
    sampler,
    seed,
    consistency_strength,
):
    start_time = perf_counter()
    adapter = pm.get_mode_adapter(arch_mode)
    is_edit_model = _is_edit_model_for_mode(arch_mode, vit, base_model)
    prepared = adapter.prepare_i2i_inputs(
        input_image=input_image,
        width=width,
        height=height,
        resize_before_i2i=resize_before_i2i,
        strength=strength,
        consistency_strength=consistency_strength,
        is_edit_model=is_edit_model,
        notify=gr.Info,
    )
    input_image = prepared["input_image"]
    resize_before_i2i = prepared["resize_before_i2i"]
    strength = prepared["strength"]
    consistency_strength = prepared["consistency_strength"]

    base_seed = random.randint(0, 2**32 - 1) if seed == -1 else int(seed)
    negative = negative if negative.strip() != "" else None
    print("RSS before get pipe:", rss_mb(), "MB")
    pipe = pm.get_pipeline(
        arch_mode,
        vit,
        sampler,
        pipe_mode="i2i",
        vlm_model_key=vlm,
        base_model_key=base_model,
    )
    print("RSS after get pipe :", rss_mb(), "MB")
    gens = [
        torch.Generator(device=_generator_device_for(pipe)).manual_seed(base_seed + i)
        for i in range(bsz * bcnt)
    ]
    out_dir = (
        BASE_OUTPUT_DIR / "i2i"
        if config.get("save_subdir_by_mode", True)
        else BASE_OUTPUT_DIR
    )
    if config.get("save_subdir_by_date", True):
        date_str = datetime.now().strftime("%Y%m%d")
        out_dir = out_dir / date_str
    out_dir.mkdir(exist_ok=True, parents=True)
    refs = []
    if enable_ref1 and ref_image1 is not None:
        refs.append(ref_image1)
    if enable_ref2 and ref_image2 is not None:
        refs.append(ref_image2)
    if enable_ref3 and ref_image3 is not None:
        refs.append(ref_image3)
    if not is_edit_model and refs:
        gr.Info(
            "Multiple input images are only supported for Edit models. Ignoring reference images."
        )
    input_images = [input_image] + refs if is_edit_model else input_image
    images, meta_list = [], []
    _apply_requested_loras(pipe, arch_mode, meta_prompt)
    for i in tqdm(range(bcnt), desc="Batches"):
        params = _build_params(
            arch_mode,
            "i2i",
            prompt,
            negative,
            cfg,
            steps,
            width,
            height,
            bsz,
            gens[i * bsz : (i + 1) * bsz],
            image=input_images,
            strength=strength,
            consistency_strength=consistency_strength,
            is_edit_model=is_edit_model,
        )
        result = pipe(**params).images
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        for j, img in enumerate(result):
            s = base_seed + i * bsz + j
            meta = dict(
                ts=ts,
                model=vit,
                prompt=prompt,
                negative=negative,
                meta_prompt=meta_prompt,
                sampler=sampler,
                steps=steps,
                cfg=cfg,
                strength=strength,
                consistency_strength=consistency_strength,
                width=width,
                height=height,
                seed=s,
            )
            meta_list.append(meta)
            fname = f"{ts}_{s}.png"
            fpath = out_dir / fname
            info = PngImagePlugin.PngInfo()
            info.add_text("parameters", json.dumps(meta))
            img.save(fpath, "PNG", pnginfo=info, optimize=True)
            images.append(str(fpath))

    images.extend(input_images if isinstance(input_images, list) else [input_images])
    meta_list.extend(
        [extract_meta(img)[1] for img in input_images]
        if isinstance(input_images, list)
        else [extract_meta(input_images)[1]]
    )

    release_memory_resources()
    elapsed = perf_counter() - start_time
    status = f"Generated {bsz * bcnt} images in {elapsed:.1f} seconds."
    return images, meta_list, status


def generate_inpaint(
    arch_mode,
    base_model,
    vlm,
    vit,
    editor_val,
    enable_ref1,
    ref_image1,
    enable_ref2,
    ref_image2,
    enable_ref3,
    ref_image3,
    prompt,
    negative,
    meta_prompt,
    cfg,
    strength,
    steps,
    width,
    height,
    bsz,
    bcnt,
    sampler,
    seed,
    consistency_strength,
):
    start_time = perf_counter()
    adapter = pm.get_mode_adapter(arch_mode)
    is_edit_model = _is_edit_model_for_mode(arch_mode, vit, base_model)
    input_image = editor_val["background"].convert("RGB")
    mask_image = _extract_mask(editor_val)
    prepared = adapter.prepare_inpaint_inputs(
        input_image=input_image,
        mask_image=mask_image,
        width=width,
        height=height,
        strength=strength,
        consistency_strength=consistency_strength,
        is_edit_model=is_edit_model,
        notify=gr.Info,
    )
    input_image = prepared["input_image"]
    mask_image = prepared["mask_image"]
    strength = prepared["strength"]
    consistency_strength = prepared["consistency_strength"]
    control_image = prepared.get("control_image", input_image)
    base_seed = random.randint(0, 2**32 - 1) if seed == -1 else int(seed)
    negative = negative if negative.strip() != "" else None
    pipe = pm.get_pipeline(
        arch_mode,
        vit,
        sampler,
        pipe_mode="inpaint",
        vlm_model_key=vlm,
        base_model_key=base_model,
    )
    gens = [
        torch.Generator(device=_generator_device_for(pipe)).manual_seed(base_seed + i)
        for i in range(bsz * bcnt)
    ]
    out_dir = (
        BASE_OUTPUT_DIR / "inpaint"
        if config.get("save_subdir_by_mode", True)
        else BASE_OUTPUT_DIR
    )
    if config.get("save_subdir_by_date", True):
        date_str = datetime.now().strftime("%Y%m%d")
        out_dir = out_dir / date_str
    out_dir.mkdir(exist_ok=True, parents=True)
    refs = []
    if enable_ref1 and ref_image1 is not None:
        refs.append(ref_image1)
    if enable_ref2 and ref_image2 is not None:
        refs.append(ref_image2)
    if enable_ref3 and ref_image3 is not None:
        refs.append(ref_image3)
    if not is_edit_model and refs:
        gr.Info(
            "Multiple input images are only supported for Edit models. Ignoring reference images."
        )
    input_images = [input_image] + refs if is_edit_model else input_image
    images, meta_list = [], []
    _apply_requested_loras(pipe, arch_mode, meta_prompt)
    for i in tqdm(range(bcnt), desc="Batches"):
        params = _build_params(
            arch_mode,
            "inpaint",
            prompt,
            negative,
            cfg,
            steps,
            width,
            height,
            bsz,
            gens[i * bsz : (i + 1) * bsz],
            image=input_images,
            mask_image=mask_image,
            control_image=control_image,
            strength=strength,
            consistency_strength=consistency_strength,
            is_edit_model=is_edit_model,
        )
        result = pipe(**params).images
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        for j, img in enumerate(result):
            s = base_seed + i * bsz + j
            meta = dict(
                ts=ts,
                model=vit,
                prompt=prompt,
                negative=negative,
                meta_prompt=meta_prompt,
                sampler=sampler,
                steps=steps,
                cfg=cfg,
                strength=strength,
                consistency_strength=consistency_strength,
                width=width,
                height=height,
                seed=s,
            )
            meta_list.append(meta)
            fname = f"{ts}_{s}.png"
            fpath = out_dir / fname
            info = PngImagePlugin.PngInfo()
            info.add_text("parameters", json.dumps(meta))
            img.save(fpath, "PNG", pnginfo=info, optimize=True)
            images.append(str(fpath))

    images.extend(input_images if isinstance(input_images, list) else [input_images])
    meta_list.extend(
        [extract_meta(img)[1] for img in input_images]
        if isinstance(input_images, list)
        else [extract_meta(input_images)[1]]
    )
    images.append(mask_image)
    meta_list.append({})  # mask has no metadata

    release_memory_resources()
    elapsed = perf_counter() - start_time
    status = f"Generated {bsz * bcnt} images in {elapsed:.1f} seconds."
    return images, meta_list, status


def send_image(path, inpaint=False):
    if path:
        img = Image.open(path).convert("RGB")
        return img if not inpaint else {"background": img}
    return None


def update_metadata_and_path(img):
    if img is None:
        return gr.update(), gr.update()
    img_fp = None
    if isinstance(img, str):
        img_fp = img
    elif isinstance(img, dict) and "filepath" in img:
        img_fp = img["filepath"]
    meta, _ = extract_meta(img_fp)
    return meta, img_fp


# ── gradio UI ────────────────────────────────────────
with gr.Blocks(
    title="GUSUQ WebUI - Gradio Unified Simple UI for Qwen-Image with Nunchaku",
    fill_width=True,
    fill_height=True,
) as demo:
    with gr.Row():
        mode_dd = gr.Dropdown(
            mode_list,
            value=default_mode,
            allow_custom_value=True,
            label="Mode",
            scale=1,
        )
        base_pipe_dd = gr.Dropdown(
            base_pipeline_list,
            value=default_base_pipeline,
            allow_custom_value=True,
            label="Base Pipeline",
            scale=1,
        )
        vit_dd = gr.Dropdown(
            vit_model_list,
            value=default_vit,
            allow_custom_value=True,
            label="ViT Model",
            scale=4,
        )
        vlm_dd = gr.Dropdown(
            vlm_model_list,
            value=default_vlm,
            allow_custom_value=True,
            label="VLM Model",
            scale=1,
        )
        sampler = gr.Dropdown(
            sampler_list, value=default_sampler, label="Sampler"
        )

    with gr.Tab("t2i") as t2i_tab:
        with gr.Row():
            with gr.Column(scale=7):
                with gr.Group():
                    prompt_t2i = gr.Textbox(lines=4, label="Positive prompt")
                    with gr.Accordion("Negative prompt", open=False):
                        negative_t2i = gr.Textbox(
                            lines=2, label="Negative prompt", show_label=False
                        )
                    with gr.Accordion("Meta prompt", open=False, visible=default_enable_lora) as meta_acc_t2i:
                        meta_prompt_t2i = gr.Textbox(
                            lines=2, label="Meta prompt", show_label=False
                        )
                        with gr.Row():
                            lora_name_t2i = gr.Dropdown(
                                choices=default_lora_list,
                                value=default_lora_list[0] if default_lora_list else None,
                                allow_custom_value=True,
                                label="LoRA Name",
                            )
                            lora_strength_t2i = gr.Number(value=1.0, label="LoRA Strength")
                            with gr.Group():
                                add_lora_btn_t2i = gr.Button("Add LoRA")
                                reload_lora_btn_t2i = gr.Button("Reload List")
                        add_lora_btn_t2i.click(
                            _add_lora_to_meta,
                            inputs=[meta_prompt_t2i, lora_name_t2i, lora_strength_t2i],
                            outputs=meta_prompt_t2i,
                        )
            with gr.Column(scale=1):
                with gr.Tab("Generate"):
                    gen_btn_t2i = gr.Button("Generate", variant="primary")
                    progress_t2i = gr.Textbox(
                        "", label="Status", interactive=False, lines=1
                    )
                with gr.Tab("Replace"):
                    with gr.Group():
                        with gr.Row():
                            find_t2i = gr.Textbox(
                                placeholder="Find",
                                show_label=False,
                                lines=1,
                                min_width=80,
                            )
                            repl_t2i = gr.Textbox(
                                placeholder="Replace",
                                show_label=False,
                                lines=1,
                                min_width=80,
                            )
                        chk_t2i = gr.CheckboxGroup(
                            choices=["Positive", "Negative"],
                            value=["Positive"],
                            show_label=False,
                        )
                        rep_btn_t2i = gr.Button("Replace")
        with gr.Row():
            with gr.Column(scale=2):
                with gr.Group():
                    with gr.Row():
                        width_t2i = gr.Slider(
                            config.get("default_width_min", 512),
                            config.get("default_width_max", 4096),
                            value=config.get("default_width", 1328),
                            step=config.get("default_width_step", 64),
                            label="Width",
                        )
                        height_t2i = gr.Slider(
                            config.get("default_height_min", 512),
                            config.get("default_height_max", 4096),
                            value=config.get("default_height", 1328),
                            step=config.get("default_height_step", 64),
                            label="Height",
                        )
                    with gr.Row():
                        gr.Button("⇅").click(
                            _swap_dims,
                            inputs=[width_t2i, height_t2i],
                            outputs=[width_t2i, height_t2i],
                        )
                        gr.Button("⤧").click(
                            rescale_dims,
                            inputs=[width_t2i, height_t2i, vit_dd],
                            outputs=[width_t2i, height_t2i],
                        )

                with gr.Row():
                    bsz_t2i = gr.Slider(
                        config.get("default_batch_size_min", 1),
                        config.get("default_batch_size_max", 8),
                        value=config.get("default_batch_size", 8),
                        step=config.get("default_batch_size_step", 1),
                        label="Batch size",
                    )
                    bcnt_t2i = gr.Slider(
                        config.get("default_batch_count_min", 1),
                        config.get("default_batch_count_max", 100),
                        value=config.get("default_batch_count", 1),
                        step=config.get("default_batch_count_step", 1),
                        label="Batch count",
                    )

                with gr.Group():
                    with gr.Row():
                        cfg_t2i = gr.Slider(
                            config.get("default_cfg_min", 0.0),
                            config.get("default_cfg_max", 20.0),
                            value=config.get("default_cfg", 1.0),
                            step=config.get("default_cfg_step", 0.1),
                            label="True CFG scale",
                        )
                        steps_t2i = gr.Slider(
                            config.get("default_steps_min", 1),
                            config.get("default_steps_max", 100),
                            value=config.get("default_steps", 4),
                            step=config.get("default_steps_step", 1),
                            label="Steps",
                        )

                    with gr.Row(equal_height=True):
                        seed_box_t2i = gr.Number(
                            value=-1, label="Seed", precision=0, scale=3
                        )
                        with gr.Column():
                            gr.Button("🎲").click(lambda: -1, None, seed_box_t2i)
                            reuse_seed_btn_t2i = gr.Button("♻")

            with gr.Column(scale=3):
                gallery_t2i = gr.Gallery(
                    label="Result Images",
                    format="png",
                    object_fit="contain",
                    show_label=False,
                    columns=config.get("gallery_columns", 4),
                    preview=True,
                    interactive=False,
                )
                with gr.Row():
                    send_t2i_i2i_btn = gr.Button("Send to i2i")
                    send_t2i_inp_btn = gr.Button("Send to inpaint")
                    send_t2i_cap_btn = gr.Button("Send to caption")
                    send_t2i_vqa_btn = gr.Button("Send to VQA")
                dest_slot_t2i = gr.Radio(
                    ["input", "ref1", "ref2", "ref3"],
                    value="input",
                    label="Send-to target (for i2i / inpaint)",
                )
                meta_state_t2i = gr.State([])
                meta_view_t2i = gr.JSON(label="Metadata", show_label=False)
                sel_t2i_path = gr.State("")
                sel_idx_t2i = gr.State(0)

        gen_btn_t2i.click(
            generate_t2i,
            inputs=[
                mode_dd,
                base_pipe_dd,
                vlm_dd,
                vit_dd,
                prompt_t2i,
                negative_t2i,
                meta_prompt_t2i,
                cfg_t2i,
                steps_t2i,
                width_t2i,
                height_t2i,
                bsz_t2i,
                bcnt_t2i,
                sampler,
                seed_box_t2i,
            ],
            outputs=[gallery_t2i, meta_state_t2i, progress_t2i],
            show_progress_on=progress_t2i,
            concurrency_limit=1,
            concurrency_id="gpu",
        )
        rep_btn_t2i.click(
            _replace_prompts,
            inputs=[find_t2i, repl_t2i, chk_t2i, prompt_t2i, negative_t2i],
            outputs=[prompt_t2i, negative_t2i],
        )
        reuse_seed_btn_t2i.click(_extract_seed, meta_view_t2i, seed_box_t2i)
        gallery_t2i.select(
            show_meta,
            inputs=meta_state_t2i,
            outputs=[meta_view_t2i, sel_t2i_path, sel_idx_t2i],
        )
        gallery_t2i.change(
            sync_meta,
            inputs=[gallery_t2i, meta_state_t2i, sel_idx_t2i],
            outputs=[meta_view_t2i, sel_idx_t2i],
        )

    with gr.Tab("i2i") as i2i_tab:
        with gr.Row():
            with gr.Column(scale=7):
                with gr.Group():
                    prompt_i2i = gr.Textbox(lines=4, label="Positive prompt")
                    with gr.Accordion("Negative prompt", open=False):
                        negative_i2i = gr.Textbox(
                            lines=2, label="Negative prompt", show_label=False
                        )
                    with gr.Accordion("Meta prompt", open=False, visible=default_enable_lora) as meta_acc_i2i:
                        meta_prompt_i2i = gr.Textbox(
                            lines=2, label="Meta prompt", show_label=False
                        )
                        with gr.Row():
                            lora_name_i2i = gr.Dropdown(
                                choices=default_lora_list,
                                value=default_lora_list[0] if default_lora_list else None,
                                allow_custom_value=True,
                                label="LoRA Name",
                            )
                            lora_strength_i2i = gr.Number(value=1.0, label="LoRA Strength")
                            with gr.Group():
                                add_lora_btn_i2i = gr.Button("Add LoRA")
                                reload_lora_btn_i2i = gr.Button("Reload List")
                        add_lora_btn_i2i.click(
                            _add_lora_to_meta,
                            inputs=[meta_prompt_i2i, lora_name_i2i, lora_strength_i2i],
                            outputs=meta_prompt_i2i,
                        )
            with gr.Column(scale=1):
                with gr.Tab("Generate"):
                    gen_i2i_btn = gr.Button("Generate", variant="primary")
                    progress_i2i = gr.Textbox(
                        "", label="Status", interactive=False, lines=1
                    )
                with gr.Tab("Replace"):
                    with gr.Group():
                        with gr.Row():
                            find_i2i = gr.Textbox(
                                placeholder="Find",
                                show_label=False,
                                lines=1,
                                min_width=80,
                            )
                            repl_i2i = gr.Textbox(
                                placeholder="Replace",
                                show_label=False,
                                lines=1,
                                min_width=80,
                            )
                        chk_i2i = gr.CheckboxGroup(
                            choices=["Positive", "Negative"],
                            value=["Positive"],
                            show_label=False,
                        )
                        rep_btn_i2i = gr.Button("Replace")

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("Picture 1 (Main Input)"):
                        init_img_i2i = gr.Image(label="Picture 1 (Input)", type="pil")
                    with gr.TabItem("Picture 2 (Ref. 1)"):
                        ref_img1_i2i = gr.Image(label="Picture 2 (Ref. 1)", type="pil")
                        enable_ref1_i2i = gr.Checkbox(
                            label="Enable Ref. 1", value=False
                        )
                    with gr.TabItem("Picture 3 (Ref. 2)"):
                        ref_img2_i2i = gr.Image(label="Picture 3 (Ref. 2)", type="pil")
                        enable_ref2_i2i = gr.Checkbox(
                            label="Enable Ref. 2", value=False
                        )
                    with gr.TabItem("Picture 4 (Ref. 3)"):
                        ref_img3_i2i = gr.Image(label="Picture 4 (Ref. 3)", type="pil")
                        enable_ref3_i2i = gr.Checkbox(
                            label="Enable Ref. 3", value=False
                        )

                with gr.Group():
                    with gr.Row():
                        width_i2i = gr.Slider(
                            config.get("default_width_min", 512),
                            config.get("default_width_max", 4096),
                            value=config.get("default_width", 1328),
                            step=config.get("default_width_step", 64),
                            label="Width",
                        )
                        height_i2i = gr.Slider(
                            config.get("default_height_min", 512),
                            config.get("default_height_max", 4096),
                            value=config.get("default_height", 1328),
                            step=config.get("default_height_step", 64),
                            label="Height",
                        )
                    with gr.Row():
                        gr.Button("⇅").click(
                            _swap_dims,
                            inputs=[width_i2i, height_i2i],
                            outputs=[width_i2i, height_i2i],
                        )
                        gr.Button("↲").click(
                            _import_dims,
                            inputs=init_img_i2i,
                            outputs=[width_i2i, height_i2i],
                        )
                        gr.Button("⤧").click(
                            rescale_dims,
                            inputs=[width_i2i, height_i2i, vit_dd],
                            outputs=[width_i2i, height_i2i],
                        )

                with gr.Row():
                    bsz_i2i = gr.Slider(
                        config.get("default_batch_size_min", 1),
                        config.get("default_batch_size_max", 8),
                        value=config.get("default_batch_size", 8),
                        step=config.get("default_batch_size_step", 1),
                        label="Batch size",
                    )
                    bcnt_i2i = gr.Slider(
                        config.get("default_batch_count_min", 1),
                        config.get("default_batch_count_max", 100),
                        value=config.get("default_batch_count", 1),
                        step=config.get("default_batch_count_step", 1),
                        label="Batch count",
                    )

                with gr.Group():
                    resize_before_i2i = gr.Checkbox(
                        label="Resize Picture 1 to selected dimensions before processing",
                        value=False,
                    )
                    with gr.Row(visible=False) as strengths_i2i:
                        denoising_strength_i2i = gr.Slider(
                            config.get("denoising_strength_min", 0.0),
                            config.get("denoising_strength_max", 1.0),
                            value=config.get("denoising_strength", 1.0),
                            step=config.get("denoising_strength_step", 0.01),
                            label="Denoising Strength",
                        )
                        consistency_strength_i2i = gr.Slider(
                            config.get("consistency_strength_min", -1.0),
                            config.get("consistency_strength_max", 1.0),
                            value=config.get("consistency_strength", 0.0),
                            step=config.get("consistency_strength_step", 0.01),
                            label="Consistency Strength",
                        )

                    with gr.Row():
                        cfg_i2i = gr.Slider(
                            config.get("default_cfg_min", 0.0),
                            config.get("default_cfg_max", 20.0),
                            value=config.get("default_cfg", 1.0),
                            step=config.get("default_cfg_step", 0.1),
                            label="True CFG scale",
                        )
                        steps_i2i = gr.Slider(
                            config.get("default_steps_min", 1),
                            config.get("default_steps_max", 100),
                            value=config.get("default_steps", 4),
                            step=config.get("default_steps_step", 1),
                            label="Steps",
                        )

                    with gr.Row(equal_height=True):
                        seed_i2i = gr.Number(
                            value=-1, label="Seed", precision=0, scale=3
                        )
                        with gr.Column():
                            gr.Button("🎲").click(lambda: -1, None, seed_i2i)
                            reuse_seed_i2i_btn = gr.Button("♻")

            with gr.Column(scale=3):
                gallery_i2i = gr.Gallery(
                    label="Result Images",
                    format="png",
                    object_fit="contain",
                    show_label=False,
                    columns=config.get("gallery_columns", 4),
                    preview=True,
                    interactive=False,
                )
                with gr.Row():
                    send_i2i_i2i_btn = gr.Button("Send to i2i")
                    send_i2i_inp_btn = gr.Button("Send to inpaint")
                    send_i2i_cap_btn = gr.Button("Send to caption")
                    send_i2i_vqa_btn = gr.Button("Send to VQA")
                dest_slot_i2i = gr.Radio(
                    ["input", "ref1", "ref2", "ref3"],
                    value="input",
                    label="Send-to target (for i2i / inpaint)",
                )
                meta_state_i2i = gr.State([])
                meta_view_i2i = gr.JSON(label="Metadata", show_label=False)
                sel_i2i_path = gr.State("")
                sel_idx_i2i = gr.State(0)

        gen_i2i_btn.click(
            generate_i2i,
            inputs=[
                mode_dd,
                base_pipe_dd,
                vlm_dd,
                vit_dd,
                init_img_i2i,
                enable_ref1_i2i,
                ref_img1_i2i,
                enable_ref2_i2i,
                ref_img2_i2i,
                enable_ref3_i2i,
                ref_img3_i2i,
                prompt_i2i,
                negative_i2i,
                meta_prompt_i2i,
                cfg_i2i,
                resize_before_i2i,
                denoising_strength_i2i,
                steps_i2i,
                width_i2i,
                height_i2i,
                bsz_i2i,
                bcnt_i2i,
                sampler,
                seed_i2i,
                consistency_strength_i2i,
            ],
            outputs=[gallery_i2i, meta_state_i2i, progress_i2i],
            show_progress_on=progress_i2i,
            concurrency_id="gpu",
        )
        rep_btn_i2i.click(
            _replace_prompts,
            inputs=[find_i2i, repl_i2i, chk_i2i, prompt_i2i, negative_i2i],
            outputs=[prompt_i2i, negative_i2i],
        )
        reuse_seed_i2i_btn.click(_extract_seed, meta_view_i2i, seed_i2i)
        resize_before_i2i.change(
            lambda resize: gr.update(visible=resize), resize_before_i2i, strengths_i2i
        )
        gallery_i2i.select(
            show_meta,
            inputs=meta_state_i2i,
            outputs=[meta_view_i2i, sel_i2i_path, sel_idx_i2i],
        )
        gallery_i2i.change(
            sync_meta,
            inputs=[gallery_i2i, meta_state_i2i, sel_idx_i2i],
            outputs=[meta_view_i2i, sel_idx_i2i],
        )

    with gr.Tab("inpaint") as inp_tab:
        with gr.Row():
            with gr.Column(scale=7):
                with gr.Group():
                    prompt_inp = gr.Textbox(lines=4, label="Positive prompt")
                    with gr.Accordion("Negative prompt", open=False):
                        negative_inp = gr.Textbox(
                            lines=2, label="Negative prompt", show_label=False
                        )
                    with gr.Accordion("Meta prompt", open=False, visible=default_enable_lora) as meta_acc_inp:
                        meta_prompt_inp = gr.Textbox(
                            lines=2, label="Meta prompt", show_label=False
                        )
                        with gr.Row():
                            lora_name_inp = gr.Dropdown(
                                choices=default_lora_list,
                                value=default_lora_list[0] if default_lora_list else None,
                                allow_custom_value=True,
                                label="LoRA Name",
                            )
                            lora_strength_inp = gr.Number(value=1.0, label="LoRA Strength")
                            with gr.Group():
                                add_lora_btn_inp = gr.Button("Add LoRA")
                                reload_lora_btn_inp = gr.Button("Reload List")
                        add_lora_btn_inp.click(
                            _add_lora_to_meta,
                            inputs=[meta_prompt_inp, lora_name_inp, lora_strength_inp],
                            outputs=meta_prompt_inp,
                        )
            with gr.Column(scale=1):
                with gr.Tab("Generate"):
                    gen_inp_btn = gr.Button("Generate", variant="primary")
                    progress_inp = gr.Textbox(
                        "", label="Status", interactive=False, lines=1
                    )
                with gr.Tab("Replace"):
                    with gr.Group():
                        with gr.Row():
                            find_inp = gr.Textbox(
                                placeholder="Find",
                                show_label=False,
                                lines=1,
                                min_width=80,
                            )
                            repl_inp = gr.Textbox(
                                placeholder="Replace",
                                show_label=False,
                                lines=1,
                                min_width=80,
                            )
                        chk_inp = gr.CheckboxGroup(
                            choices=["Positive", "Negative"],
                            value=["Positive"],
                            show_label=False,
                        )
                        rep_btn_inp = gr.Button("Replace")

        with gr.Row():
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("Picture 1 (Main Input)"):
                        img_mask_inp = gr.ImageEditor(
                            label="Picture 1 (Main Input, Draw mask)",
                            type="pil",
                            layers=False,
                            show_fullscreen_button=True,
                            height=800,
                        )
                    with gr.TabItem("Picture 2 (Ref. 1)"):
                        ref_img1_inp = gr.Image(label="Picture 2 (Ref. 1)", type="pil")
                        enable_ref1_inp = gr.Checkbox(
                            label="Enable Ref. 1", value=False
                        )
                    with gr.TabItem("Picture 3 (Ref. 2)"):
                        ref_img2_inp = gr.Image(label="Picture 3 (Ref. 2)", type="pil")
                        enable_ref2_inp = gr.Checkbox(
                            label="Enable Ref. 2", value=False
                        )
                    with gr.TabItem("Picture 4 (Ref. 3)"):
                        ref_img3_inp = gr.Image(label="Picture 4 (Ref. 3)", type="pil")
                        enable_ref3_inp = gr.Checkbox(
                            label="Enable Ref. 3", value=False
                        )

                with gr.Group():
                    with gr.Row():
                        width_inp = gr.Slider(
                            config.get("default_width_min", 512),
                            config.get("default_width_max", 4096),
                            value=config.get("default_width", 1328),
                            step=config.get("default_width_step", 64),
                            label="Width",
                        )
                        height_inp = gr.Slider(
                            config.get("default_height_min", 512),
                            config.get("default_height_max", 4096),
                            value=config.get("default_height", 1328),
                            step=config.get("default_height_step", 64),
                            label="Height",
                        )
                    with gr.Row():
                        gr.Button("⇅").click(
                            _swap_dims,
                            inputs=[width_inp, height_inp],
                            outputs=[width_inp, height_inp],
                        )
                        gr.Button("↲").click(
                            _import_dims,
                            inputs=img_mask_inp,
                            outputs=[width_inp, height_inp],
                        )
                        gr.Button("⤧").click(
                            rescale_dims,
                            inputs=[width_inp, height_inp, vit_dd],
                            outputs=[width_inp, height_inp],
                        )

                with gr.Row():
                    bsz_inp = gr.Slider(
                        config.get("default_batch_size_min", 1),
                        config.get("default_batch_size_max", 8),
                        value=config.get("default_batch_size", 8),
                        step=config.get("default_batch_size_step", 1),
                        label="Batch size",
                    )
                    bcnt_inp = gr.Slider(
                        config.get("default_batch_count_min", 1),
                        config.get("default_batch_count_max", 100),
                        value=config.get("default_batch_count", 1),
                        step=config.get("default_batch_count_step", 1),
                        label="Batch count",
                    )

                with gr.Group():
                    with gr.Row():
                        denoising_strength_inp = gr.Slider(
                            config.get("default_denoising_strength_min", 0.0),
                            config.get("default_denoising_strength_max", 1.0),
                            value=config.get("default_denoising_strength", 1.0),
                            step=config.get("default_denoising_strength_step", 0.01),
                            label="Denoising Strength",
                        )
                        consistency_strength_inp = gr.Slider(
                            config.get("default_consistency_strength_min", -1.0),
                            config.get("default_consistency_strength_max", 1.0),
                            value=config.get("default_consistency_strength", 0.0),
                            step=config.get("default_consistency_strength_step", 0.01),
                            label="Consistency Strength",
                        )

                    with gr.Row():
                        cfg_inp = gr.Slider(
                            config.get("default_cfg_min", 0.0),
                            config.get("default_cfg_max", 20.0),
                            value=config.get("default_cfg", 1.0),
                            step=config.get("default_cfg_step", 0.1),
                            label="True CFG scale",
                        )
                        steps_inp = gr.Slider(
                            config.get("default_steps_min", 1),
                            config.get("default_steps_max", 100),
                            value=config.get("default_steps", 4),
                            step=config.get("default_steps_step", 1),
                            label="Steps",
                        )

                    with gr.Row(equal_height=True):
                        seed_inp = gr.Number(
                            value=-1, label="Seed", precision=0, scale=2
                        )
                        with gr.Column():
                            gr.Button("🎲").click(lambda: -1, None, seed_inp)
                            reuse_seed_inp_btn = gr.Button("♻")

            with gr.Column(scale=3):
                gallery_inp = gr.Gallery(
                    label="Result Images",
                    format="png",
                    object_fit="contain",
                    show_label=False,
                    columns=config.get("gallery_columns", 4),
                    preview=True,
                    interactive=False,
                )
                with gr.Row():
                    send_inp_i2i_btn = gr.Button("Send to i2i")
                    send_inp_inp_btn = gr.Button("Send to inpaint")
                    send_inp_cap_btn = gr.Button("Send to caption")
                    send_inp_vqa_btn = gr.Button("Send to VQA")
                dest_slot_inp = gr.Radio(
                    ["input", "ref1", "ref2", "ref3"],
                    value="input",
                    label="Send-to target (for i2i / inpaint)",
                )
                meta_state_inp = gr.State([])
                meta_view_inp = gr.JSON(label="Metadata", show_label=False)
                sel_inp_path = gr.State("")
                sel_idx_inp = gr.State(0)

        gen_inp_btn.click(
            generate_inpaint,
            inputs=[
                mode_dd,
                base_pipe_dd,
                vlm_dd,
                vit_dd,
                img_mask_inp,
                enable_ref1_inp,
                ref_img1_inp,
                enable_ref2_inp,
                ref_img2_inp,
                enable_ref3_inp,
                ref_img3_inp,
                prompt_inp,
                negative_inp,
                meta_prompt_inp,
                cfg_inp,
                denoising_strength_inp,
                steps_inp,
                width_inp,
                height_inp,
                bsz_inp,
                bcnt_inp,
                sampler,
                seed_inp,
                consistency_strength_inp,
            ],
            outputs=[gallery_inp, meta_state_inp, progress_inp],
            show_progress_on=progress_inp,
            concurrency_id="gpu",
        )
        rep_btn_inp.click(
            _replace_prompts,
            inputs=[find_inp, repl_inp, chk_inp, prompt_inp, negative_inp],
            outputs=[prompt_inp, negative_inp],
        )
        reuse_seed_inp_btn.click(_extract_seed, meta_view_inp, seed_inp)
    gallery_inp.select(
        show_meta,
        inputs=meta_state_inp,
        outputs=[meta_view_inp, sel_inp_path, sel_idx_inp],
    )
    gallery_inp.change(
        sync_meta,
        inputs=[gallery_inp, meta_state_inp, sel_idx_inp],
        outputs=[meta_view_inp, sel_idx_inp],
    )

    with gr.Tab("vlm") as vlm_tab:
        with gr.Tabs():
            with gr.TabItem("caption"):
                with gr.Row():
                    with gr.Column():
                        img_cap = gr.Image(type="pil", label="Image")
                        with gr.Row():
                            send_cap_i2i_btn = gr.Button("Send to i2i")
                            send_cap_inp_btn = gr.Button("Send to inpaint")
                            send_cap_vqa_btn = gr.Button("Send to VQA")
                        dest_slot_cap = gr.Radio(
                            ["input", "ref1", "ref2", "ref3"],
                            value="input",
                            label="Send-to target (for i2i / inpaint)",
                        )
                    with gr.Column():
                        length_dd = gr.Dropdown(
                            choices=[
                                "default",
                                "very short",
                                "short",
                                "medium",
                                "long",
                                "very long",
                            ],
                            value="default",
                            label="Length (preset)",
                        )
                        word_slider = gr.Slider(
                            minimum=0,
                            maximum=5000,
                            step=50,
                            value=0,
                            label="Word limit (0 = ignore)",
                        )
                        prompt_cap = gr.Textbox(
                            label="Prompt",
                            lines=2,
                            show_copy_button=True,
                            value=build_caption_prompt("default", None),
                        )
                        with gr.Row(equal_height=True):
                            with gr.Column():
                                max_tkn_cap = gr.Slider(
                                    128,
                                    8192,
                                    value=1024,
                                    step=128,
                                    label="Max tokens",
                                )
                                cap_btn = gr.Button(
                                    "Generate Caption", variant="primary"
                                )
                            progress_cap = gr.Textbox(
                                "", label="Status", interactive=False, lines=2
                            )
                        cap_out = gr.Textbox(
                            label="Caption", lines=4, show_copy_button=True
                        )
                meta_state_cap = gr.State([])
                sel_cap_path = gr.State("")
                img_cap.change(
                    update_metadata_and_path,
                    inputs=img_cap,
                    outputs=[meta_state_cap, sel_cap_path],
                )

                length_dd.change(
                    build_caption_prompt,
                    inputs=[length_dd, word_slider],
                    outputs=prompt_cap,
                )
                word_slider.change(
                    build_caption_prompt,
                    inputs=[length_dd, word_slider],
                    outputs=prompt_cap,
                )
                cap_btn.click(
                    partial(vl_generate, pm),
                    inputs=[mode_dd, img_cap, prompt_cap, base_pipe_dd, vlm_dd, max_tkn_cap],
                    outputs=[cap_out, progress_cap],
                    api_name="vlm_caption",
                    show_progress_on=progress_cap,
                    concurrency_id="gpu",
                )

            with gr.TabItem("inference"):
                with gr.Row():
                    with gr.Column():
                        img_vqa = gr.Image(type="pil", label="Image")
                        with gr.Row():
                            send_vqa_i2i_btn = gr.Button("Send to i2i")
                            send_vqa_inp_btn = gr.Button("Send to inpaint")
                            send_vqa_cap_btn = gr.Button("Send to caption")
                        dest_slot_vqa = gr.Radio(
                            ["input", "ref1", "ref2", "ref3"],
                            value="input",
                            label="Send-to target (for i2i / inpaint)",
                        )
                    with gr.Column():
                        q_box = gr.Textbox(
                            label="Question", lines=2, show_copy_button=True
                        )
                        with gr.Row(equal_height=True):
                            with gr.Column():
                                max_tkn_vqa = gr.Slider(
                                    128,
                                    8192,
                                    value=1024,
                                    step=128,
                                    label="Max tokens",
                                )
                                ask_btn = gr.Button("Ask", variant="primary")
                            progress_vqa = gr.Textbox(
                                "", label="Status", interactive=False, lines=2
                            )
                        ans_out = gr.Textbox(
                            label="Answer", lines=4, show_copy_button=True
                        )
                meta_state_vqa = gr.State([])
                sel_vqa_path = gr.State("")
                img_vqa.change(
                    update_metadata_and_path,
                    inputs=img_vqa,
                    outputs=[meta_state_vqa, sel_vqa_path],
                )
                ask_btn.click(
                    partial(vl_generate, pm),
                    inputs=[mode_dd, img_vqa, q_box, base_pipe_dd, vlm_dd, max_tkn_vqa],
                    outputs=[ans_out, progress_vqa],
                    api_name="vlm_VQA",
                    show_progress_on=progress_vqa,
                    concurrency_id="gpu",
                )

    with gr.Tab("png info") as png_tab:
        with gr.Row():
            png_in = gr.Image(type="filepath", label="PNG")
            with gr.Column():
                with gr.Tabs():
                    with gr.TabItem("JSON"):
                        meta_json = gr.JSON(label="Metadata (JSON)", show_label=False)
                    with gr.TabItem("Text"):
                        meta_text = gr.Text(
                            label="Metadata (raw)",
                            lines=12,
                            interactive=False,
                            show_copy_button=True,
                        )

                with gr.Row():
                    send_info_t2i_btn = gr.Button("Send to t2i")
                    send_info_i2i_btn = gr.Button("Send to i2i")
                    send_info_inp_btn = gr.Button("Send to inpaint")
                with gr.Row():
                    send_info_cap_btn = gr.Button("Send to caption")
                    send_info_vqa_btn = gr.Button("Send to VQA")
                dest_slot_info = gr.Radio(
                    ["input", "ref1", "ref2", "ref3"],
                    value="input",
                    label="Send-to target (for i2i / inpaint)",
                )

    png_in.change(extract_meta, png_in, [meta_text, meta_json])

    tab_list = [t2i_tab, i2i_tab, inp_tab, vlm_tab, png_tab]
    upper_dds = [base_pipe_dd, vlm_dd, vit_dd, sampler]
    meta_components = [
        meta_acc_t2i,
        meta_acc_i2i,
        meta_acc_inp,
        lora_name_t2i,
        lora_name_i2i,
        lora_name_inp,
    ]
    mode_dd.select(
        _apply_mode,
        [mode_dd, base_pipe_dd, vlm_dd, vit_dd, sampler, lora_name_t2i, lora_name_i2i, lora_name_inp],
        tab_list + upper_dds + meta_components,
    )

    reload_lora_inputs = [mode_dd, lora_name_t2i, lora_name_i2i, lora_name_inp]
    reload_lora_outputs = [lora_name_t2i, lora_name_i2i, lora_name_inp]
    reload_lora_btn_t2i.click(_reload_lora_list, inputs=reload_lora_inputs, outputs=reload_lora_outputs)
    reload_lora_btn_i2i.click(_reload_lora_list, inputs=reload_lora_inputs, outputs=reload_lora_outputs)
    reload_lora_btn_inp.click(_reload_lora_list, inputs=reload_lora_inputs, outputs=reload_lora_outputs)

    # send buttons

    send_info_t2i_btn.click(
        apply_config_to_t2i,
        inputs=[png_in, meta_text],
        outputs=[
            prompt_t2i,
            negative_t2i,
            meta_prompt_t2i,
            cfg_t2i,
            steps_t2i,
            width_t2i,
            height_t2i,
            sampler,
            seed_box_t2i,
        ],
    )

    send_info_i2i_btn.click(
        apply_config_to_i2i,
        inputs=[png_in, meta_text, dest_slot_info],
        outputs=[
            init_img_i2i,
            ref_img1_i2i,
            ref_img2_i2i,
            ref_img3_i2i,
            prompt_i2i,
            negative_i2i,
            meta_prompt_i2i,
            cfg_i2i,
            denoising_strength_i2i,
            consistency_strength_i2i,
            steps_i2i,
            width_i2i,
            height_i2i,
            sampler,
            seed_i2i,
        ],
    )

    send_info_inp_btn.click(
        apply_config_to_inpaint,
        inputs=[png_in, meta_text, dest_slot_info],
        outputs=[
            img_mask_inp,
            ref_img1_inp,
            ref_img2_inp,
            ref_img3_inp,
            prompt_inp,
            negative_inp,
            meta_prompt_inp,
            cfg_inp,
            denoising_strength_inp,
            consistency_strength_inp,
            steps_inp,
            width_inp,
            height_inp,
            sampler,
            seed_inp,
        ],
    )

    send_t2i_i2i_btn.click(
        apply_config_to_i2i,
        inputs=[sel_t2i_path, meta_view_t2i, dest_slot_t2i],
        outputs=[
            init_img_i2i,
            ref_img1_i2i,
            ref_img2_i2i,
            ref_img3_i2i,
            prompt_i2i,
            negative_i2i,
            meta_prompt_i2i,
            cfg_i2i,
            denoising_strength_i2i,
            consistency_strength_i2i,
            steps_i2i,
            width_i2i,
            height_i2i,
            sampler,
            seed_i2i,
        ],
    )
    send_t2i_inp_btn.click(
        apply_config_to_inpaint,
        inputs=[sel_t2i_path, meta_view_t2i, dest_slot_t2i],
        outputs=[
            img_mask_inp,
            ref_img1_inp,
            ref_img2_inp,
            ref_img3_inp,
            prompt_inp,
            negative_inp,
            meta_prompt_inp,
            cfg_inp,
            denoising_strength_inp,
            consistency_strength_inp,
            steps_inp,
            width_inp,
            height_inp,
            sampler,
            seed_inp,
        ],
    )
    send_i2i_i2i_btn.click(
        apply_config_to_i2i,
        inputs=[sel_i2i_path, meta_view_i2i, dest_slot_i2i],
        outputs=[
            init_img_i2i,
            ref_img1_i2i,
            ref_img2_i2i,
            ref_img3_i2i,
            prompt_i2i,
            negative_i2i,
            meta_prompt_i2i,
            cfg_i2i,
            denoising_strength_i2i,
            consistency_strength_i2i,
            steps_i2i,
            width_i2i,
            height_i2i,
            sampler,
            seed_i2i,
        ],
    )
    send_i2i_inp_btn.click(
        apply_config_to_inpaint,
        inputs=[sel_i2i_path, meta_view_i2i, dest_slot_i2i],
        outputs=[
            img_mask_inp,
            ref_img1_inp,
            ref_img2_inp,
            ref_img3_inp,
            prompt_inp,
            negative_inp,
            meta_prompt_inp,
            cfg_inp,
            denoising_strength_inp,
            consistency_strength_inp,
            steps_inp,
            width_inp,
            height_inp,
            sampler,
            seed_inp,
        ],
    )
    send_inp_i2i_btn.click(
        apply_config_to_i2i,
        inputs=[sel_inp_path, meta_view_inp, dest_slot_inp],
        outputs=[
            init_img_i2i,
            ref_img1_i2i,
            ref_img2_i2i,
            ref_img3_i2i,
            prompt_i2i,
            negative_i2i,
            meta_prompt_i2i,
            cfg_i2i,
            denoising_strength_i2i,
            consistency_strength_i2i,
            steps_i2i,
            width_i2i,
            height_i2i,
            sampler,
            seed_i2i,
        ],
    )
    send_inp_inp_btn.click(
        apply_config_to_inpaint,
        inputs=[sel_inp_path, meta_view_inp, dest_slot_inp],
        outputs=[
            img_mask_inp,
            ref_img1_inp,
            ref_img2_inp,
            ref_img3_inp,
            prompt_inp,
            negative_inp,
            meta_prompt_inp,
            cfg_inp,
            denoising_strength_inp,
            consistency_strength_inp,
            steps_inp,
            width_inp,
            height_inp,
            sampler,
            seed_inp,
        ],
    )
    send_cap_i2i_btn.click(
        apply_config_to_i2i,
        inputs=[sel_cap_path, meta_state_cap, dest_slot_cap],
        outputs=[
            init_img_i2i,
            ref_img1_i2i,
            ref_img2_i2i,
            ref_img3_i2i,
            prompt_i2i,
            negative_i2i,
            meta_prompt_i2i,
            cfg_i2i,
            denoising_strength_i2i,
            consistency_strength_i2i,
            steps_i2i,
            width_i2i,
            height_i2i,
            sampler,
            seed_i2i,
        ],
    )
    send_cap_inp_btn.click(
        apply_config_to_inpaint,
        inputs=[sel_cap_path, meta_state_cap, dest_slot_cap],
        outputs=[
            img_mask_inp,
            ref_img1_inp,
            ref_img2_inp,
            ref_img3_inp,
            prompt_inp,
            negative_inp,
            meta_prompt_inp,
            cfg_inp,
            denoising_strength_inp,
            consistency_strength_inp,
            steps_inp,
            width_inp,
            height_inp,
            sampler,
            seed_inp,
        ],
    )
    send_vqa_i2i_btn.click(
        apply_config_to_i2i,
        inputs=[sel_vqa_path, meta_state_vqa, dest_slot_vqa],
        outputs=[
            init_img_i2i,
            ref_img1_i2i,
            ref_img2_i2i,
            ref_img3_i2i,
            prompt_i2i,
            negative_i2i,
            meta_prompt_i2i,
            cfg_i2i,
            denoising_strength_i2i,
            consistency_strength_i2i,
            steps_i2i,
            width_i2i,
            height_i2i,
            sampler,
            seed_i2i,
        ],
    )
    send_vqa_inp_btn.click(
        apply_config_to_inpaint,
        inputs=[sel_vqa_path, meta_state_vqa, dest_slot_vqa],
        outputs=[
            img_mask_inp,
            ref_img1_inp,
            ref_img2_inp,
            ref_img3_inp,
            prompt_inp,
            negative_inp,
            meta_prompt_inp,
            cfg_inp,
            denoising_strength_inp,
            consistency_strength_inp,
            steps_inp,
            width_inp,
            height_inp,
            sampler,
            seed_inp,
        ],
    )

    # send image buttons

    send_info_cap_btn.click(send_image, inputs=png_in, outputs=img_cap)
    send_info_vqa_btn.click(send_image, inputs=png_in, outputs=img_vqa)
    send_vqa_cap_btn.click(send_image, inputs=img_vqa, outputs=img_cap)
    send_cap_vqa_btn.click(send_image, inputs=img_cap, outputs=img_vqa)
    send_t2i_cap_btn.click(send_image, inputs=sel_t2i_path, outputs=img_cap)
    send_t2i_vqa_btn.click(send_image, inputs=sel_t2i_path, outputs=img_vqa)
    send_i2i_cap_btn.click(send_image, inputs=sel_i2i_path, outputs=img_cap)
    send_i2i_vqa_btn.click(send_image, inputs=sel_i2i_path, outputs=img_vqa)
    send_inp_cap_btn.click(send_image, inputs=sel_inp_path, outputs=img_cap)
    send_inp_vqa_btn.click(send_image, inputs=sel_inp_path, outputs=img_vqa)

demo.launch(
    server_name=server_name,
    server_port=port,
    # share=True
)
