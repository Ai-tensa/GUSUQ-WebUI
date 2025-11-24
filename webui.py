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
from PIL import Image, PngImagePlugin, ImageChops
from pipeline_manager import PipelineManager
from constants import (
    SAMPLERS,
    load_vit_model_table,
)
from cap import build_caption_prompt, vl_generate
from metadata import (
    apply_config_to_t2i,
    apply_config_to_i2i,
    apply_config_to_inpaint,
    extract_meta,
    show_meta,
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
    "--opt-pol-yaml",
    type=Path,
    default=Path("config/opt_pol.yaml"),
    help="Path to optimization-policy YAML",
)
parser.add_argument(
    "--vit-models-yaml",
    type=Path,
    default=Path("config/vit_models.yaml"),
    help="Path to ViT models YAML",
)
args = parser.parse_args()

if not args.user_config_yaml.exists():
    raise FileNotFoundError(f"User config not found: {args.user_config_yaml}")
with open(args.user_config_yaml, "r") as f:
    config = yaml.safe_load(f)

if not args.opt_pol_yaml.exists():
    raise FileNotFoundError(
        f"Optimization policy config not found: {args.opt_pol_yaml}"
    )
with open(args.opt_pol_yaml, "r") as f:
    opt_pol_cfg = yaml.safe_load(f)
if not args.vit_models_yaml.exists():
    raise FileNotFoundError(f"ViT models config not found: {args.vit_models_yaml}")
vit_model_table = load_vit_model_table(args.vit_models_yaml)
BASE_OUTPUT_DIR = Path(config.get("output_dir", "outputs"))

torch.backends.cuda.matmul.allow_tf32 = True
pm = PipelineManager(opt_pol_cfg, vit_model_table)


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


def _rescale_dims(w, h, model_name):
    if not isinstance(w, (int, float)) or not isinstance(h, (int, float)):
        return gr.update(), gr.update()
    if "edit" not in str(model_name).lower():
        gr.Info("Rescaling is only available for Edit models.")
        return gr.update(), gr.update()

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


def generate_t2i(
    model, prompt, negative, cfg, steps, width, height, bsz, bcnt, sampler, seed
):
    base_seed = random.randint(0, 2**32 - 1) if seed == -1 else int(seed)
    negative = negative if negative.strip() != "" else None

    print("RSS before get pipe:", rss_mb(), "MB")
    pipe = pm.get_pipeline(model, sampler)
    print("RSS after get pipe :", rss_mb(), "MB")
    gens = [
        torch.Generator(device=pipe.transformer.device).manual_seed(base_seed + i)
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
    for i in tqdm(range(bcnt), desc="Batches"):
        result = pipe(
            prompt=prompt,
            negative_prompt=negative,
            true_cfg_scale=cfg,
            num_inference_steps=steps,
            width=width,
            height=height,
            num_images_per_prompt=bsz,
            generator=gens[i * bsz : (i + 1) * bsz],
        ).images
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        for j, img in enumerate(result):
            s = base_seed + i * bsz + j
            meta = dict(
                ts=ts,
                model=model,
                prompt=prompt,
                negative=negative,
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
    return images, meta_list


def generate_i2i(
    model,
    input_image,
    enable_ref1,
    ref_image1,
    enable_ref2,
    ref_image2,
    enable_ref3,
    ref_image3,
    prompt,
    negative,
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
    is_edit_model = vit_model_table[model]["edit"]
    if consistency_strength != 0.0 and not is_edit_model:
        gr.Info("Consistency strength is only supported for Edit models. Ignoring it.")
    if strength != 1.0 and is_edit_model:
        gr.Info("Strength is not used for Edit models. Ignoring it.")
    base_seed = random.randint(0, 2**32 - 1) if seed == -1 else int(seed)
    negative = negative if negative.strip() != "" else None
    print("RSS before get pipe:", rss_mb(), "MB")
    pipe = pm.get_pipeline(model, sampler, mode="i2i")
    print("RSS after get pipe :", rss_mb(), "MB")
    gens = [
        torch.Generator(device=pipe.transformer.device).manual_seed(base_seed + i)
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
    for i in tqdm(range(bcnt), desc="Batches"):
        result = (
            pipe(
                image=input_images,
                prompt=prompt,
                negative_prompt=negative,
                true_cfg_scale=cfg,
                strength=strength,
                num_inference_steps=steps,
                width=width,
                height=height,
                num_images_per_prompt=bsz,
                generator=gens[i * bsz : (i + 1) * bsz],
            ).images
            if not is_edit_model
            else pipe(
                image=input_images,
                prompt=prompt,
                negative_prompt=negative,
                true_cfg_scale=cfg,
                num_inference_steps=steps,
                width=width,
                height=height,
                num_images_per_prompt=bsz,
                generator=gens[i * bsz : (i + 1) * bsz],
                consistency_strength=consistency_strength,
            ).images
        )
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        for j, img in enumerate(result):
            s = base_seed + i * bsz + j
            meta = dict(
                ts=ts,
                model=model,
                prompt=prompt,
                negative=negative,
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
    return images, meta_list


def generate_inpaint(
    model,
    editor_val,
    enable_ref1,
    ref_image1,
    enable_ref2,
    ref_image2,
    enable_ref3,
    ref_image3,
    prompt,
    negative,
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
    is_edit_model = vit_model_table[model]["edit"]
    if consistency_strength != 0.0 and not is_edit_model:
        gr.Info("Consistency strength is only supported for Edit models. Ignoring it.")
    input_image = editor_val["background"].convert("RGB")
    mask_image = _extract_mask(editor_val)
    base_seed = random.randint(0, 2**32 - 1) if seed == -1 else int(seed)
    negative = negative if negative.strip() != "" else None
    pipe = pm.get_pipeline(model, sampler, mode="inpaint")
    gens = [
        torch.Generator(device=pipe.transformer.device).manual_seed(base_seed + i)
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
    for i in tqdm(range(bcnt), desc="Batches"):
        result = (
            pipe(
                image=input_images,
                mask_image=mask_image,
                prompt=prompt,
                negative_prompt=negative,
                true_cfg_scale=cfg,
                strength=strength,
                num_inference_steps=steps,
                width=width,
                height=height,
                num_images_per_prompt=bsz,
                generator=gens[i * bsz : (i + 1) * bsz],
            ).images
            if not is_edit_model
            else pipe(
                image=input_images,
                mask_image=mask_image,
                prompt=prompt,
                negative_prompt=negative,
                true_cfg_scale=cfg,
                strength=strength,
                num_inference_steps=steps,
                width=width,
                height=height,
                num_images_per_prompt=bsz,
                generator=gens[i * bsz : (i + 1) * bsz],
                consistency_strength=consistency_strength,
            ).images
        )
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        for j, img in enumerate(result):
            s = base_seed + i * bsz + j
            meta = dict(
                ts=ts,
                model=model,
                prompt=prompt,
                negative=negative,
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
    return images, meta_list


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
        vit_model_list = list(vit_model_table.keys())
        model_dd = gr.Dropdown(
            vit_model_list,
            value=vit_model_list[0],
            label="Model",
            scale=7,
        )
        sampler = gr.Dropdown(
            list(SAMPLERS.keys()), value="FlowMatchEuler", label="Sampler"
        )

    with gr.Tab("t2i"):
        with gr.Row():
            with gr.Column(scale=7):
                prompt_t2i = gr.Textbox(lines=4, label="Positive prompt")
                negative_t2i = gr.Textbox(lines=1, label="Negative prompt")
            with gr.Column(scale=1):
                with gr.Tab("Generate"):
                    gen_btn_t2i = gr.Button("Generate", variant="primary")
                    progress_t2i = gr.HTML(
                        "<div style='width:100%;height:72px;'></div>",
                        elem_id="progress_t2i",
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
                        gr.Button("⇅", elem_id="btn_swap_dims").click(
                            _swap_dims,
                            inputs=[width_t2i, height_t2i],
                            outputs=[width_t2i, height_t2i],
                        )
                        gr.Button("⤧", elem_id="btn_rescale_dims").click(
                            _rescale_dims,
                            inputs=[width_t2i, height_t2i, model_dd],
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
                            gr.Button("🎲", elem_id="btn_random_seed").click(
                                lambda: -1, None, seed_box_t2i
                            )
                            reuse_seed_btn_t2i = gr.Button(
                                "♻", elem_id="btn_reuse_seed"
                            )

            with gr.Column(scale=3):
                gallery_t2i = gr.Gallery(
                    label="Result Images",
                    format="png",
                    object_fit="contain",
                    show_label=False,
                    columns=config.get("gallery_columns", 4),
                    preview=True,
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

        gen_btn_t2i.click(
            generate_t2i,
            inputs=[
                model_dd,
                prompt_t2i,
                negative_t2i,
                cfg_t2i,
                steps_t2i,
                width_t2i,
                height_t2i,
                bsz_t2i,
                bcnt_t2i,
                sampler,
                seed_box_t2i,
            ],
            outputs=[gallery_t2i, meta_state_t2i],
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
            show_meta, inputs=meta_state_t2i, outputs=[meta_view_t2i, sel_t2i_path]
        )

    with gr.Tab("i2i"):
        with gr.Row():
            with gr.Column(scale=7):
                prompt_i2i = gr.Textbox(lines=4, label="Positive prompt")
                negative_i2i = gr.Textbox(lines=1, label="Negative prompt")
            with gr.Column(scale=1):
                with gr.Tab("Generate"):
                    gen_i2i_btn = gr.Button("Generate", variant="primary")
                    progress_i2i = gr.HTML(
                        "<div style='width:100%;height:72px;'></div>",
                        elem_id="progress_i2i",
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
                            _rescale_dims,
                            inputs=[width_i2i, height_i2i, model_dd],
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
                    with gr.Row():
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

        gen_i2i_btn.click(
            generate_i2i,
            inputs=[
                model_dd,
                init_img_i2i,
                enable_ref1_i2i,
                ref_img1_i2i,
                enable_ref2_i2i,
                ref_img2_i2i,
                enable_ref3_i2i,
                ref_img3_i2i,
                prompt_i2i,
                negative_i2i,
                cfg_i2i,
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
            outputs=[gallery_i2i, meta_state_i2i],
            show_progress_on=progress_i2i,
            concurrency_id="gpu",
        )
        rep_btn_i2i.click(
            _replace_prompts,
            inputs=[find_i2i, repl_i2i, chk_i2i, prompt_i2i, negative_i2i],
            outputs=[prompt_i2i, negative_i2i],
        )
        reuse_seed_i2i_btn.click(_extract_seed, meta_view_i2i, seed_i2i)
        gallery_i2i.select(
            show_meta, inputs=meta_state_i2i, outputs=[meta_view_i2i, sel_i2i_path]
        )

    with gr.Tab("inpaint"):
        with gr.Row():
            with gr.Column(scale=7):
                prompt_inp = gr.Textbox(lines=4, label="Positive prompt")
                negative_inp = gr.Textbox(lines=1, label="Negative prompt")
            with gr.Column(scale=1):
                with gr.Tab("Generate"):
                    gen_inp_btn = gr.Button("Generate", variant="primary")
                    progress_inp = gr.HTML(
                        "<div style='width:100%;height:72px;'></div>",
                        elem_id="progress_inpaint",
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
                            _rescale_dims,
                            inputs=[width_inp, height_inp, model_dd],
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

        gen_inp_btn.click(
            generate_inpaint,
            inputs=[
                model_dd,
                img_mask_inp,
                enable_ref1_inp,
                ref_img1_inp,
                enable_ref2_inp,
                ref_img2_inp,
                enable_ref3_inp,
                ref_img3_inp,
                prompt_inp,
                negative_inp,
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
            outputs=[gallery_inp, meta_state_inp],
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
        show_meta, inputs=meta_state_inp, outputs=[meta_view_inp, sel_inp_path]
    )

    with gr.Tab("vlm"):
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
                        cap_btn = gr.Button("Generate Caption", variant="primary")
                        progress_cap = gr.HTML(
                            "<div style='width:100%;height:90px;'></div>",
                            elem_id="progress_cap",
                            padding=False,
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
                    inputs=[img_cap, prompt_cap],
                    outputs=cap_out,
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
                        ask_btn = gr.Button("Ask", variant="primary")
                        progress_vqa = gr.HTML(
                            "<div style='width:100%;height:90px;'></div>",
                            elem_id="progress_vqa",
                            padding=False,
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
                    inputs=[img_vqa, q_box],
                    outputs=ans_out,
                    api_name="vlm_VQA",
                    show_progress_on=progress_vqa,
                    concurrency_id="gpu",
                )

    with gr.Tab("png info"):
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

    # send buttons

    send_info_t2i_btn.click(
        apply_config_to_t2i,
        inputs=[png_in, meta_text],
        outputs=[
            prompt_t2i,
            negative_t2i,
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

demo.launch()
