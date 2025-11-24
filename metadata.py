import gradio as gr
import json
import os
from PIL import Image
from pathlib import Path


def show_meta(evt: gr.SelectData, metas):
    return metas[evt.index] if 0 <= evt.index < len(metas) else {}, evt.value["image"][
        "path"
    ]


_dest_options = ["input", "ref1", "ref2", "ref3"]


def apply_config_to_t2i(path, meta_str):
    if isinstance(meta_str, gr.Json):
        meta_str = meta_str.value
    if not path or not meta_str or not os.path.exists(path):
        return [gr.update()] * 8
    if isinstance(meta_str, dict):
        meta = meta_str
    else:
        meta = json.loads(meta_str)
    return [
        gr.update(value=meta.get("prompt", "")),
        gr.update(value=meta.get("negative", "")),
        gr.update(value=float(meta.get("cfg", 1.0))),
        gr.update(value=int(meta.get("steps", 4))),
        gr.update(value=int(meta.get("width", 1328))),
        gr.update(value=int(meta.get("height", 1328))),
        gr.update(value=meta.get("sampler", "FlowMatchEuler")),
        gr.update(value=int(meta.get("seed", -1))),
    ]


def apply_config_to_i2i(path, meta_str, dest="input"):
    if not path or not os.path.exists(path):
        imgs = [gr.update()] * len(_dest_options)
    else:
        img = Image.open(path).convert("RGB")
        imgs = [img if d == dest else gr.update() for d in _dest_options]
    if isinstance(meta_str, gr.Json):
        meta_str = meta_str.value
    if not meta_str:
        configs = [gr.update()] * 10
        return imgs + configs
    if isinstance(meta_str, dict):
        meta = meta_str
    else:
        meta = json.loads(meta_str)
    configs = [
        gr.update(value=meta.get("prompt", "")),
        gr.update(value=meta.get("negative", "")),
        gr.update(value=float(meta.get("cfg", 1.0))),
        gr.update(value=float(meta.get("strength", 1.0))),
        gr.update(value=float(meta.get("consistency_strength", 0.0))),
        gr.update(value=int(meta.get("steps", 4))),
        gr.update(value=int(meta.get("width", 1328))),
        gr.update(value=int(meta.get("height", 1328))),
        gr.update(value=meta.get("sampler", "FlowMatchEuler")),
        gr.update(value=int(meta.get("seed", -1))),
    ]
    return imgs + configs


def apply_config_to_inpaint(path, meta_str, dest="input"):
    if not path or not os.path.exists(path):
        imgs = [gr.update()] * len(_dest_options)
    else:
        img = Image.open(path).convert("RGB")
        imgs = [img if d == dest else gr.update() for d in _dest_options]
        imgs[0] = {"background": imgs[0], "layers": [], "composite": imgs[0]} if dest == "input" else gr.update()
    if isinstance(meta_str, gr.Json):
        meta_str = meta_str.value
    if not meta_str:
        configs = [gr.update()] * 10
        return imgs + configs
    if isinstance(meta_str, dict):
        meta = meta_str
    else:
        meta = json.loads(meta_str)
    configs = [
        gr.update(value=meta.get("prompt", "")),
        gr.update(value=meta.get("negative", "")),
        gr.update(value=float(meta.get("cfg", 1.0))),
        gr.update(value=float(meta.get("strength", 1.0))),
        gr.update(value=float(meta.get("consistency_strength", 0.0))),
        gr.update(value=int(meta.get("steps", 4))),
        gr.update(value=int(meta.get("width", 1328))),
        gr.update(value=int(meta.get("height", 1328))),
        gr.update(value=meta.get("sampler", "FlowMatchEuler")),
        gr.update(value=int(meta.get("seed", -1))),
    ]
    return imgs + configs


def extract_meta(image):
    if not image:
        return "", {}
    if isinstance(image, Path) or isinstance(image, str):
        im = Image.open(image)
    else:
        im = image
    raw = im.info.get("parameters", "")
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        obj = {}
    return raw, obj
