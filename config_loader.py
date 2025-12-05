from typing import Any
from pathlib import Path
import yaml

def load_mode_config(path: Path) -> dict:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return raw or {}

def load_vlm_model_table(yaml_path: Path) -> dict[str, dict[str, Any]]:
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f) or []
    table = {}
    for arch, cfg in raw.items():
        for item in cfg.get("models", []):
            name = item["name"]
            entry = {
                "arch": arch,
                "model_class": cfg["model_class"],
                "processor_class": cfg["processor_class"],
                "tokenizer_class": cfg["tokenizer_class"],
                "id": item["path_or_id"],
                "variants": item["variants"],
                "dtype": item.get("dtype", "bfloat16"),
            }
            table[name] = entry
    return table


def load_vit_model_table(yaml_path: Path) -> dict[str, dict[str, Any]]:
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f) or []
    table = {}
    for item in raw:
        name = item["name"]
        table[name] = {
            "path": item["path_or_url"],
            "edit": bool(item.get("edit", False)),
        }
    return table


def filter_models(model_list, model_table, allowed_arch, allowed_variants):
    filtered = []
    for name in model_list:
        entry = model_table[name]
        if allowed_arch != "All" and entry["arch"] != allowed_arch:
            continue

        if (
            allowed_variants != "All"
            and entry.get("variants", None) not in allowed_variants
        ):
            continue
        filtered.append(name)
    return filtered
