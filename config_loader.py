from typing import Any
from pathlib import Path
import shutil
import yaml


def load_yaml_config(path: Path) -> dict:
    """ Generic YAML loader for configuration files. """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return raw or {}


def ensure_yaml_from_sample(path: Path, sample_path: Path, label: str) -> Path:
    """Create a missing YAML config by copying a sample file."""
    if path.exists():
        return path

    if not sample_path.exists():
        raise FileNotFoundError(
            f"{label} config not found: {path} (sample missing: {sample_path})"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(sample_path, path)
    print(f"[config] Created missing {label} config: {path} <- {sample_path}")
    return path

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
            "arch": item.get("arch", "Qwen-Image"),
        }
    return table


def load_base_pipeline_table(yaml_path: Path) -> dict[str, dict[str, Any]]:
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f) or []
    table = {}
    for item in raw:
        name = item["name"]
        entry = {k: v for k, v in item.items() if k != "name"}
        entry.setdefault("arch", "All")
        table[name] = entry
    return table


def filter_models(model_table, allowed_arch, allowed_variants):
    filtered = []
    model_list = list(model_table.keys())
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
