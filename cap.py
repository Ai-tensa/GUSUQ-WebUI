import torch
from PIL import Image
from time import perf_counter
from pipeline_manager import PipelineManager
from utils import release_memory_resources

def vl_generate(pm: PipelineManager, mode: str, image: Image.Image, prompt: str,
                 base_model_key: str = None, vlm_model_key: str = None, max_new_tokens: int = 1024, temperature: float = 0.7) -> str:
    start_time = perf_counter()
    msgs = [{"role": "user",
             "content": [{"type": "image", "image": image},
                         {"type": "text", "text": prompt}]}]
    opt_policy = pm.opt_pol_cfg.get("opt_policy", None)
    te, tkn, proc = pm.get_vlm(mode, vlm_model_key, base_model_key=base_model_key)
    inputs = proc.apply_chat_template(
        msgs, add_generation_prompt=True,
        tokenize=True, return_dict=True,
        return_tensors="pt")

    device = "cuda"
    if opt_policy == "high_vram":
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        te.to(device, non_blocking=True)

    elif opt_policy == "mid_vram" or opt_policy == "low_vram":
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        if not pm.is_set_te_offload:
            te.to(device, non_blocking=True)

    with torch.no_grad():
        outs = te.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=temperature)
    gen_ids = outs[:, inputs["input_ids"].shape[1]:]
    if opt_policy == "high_vram":
        te.to("cpu")
    elif not pm.is_set_te_offload and (opt_policy == "mid_vram" or opt_policy == "low_vram"):
        te.to("cpu")

    release_memory_resources()
    elapsed = perf_counter() - start_time
    status = f"Inference completed in {elapsed:.1f} seconds."
    return tkn.batch_decode(
        gen_ids, skip_special_tokens=True,
        clean_up_tokenization_spaces=True)[0], status

def build_caption_prompt(length: str, word_limit: int) -> str:
    if word_limit and word_limit > 0:
        return f"Write a detailed description for this image in {word_limit} words or less."
    if length != "default":
        return f"Write a {length} detailed description for this image."
    return "Write a detailed description for this image."