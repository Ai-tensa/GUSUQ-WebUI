import torch
from PIL import Image
from pipeline_manager import PipelineManager
from utils import release_memory_resources

def vl_generate(pm: PipelineManager, image: Image.Image, prompt: str,
                 max_new_tokens: int = 1024, temperature: float = 0.7) -> str:
    msgs = [{"role": "user",
             "content": [{"type": "image", "image": image},
                         {"type": "text", "text": prompt}]}]
    opt_policy = pm.opt_pol_cfg.get("opt_policy", None)
    pm._load_vlm()
    proc = pm.vision_processor
    tokenizer = pm.tokenizer
    inputs = proc.apply_chat_template(
        msgs, add_generation_prompt=True,
        tokenize=True, return_dict=True,
        return_tensors="pt")

    if opt_policy == "high_vram":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        pm.text_encoder.to(device, non_blocking=True)

    with torch.no_grad():
        outs = pm.text_encoder.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=temperature)
    gen_ids = outs[:, inputs["input_ids"].shape[1]:]
    if opt_policy == "high_vram":
        pm.text_encoder.to("cpu", non_blocking=True)

    release_memory_resources()

    return tokenizer.batch_decode(
        gen_ids, skip_special_tokens=True,
        clean_up_tokenization_spaces=True)[0]

def build_caption_prompt(length: str, word_limit: int) -> str:
    if word_limit and word_limit > 0:
        return f"Write a detailed description for this image in {word_limit} words or less."
    if length != "default":
        return f"Write a {length} detailed description for this image."
    return "Write a detailed description for this image."