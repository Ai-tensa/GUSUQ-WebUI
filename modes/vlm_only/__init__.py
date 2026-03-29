import gradio as gr


cfg_param_key = "true_cfg_scale"


def get_pipeline(*args, **kwargs):
    gr.Error("Current mode does not support image generation pipelines.")
    raise RuntimeError("Image generation is disabled for current mode.")

