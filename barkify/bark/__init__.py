from .model import GPT, GPTConfig
from .model_fine import FineGPT, FineGPTConfig

from .generation import SAMPLE_RATE, preload_models
from .generation import generate_fine, codec_decode

def create_infer_model(model_config):
    gptconf = GPTConfig(**model_config)
    return GPT(gptconf).eval()