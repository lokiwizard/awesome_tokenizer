from diffusers import AutoencoderKL
from cosmos_tokenizer.image_lib import ImageTokenizer

VAE_NAMES = [
    "stabilityai/sdxl-vae"
]


def load_vae(vae_name: str) -> AutoencoderKL:
    if vae_name not in VAE_NAMES:
        raise ValueError(f"VAE '{vae_name}' is not supported. Supported VAEs: {VAE_NAMES}")

    vae = AutoencoderKL.from_pretrained(vae_name, subfolder="vae")

    return vae

