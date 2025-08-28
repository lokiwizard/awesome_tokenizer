import torch
from titok_tokenizer.models.maskgit import ImageBert
from titok_tokenizer.titok import TiTok
from omegaconf import OmegaConf


@torch.no_grad()
def sample_fn(generator,
              tokenizer,
              labels=None,
              guidance_scale=3.0,
              guidance_decay="constant",
              guidance_scale_pow=3.0,
              randomize_temperature=2.0,
              softmax_temperature_annealing=False,
              num_sample_steps=10,
              device="cuda",
              return_tensor=False):
    generator.eval()
    tokenizer.eval()
    if labels is None:
        # goldfish, chicken, tiger, cat, hourglass, ship, dog, race car, airliner, teddy bear, random
        labels = [281, 282, 282, 604, 724, 179, 751, 404, 850, 1000]

    if not isinstance(labels, torch.Tensor):
        labels = torch.LongTensor(labels).to(device)

    generated_tokens = generator.generate(
        condition=labels,
        guidance_scale=guidance_scale,
        guidance_decay=guidance_decay,
        guidance_scale_pow=guidance_scale_pow,
        randomize_temperature=randomize_temperature,
        softmax_temperature_annealing=softmax_temperature_annealing,
        num_sample_steps=num_sample_steps)

    generated_image = tokenizer.decode_tokens(
        generated_tokens.view(generated_tokens.shape[0], -1)
    )
    if return_tensor:
        return generated_image

    generated_image = torch.clamp(generated_image, 0.0, 1.0)
    generated_image = (generated_image * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

    return generated_image


if __name__ == "__main__":
    config_path = r"D:\pyproject\awesome_tokenizer\titok_tokenizer\configs\titok_b64.yaml"
    config = OmegaConf.load(config_path)

    titok_tokenizer = TiTok(config)
    tokenizer_state_dict = torch.load(r"D:\pyproject\awesome_tokenizer\titok_tokenizer\checkpoints\tokenizer_titok_b64.bin",
                            map_location="cpu")
    titok_tokenizer.load_state_dict(tokenizer_state_dict, strict=True)

    generator = ImageBert(config)
    gen_state_dict = torch.load(r"D:\pyproject\awesome_tokenizer\titok_tokenizer\checkpoints\generator_titok_b64.bin",
                            map_location="cpu")
    generator.load_state_dict(gen_state_dict, strict=True)

    titok_tokenizer = titok_tokenizer.to("cuda")
    generator = generator.to("cuda")

    images = sample_fn(generator, titok_tokenizer, device="cuda", return_tensor=False)

    # save images
    from PIL import Image
    import os
    os.makedirs("./outputs", exist_ok=True)
    for i, img in enumerate(images):
        Image.fromarray(img).save(f"./outputs/sample_{i}.png")








