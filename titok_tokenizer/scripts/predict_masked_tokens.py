import torch
from titok_tokenizer.models.maskgit import ImageBert
from titok_tokenizer.titok import TiTok
from omegaconf import OmegaConf
from PIL import Image
import numpy as np


def load_image(image_path, image_size=(256, 256)):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(image_size)
    image = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0) / 255.0
    return image


def mask_tokens(input_tokens, mask_token_id=4096, mask_ratio=0.15, random_mask=True):
    """
    Randomly mask a portion of the input tokens.
    """
    input_tokens = input_tokens.clone()
    num_tokens = input_tokens.size(2)
    num_mask = int(num_tokens * mask_ratio)
    if random_mask:
        mask_indices = torch.randperm(num_tokens)[:num_mask]
    else:
        start_index = (num_tokens - num_mask) // 2  # center mask
        mask_indices = torch.arange(start_index, start_index + num_mask)
    input_tokens[:, :, mask_indices] = mask_token_id
    return input_tokens, mask_indices


if __name__ == "__main__":
    config_path = r"D:\pyproject\awesome_tokenizer\titok_tokenizer\configs\titok_b64.yaml"
    config = OmegaConf.load(config_path)

    titok_tokenizer = TiTok(config)
    tokenizer_state_dict = torch.load(
        r"D:\pyproject\awesome_tokenizer\titok_tokenizer\checkpoints\tokenizer_titok_b64.bin",
        map_location="cpu")
    titok_tokenizer.load_state_dict(tokenizer_state_dict, strict=True)

    generator = ImageBert(config)
    gen_state_dict = torch.load(r"D:\pyproject\awesome_tokenizer\titok_tokenizer\checkpoints\generator_titok_b64.bin",
                                map_location="cpu")
    generator.load_state_dict(gen_state_dict, strict=True)

    titok_tokenizer = titok_tokenizer.to("cuda")
    generator = generator.to("cuda")

    test_image_path = r"D:\pyproject\awesome_tokenizer\titok_tokenizer\cat.jpg"
    test_image = load_image(test_image_path).to("cuda")

    with torch.no_grad():
        tokens = titok_tokenizer.encode(test_image)[1]["min_encoding_indices"]
        masked_tokens, mask_indices = mask_tokens(tokens, mask_token_id=4096, mask_ratio=0., random_mask=False)
        print("Original Tokens:", tokens)
        print("Masked Tokens:", masked_tokens)

        #logits = generator.predict_masked_tokens(masked_tokens.squeeze(1), condition=torch.tensor([281]))

        # get the predicted tokens at the masked positions
        predicted_tokens = generator.generate(condition=torch.tensor([281]).to("cuda"),
                                              input_ids=masked_tokens.squeeze(1).to("cuda"),
                                              )


        #predicted_masked_tokens = masked_tokens.clone()
        #predicted_masked_tokens[:, :, mask_indices] = predicted_tokens[:, mask_indices]

        rec = titok_tokenizer.decode_tokens(predicted_tokens.view(predicted_tokens.shape[0], -1))
        rec = torch.clamp(rec, 0.0, 1.0)
        rec = (rec * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        rec_image = Image.fromarray(rec[0])
        rec_image.save("./outputs/reconstructed_image.png")


