from cosmos_tokenizer.image_lib import ImageTokenizer


ENCODER_PATH = r"D:\pyproject\awesome_tokenizer\cosmos_tokenizer\checkpoints\DI16X16_encoder.jit"


def load_encoder():
    encoder = ImageTokenizer(checkpoint_enc=ENCODER_PATH)
    return encoder

