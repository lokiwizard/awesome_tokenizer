from cosmos_tokenizer.image_lib import ImageTokenizer


ENCODER_PATH = r"/\cosmos_tokenizer\checkpoints\DI8X8_encoder.jit"


def load_encoder():
    encoder = ImageTokenizer(checkpoint_enc=ENCODER_PATH)
    return encoder

