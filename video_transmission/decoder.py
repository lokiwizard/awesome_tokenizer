from cosmos_tokenizer.image_lib import ImageTokenizer

DECODER_PATH = r"D:\pyproject\awesome_tokenizer\cosmos_tokenizer\checkpoints\DI8X8_decoder.jit"


def load_decoder():
    decoder = ImageTokenizer(checkpoint_dec=DECODER_PATH)
    return decoder