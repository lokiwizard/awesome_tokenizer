import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict

from IBQ.modules.blocks import Encoder, Decoder
from IBQ.modules.quantize import IndexPropagationQuantize
from IBQ.modules.quantize import VectorQuantizer2 as VectorQuantizer


class VQModel(nn.Module):

    def __init__(self,
                 ddconfig,
                 n_embed,
                 embed_dim,
                 beta=0.25,
                 use_entropy_loss=False,
                 remap=None,
                 cosine_similarity=False,
                 entropy_temperature=0.01,
                 sample_minimization_weight=1.0,
                 batch_maximization_weight=1.0,):

        super().__init__()

        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize = IndexPropagationQuantize(n_embed, embed_dim, beta, use_entropy_loss,
                                                 remap=remap, cosine_similarity=cosine_similarity,
                                                 entropy_temperature=entropy_temperature,
                                                 sample_minimization_weight=sample_minimization_weight,
                                                 batch_maximization_weight=batch_maximization_weight)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)


    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

if __name__ == "__main__":
    from omegaconf import OmegaConf
    config_path = r"D:\pyproject\awesome_tokenizer\IBQ\config\imagenet_ibqgan_16384.yaml"
    ckpt_path = r"D:\pyproject\awesome_tokenizer\IBQ\checkpoint\imagenet256_16384.ckpt"
    state = torch.load(ckpt_path, map_location="cpu")["state_dict"]


    config = OmegaConf.load(config_path)
    ddconfig = config.model.init_args.ddconfig
    n_embed = config.model.init_args.n_embed
    embed_dim = config.model.init_args.embed_dim
    model = VQModel(ddconfig, n_embed, embed_dim)
    model.load_state_dict(state, strict=False)
    model.eval()
    x = torch.randn(1, 3, 256, 256)
    quant, diff, info = model.encode(x)
    print(quant.shape, diff, info[-1].shape)
    dec = model.decode(quant)
    print(dec.shape)











