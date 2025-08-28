import torch
import torch.nn.functional as F
import torch.nn as nn
from vae.modules.quantize import VectorQuantizer2 as VectorQuantizer
from vae.modules.diffusion_models import Encoder, Decoder


class VQModel(nn.Module):

    def __init__(self,
                 ddconfig,
                 n_embded,
                 embed_dim,
                 ckpt_path=None,
                 remap=None,
                 sane_index_shape=False,
                 ignore_keys=[]):

        super().__init__()

        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        self.quantize = VectorQuantizer(n_embded, embed_dim, beta=0.25,
                                        remap=remap, sane_index_shape=sane_index_shape)

        self.quant_conv = torch.nn.Conv2d(ddconfig['z_channels'], embed_dim, 1)  # 1x1 conv
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig['z_channels'], 1)

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

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
        quant_b = self.quantize.get_codebook_entry(code_b.view(-1),
                                                   (code_b.size(0), code_b.size(1), code_b.size(2), 256))
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")


if __name__ == "__main__":
    from config import vqgan_config
    model = VQModel(ddconfig=vqgan_config.ddconfig,
                    n_embded=vqgan_config.n_embed,
                    embed_dim=vqgan_config.embed_dim,
                    ckpt_path=r"/\vae\maskgit\checkpoints\vqgan.ckpt")


    x = torch.randn(2, 3, 256, 256)
    quant, diff, _ = model.encode(x)
    print(quant.shape, diff)
    dec = model.decode(quant)
    print(dec.shape)
