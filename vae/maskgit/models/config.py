from dataclasses import dataclass

@dataclass
class vqgan_config:
    embed_dim = 256
    n_embed = 1024

    ddconfig = {
        "double_z": False,
        "z_channels": 256,
        "in_channels": 3,
        "out_ch": 3,
        "ch": 128,
        "ch_mult": [1, 1, 2, 2, 4],
        "num_res_blocks": 2,
        "attn_resolutions": [16],
        "dropout": 0.0,
        "resolution": 256,
    }



