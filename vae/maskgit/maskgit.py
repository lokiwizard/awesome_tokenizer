import torch
import torch.nn as nn
import os
import random
import math
from tqdm import tqdm
from models.vqgan import VQModel
from models.transformers import MaskTransformer
from models.config import vqgan_config
import numpy as np
from torchvision import transforms
import torchvision.utils as vutils
from PIL import Image


class MaskGit(nn.Module):

    def __init__(self, vqconfig, vq_ckpt, transformer_ckpt):
        # we use the default setting of the transformer, so we don't need to pass any config
        super(MaskGit, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.ae = VQModel(ddconfig=vqgan_config.ddconfig,
                          n_embded=vqgan_config.n_embed,
                          embed_dim=vqgan_config.embed_dim,
                          ckpt_path=vq_ckpt)

        self.vit = MaskTransformer(
            img_size=256,
            hidden_dim=768,
            codebook_size=1024,
            depth=24,
            heads=16,
            mlp_dim=3072,
            dropout=0.1
        )
        self.vit.load_state_dict(torch.load(transformer_ckpt, map_location="cpu"), strict=False)
        self.vit.to(self.device)
        self.ae.to(self.device)
        self.mask_value = 1024
        self.patch_size = 16  # 256 / 16 = 16

    def adap_sche(self, step, mode="arccos", leave=False):
        """ Create a sampling scheduler
           :param
            step  -> int:  number of prediction during inference
            mode  -> str:  the rate of value to unmask
            leave -> bool: tqdm arg on either to keep the bar or not
           :return
            scheduler -> torch.LongTensor(): the list of token to predict at each step
        """
        r = torch.linspace(1, 0, step)
        if mode == "root":  # root scheduler
            val_to_mask = 1 - (r ** .5)
        elif mode == "linear":  # linear scheduler
            val_to_mask = 1 - r
        elif mode == "square":  # square scheduler
            val_to_mask = 1 - (r ** 2)
        elif mode == "cosine":  # cosine scheduler
            val_to_mask = torch.cos(r * math.pi * 0.5)
        elif mode == "arccos":  # arc cosine scheduler
            val_to_mask = torch.arccos(r) / (math.pi * 0.5)
        else:
            return

        # fill the scheduler by the ratio of tokens to predict at each step
        sche = (val_to_mask / val_to_mask.sum()) * (self.patch_size * self.patch_size)
        sche = sche.round()
        sche[sche == 0] = 1  # add 1 to predict a least 1 token / step
        sche[-1] += (self.patch_size * self.patch_size) - sche.sum()  # need to sum up nb of code
        return tqdm(sche.int(), leave=leave)

    def sample(self, init_code=None, nb_sample=50, labels=None, sm_temp=1, w=3,
               randomize="linear", r_temp=4.5, sched_mode="arccos", step=12):
        """ Generate sample with the MaskGIT model
           :param
            init_code   -> torch.LongTensor: nb_sample x 16 x 16, the starting initialization code
            nb_sample   -> int:              the number of image to generated
            labels      -> torch.LongTensor: the list of classes to generate
            sm_temp     -> float:            the temperature before softmax
            w           -> float:            scale for the classifier free guidance
            randomize   -> str:              linear|warm_up|random|no, either or not to add randomness
            r_temp      -> float:            temperature for the randomness
            sched_mode  -> str:              root|linear|square|cosine|arccos, the shape of the scheduler
            step:       -> int:              number of step for the decoding
           :return
            x          -> torch.FloatTensor: nb_sample x 3 x 256 x 256, the generated images
            code       -> torch.LongTensor:  nb_sample x step x 16 x 16, the code corresponding to the generated images
        """
        self.vit.eval()
        l_codes = []  # Save the intermediate codes predicted
        l_mask = []  # Save the intermediate masks
        with torch.no_grad():
            if labels is None:  # Default classes generated
                # goldfish, chicken, tiger cat, hourglass, ship, dog, race car, airliner, teddy bear, random
                labels = [1, 7, 282, 604, 724, 179, 751, 404, 850, random.randint(0, 999)] * (nb_sample // 10)
                labels = torch.LongTensor(labels).to(self.device)

            drop = torch.ones(nb_sample, dtype=torch.bool).to(self.device)
            if init_code is not None:  # Start with a pre-define code
                code = init_code
                mask = (init_code == 1024).float().view(nb_sample, self.patch_size * self.patch_size)
            else:  # Initialize a code
                if self.mask_value < 0:  # Code initialize with random tokens
                    code = torch.randint(0, 1024, (nb_sample, self.patch_size, self.patch_size)).to(self.device)
                else:  # Code initialize with masked tokens
                    code = torch.full((nb_sample, self.patch_size, self.patch_size), self.mask_value).to(
                        self.device)
                mask = torch.ones(nb_sample, self.patch_size * self.patch_size).to(self.device)

            # Instantiate scheduler
            if isinstance(sched_mode, str):  # Standard ones
                scheduler = self.adap_sche(step, mode=sched_mode)
            else:  # Custom one
                scheduler = sched_mode

            # Beginning of sampling, t = number of token to predict a step "indice"
            for indice, t in enumerate(scheduler):
                if mask.sum() < t:  # Cannot predict more token than 16*16 or 32*32
                    t = int(mask.sum().item())

                if mask.sum() == 0:  # Break if code is fully predicted
                    break

                with torch.cuda.amp.autocast():  # half precision
                    if w != 0:
                        # Model Prediction
                        logit = self.vit(torch.cat([code.clone(), code.clone()], dim=0),
                                         torch.cat([labels, labels], dim=0),
                                         torch.cat([~drop, drop], dim=0))
                        logit_c, logit_u = torch.chunk(logit, 2, dim=0)
                        _w = w * (indice / (len(scheduler) - 1))
                        # Classifier Free Guidance
                        logit = (1 + _w) * logit_c - _w * logit_u
                    else:
                        logit = self.vit(code.clone(), labels, drop_label=~drop)

                prob = torch.softmax(logit * sm_temp, -1)
                # Sample the code from the softmax prediction
                distri = torch.distributions.Categorical(probs=prob)
                pred_code = distri.sample()

                conf = torch.gather(prob, 2, pred_code.view(nb_sample, self.patch_size * self.patch_size, 1))

                if randomize == "linear":  # add gumbel noise decreasing over the sampling process
                    ratio = (indice / len(scheduler))
                    rand = r_temp * np.random.gumbel(size=(nb_sample, self.patch_size * self.patch_size)) * (1 - ratio)
                    conf = torch.log(conf.squeeze()) + torch.from_numpy(rand).to(self.device)
                elif randomize == "warm_up":  # chose random sample for the 2 first steps
                    conf = torch.rand_like(conf) if indice < 2 else conf
                elif randomize == "random":  # chose random prediction at each step
                    conf = torch.rand_like(conf)

                # do not predict on already predicted tokens
                conf[~mask.bool()] = -math.inf

                # chose the predicted token with the highest confidence
                tresh_conf, indice_mask = torch.topk(conf.view(nb_sample, -1), k=t, dim=-1)
                tresh_conf = tresh_conf[:, -1]

                # replace the chosen tokens
                conf = (conf >= tresh_conf.unsqueeze(-1)).view(nb_sample, self.patch_size, self.patch_size)
                f_mask = (mask.view(nb_sample, self.patch_size, self.patch_size).float() * conf.view(nb_sample,
                                                                                                     self.patch_size,
                                                                                                     self.patch_size).float()).bool()
                code[f_mask] = pred_code.view(nb_sample, self.patch_size, self.patch_size)[f_mask]

                # update the mask
                for i_mask, ind_mask in enumerate(indice_mask):
                    mask[i_mask, ind_mask] = 0
                l_codes.append(pred_code.view(nb_sample, self.patch_size, self.patch_size).clone())
                l_mask.append(mask.view(nb_sample, self.patch_size, self.patch_size).clone())

            # decode the final prediction
            _code = torch.clamp(code, 0, 1023)  # VQGAN has only 1024 codebook
            x = self.ae.decode_code(_code)
            x = (torch.clamp(x, -1, 1) + 1) / 2
        return x, l_codes, l_mask


if __name__ == "__main__":
    vq_ckpt = r"D:\pyproject\awesome_diffusion\vae\maskgit\checkpoints\vqgan.ckpt"
    transformer_ckpt = r"D:\pyproject\awesome_diffusion\vae\maskgit\checkpoints\MaskGIT_ImageNet_256.pth"
    model = MaskGit(vqgan_config, vq_ckpt, transformer_ckpt)
    model.to(model.device)

    cat_img = r"D:\pyproject\awesome_diffusion\vae\cat.jpg"
    cat_img = Image.open(cat_img).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    cat_img = transform(cat_img).unsqueeze(0).to(model.device)
    with torch.no_grad():
        _, _, info = model.ae.encode(cat_img)
    init_code = info[2].view(1, 16, 16)

    # random mask some init_code
    mask_ratio = 0.6
    mask = torch.rand(init_code.shape, device=init_code.device) < mask_ratio
    init_code[mask] = 1024  # mask token

    nb_imgs = 1
    labels = torch.LongTensor([1000] * nb_imgs).to(model.device)
    gen_sample = model.sample(nb_sample=nb_imgs, labels=labels, sm_temp=1, w=3,
                              randomize="linear", r_temp=4.5, sched_mode="arccos",
                              step=100, init_code=init_code)[0]
    print(gen_sample.shape)

    output_image = transforms.ToPILImage()(vutils.make_grid(gen_sample, nrow=2, padding=0, normalize=True))
    output_image.save(f"res.png")
