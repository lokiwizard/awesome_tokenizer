# Awesome Tokenizer

A collection of **image tokenizers** implemented in PyTorch.  
These tokenizers convert images into discrete tokens for applications such as image generation, compression, and representation learning.

## 📑 Table of Checkpoints

| Model Name      | Description                                                              | Checkpoint |
|-----------------|--------------------------------------------------------------------------|------------|
| **VQ-GAN**      | Classic vector quantization for images                                   | [Download](https://huggingface.co/llvictorll/Maskgit-pytorch/tree/main/pretrained_maskgit/VQGAN) |
| **TiTOK**       | Can compress the image into few 1-D tokens(32, 64, 128)                  | [Download](https://huggingface.co/yucornetto/models) |
| **Cosmos**      | a high performance image tokenizer trained by nvdia                      | [Download](https://huggingface.co/collections/nvidia/cosmos-tokenizer-672b93023add81b66a8ff8e6) |
| **ONE_D_Piece** | a novel image tokenizer supporting quality-controllable image compression | [Download](https://github.com/turingmotors/One-D-Piece) |

You can download their checkpoints and use them directly in your projects.

## 🚀 Quick Start
I provide a demo jupyter notebook to demonstrate how to use these tokenizers.

## References
You can find more details about each model in their respective papers:
- [VQ-GAN](https://arxiv.org/abs/2012.09841)
- [TiTOK](https://arxiv.org/abs/2406.07550)
- [Cosmos](https://arxiv.org/abs/2501.03575)
- [ONE-D-PIECE](https://arxiv.org/abs/2501.10064)