import math
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


def _expand_token(token, batch_size: int):
    return token.unsqueeze(0).expand(batch_size, -1, -1)


def _bipartite_soft_matching(metric: torch.Tensor, r: int):
    """
    metric: [B, Tspan, C]（仅对像素段传入）
    返回 (merge_fn, unmerge_fn)；通常只用 merge_fn
    """
    Tspan = metric.shape[1]
    r = min(int(r), Tspan // 2)
    if r <= 0:
        return (lambda x: x), (lambda x: x)

    with torch.no_grad():
        m = metric / (metric.norm(dim=-1, keepdim=True) + 1e-12)
        a, b = m[..., ::2, :], m[..., 1::2, :]             # 偶数位=A，奇数位=B
        scores = a @ b.transpose(-1, -2)                   # [B, T/2, T/2]
        node_max, node_idx = scores.max(dim=-1)            # A 的最佳 B
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        unm_idx = edge_idx[..., r:, :]                     # 不合并的 A
        src_idx = edge_idx[..., :r, :]                     # 要合并的 A
        dst_idx = node_idx[..., None].gather(-2, index=src_idx)

    def _merge(x: torch.Tensor, reduce="mean") -> torch.Tensor:
        # x: [B, Tspan, C]
        src, dst = x[..., ::2, :], x[..., 1::2, :]         # [B, T/2, C]
        B, t1, C = src.shape
        unm = src.gather(-2, unm_idx.expand(B, t1 - r, C))
        src_take = src.gather(-2, src_idx.expand(B, r, C))
        if hasattr(dst, "scatter_reduce"):                 # torch>=1.12
            dst = dst.scatter_reduce(-2, dst_idx.expand(B, r, C), src_take, reduce=reduce)
        else:                                              # 旧版 fallback：mean = sum / count
            idx = dst_idx.expand(B, r, C)
            lin = (torch.arange(B, device=x.device)[:, None, None] * t1 + idx.squeeze(-1)).reshape(-1)
            acc = dst.reshape(B * t1, C).clone()
            acc.index_add_(0, lin, src_take.reshape(-1, C))
            cnt = torch.zeros(B * t1, 1, device=x.device, dtype=x.dtype)
            cnt.index_add_(0, lin, torch.ones_like(src_take[..., :1]).reshape(-1, 1))
            dst = (acc / cnt.clamp_min_(1)).reshape(B, t1, C)
        return torch.cat([unm, dst], dim=1)                # [B, Tspan - r, C]

    def _unmerge(x: torch.Tensor) -> torch.Tensor:
        B, _, C = x.shape
        unm_len = unm_idx.shape[1]
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        src_rec = dst.gather(-2, dst_idx.expand(B, r, C))
        out = x.new_zeros(B, Tspan, C)
        out[..., 1::2, :] = dst
        out.scatter_(-2, (2 * unm_idx).expand(B, unm_len, C), unm)
        out.scatter_(-2, (2 * src_idx).expand(B, r, C), src_rec)
        return out

    return _merge, _unmerge


def _make_span_merger(metric_all: torch.Tensor, span: tuple[int, int], r: int):
    """
    只在 span=(s,e) 区间做 ToMe；metric_all: [B, S, C]
    返回 merge_partial(x_BSC) 与实际合并数 r_eff
    """
    s, e = span
    merge_loc, _ = _bipartite_soft_matching(metric_all[:, s:e, :], r=r)
    r_eff = min(int(r), max(0, (e - s) // 2))

    def merge_partial(x_BSC: torch.Tensor) -> torch.Tensor:
        return torch.cat([x_BSC[:, :s, :], merge_loc(x_BSC[:, s:e, :]), x_BSC[:, e:, :]], dim=1)

    return merge_partial, r_eff


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        act_layer = nn.GELU,
        norm_layer = nn.LayerNorm,
        latent_len: int = 64,        # 仅配置 latent 段长度（固定在序列尾部）
    ):
        super().__init__()
        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=False)  # 保持原实现
        self.mlp_ratio = mlp_ratio
        self.latent_len = int(latent_len)  # 只是整数配置，不是参数

        if mlp_ratio > 0:
            self.ln_2 = norm_layer(d_model)
            mlp_width = int(d_model * mlp_ratio)
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, mlp_width)),
                ("gelu", act_layer()),
                ("c_proj", nn.Linear(mlp_width, d_model)),
            ]))

    def attention(self, x_SBC: torch.Tensor):
        # x_SBC: [S, B, C]
        return self.attn(x_SBC, x_SBC, x_SBC, need_weights=False)[0]

    @torch.no_grad()
    def _k_proj_from_attn(self, h_SBC: torch.Tensor) -> torch.Tensor:
        """
        复用 MHA 的 in_proj 权重抽取 K（不新增参数）
        入: h_SBC [S, B, C]；出: [B, S, C]（batch-first 便于 ToMe）
        """
        W = self.attn.in_proj_weight  # [3C, C]
        b = self.attn.in_proj_bias    # [3C] or None
        C = h_SBC.shape[-1]
        Wk = W[C:2*C, :]
        bk = b[C:2*C] if b is not None else None
        KSBC = F.linear(h_SBC, Wk, bk)              # [S, B, C]
        return KSBC.transpose(0, 1).contiguous()    # [B, S, C]

    def forward(
        self,
        x_SBC: torch.Tensor,   # [S, B, d]，排列为 [pixel tokens, latent tokens]
        r: int = 0             # 本层要合并的 pixel tokens 数（自动截到 <= pixel_len//2）
    ):
        S, B, C = x_SBC.shape
        assert 0 <= self.latent_len <= S, "invalid latent_len"
        pixel_len = S - self.latent_len             # 只合并前 pixel_len 段

        # --- Self-Attention（保持原权重与行为） ---
        h_SBC = self.ln_1(x_SBC)
        attn_out = self.attention(h_SBC)
        x_SBC = x_SBC + attn_out

        # --- ToMe：只对像素段 [0:pixel_len) 合并；latent 段 [pixel_len:S) 不动 ---
        if r > 0 and pixel_len > 1:
            # 1) 用现有权重计算 K 作为相似度度量
            K_BSC = self._k_proj_from_attn(h_SBC)                    # [B, S, C]
            # 2) 仅在像素段做匹配/合并
            merge_pix, r_eff = _make_span_merger(K_BSC, span=(0, pixel_len), r=r)
            # 3) 对 x 应用相同合并（先转 [B,S,C]，合并后再转回 [S,B,C]）
            x_BSC = x_SBC.transpose(0, 1).contiguous()               # [B, S, C]
            x_BSC = merge_pix(x_BSC)                                 # [B, S - r_eff, C]
            x_SBC = x_BSC.transpose(0, 1).contiguous()               # [S - r_eff, B, C]
        # else: 不合并，保持长度不变

        # --- MLP（在缩短后的序列上运行） ---
        if self.mlp_ratio > 0:
            x_SBC = x_SBC + self.mlp(self.ln_2(x_SBC))

        return x_SBC  # [S - r_eff, B, d]


class TiTokEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config.dataset.preprocessing.crop_size
        self.patch_size = config.model.vq_model.vit_enc_patch_size
        self.grid_size = self.image_size // self.patch_size
        self.model_size = config.model.vq_model.vit_enc_model_size
        self.num_latent_tokens = config.model.vq_model.num_latent_tokens
        self.token_size = config.model.vq_model.token_size
        self.r = config.model.vq_model.tome_r

        if config.model.vq_model.get("quantize_mode", "vq") == "vae":
            self.token_size = self.token_size * 2

        self.is_legacy = config.model.vq_model.get("is_legacy", True)

        self.width = {"small": 512, "base": 768, "large": 1024}[self.model_size]
        self.num_layers = {"small": 8, "base": 12, "large": 24}[self.model_size]
        self.num_heads = {"small": 8, "base": 12, "large": 16}[self.model_size]

        self.patch_embed = nn.Conv2d(3, self.width, kernel_size=self.patch_size, stride=self.patch_size, bias=True)

        scale = self.width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(1, self.width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(self.grid_size ** 2 + 1, self.width))
        self.latent_token_positional_embedding = nn.Parameter(scale * torch.randn(self.num_latent_tokens, self.width))

        self.ln_pre = nn.LayerNorm(self.width)
        self.transformer = nn.ModuleList()
        for _ in range(self.num_layers):
            self.transformer.append(
                ResidualAttentionBlock(
                    self.width, self.num_heads, mlp_ratio=4.0,
                    latent_len=self.num_latent_tokens,  # 告诉 block 尾部 latent 的长度
                )
            )
        self.ln_post = nn.LayerNorm(self.width)
        self.conv_out = nn.Conv2d(self.width, self.token_size, kernel_size=1, bias=True)

    def forward(self, pixel_values, latent_tokens):
        B = pixel_values.shape[0]

        # patchify
        x = self.patch_embed(pixel_values)                 # [B, C, H/ps, W/ps]
        x = x.reshape(B, self.width, -1)                   # [B, C, grid^2]
        x = x.permute(0, 2, 1)                             # [B, grid^2, width]

        # prepend CLS, add pos
        x = torch.cat([_expand_token(self.class_embedding, B).to(x.dtype), x], dim=1)  # [B, 1+grid^2, width]
        x = x + self.positional_embedding.to(x.dtype)

        # append latents + pos
        latent_tokens = _expand_token(latent_tokens, B).to(x.dtype)
        latent_tokens = latent_tokens + self.latent_token_positional_embedding.to(x.dtype)
        x = torch.cat([x, latent_tokens], dim=1)           # [B, 1+grid^2+L, width]

        # pre-norm and swap to [S, B, d]
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)                             # [S, B, d]


        for blk in self.transformer:
            x = blk(x, r=self.r)
            print(x.shape)

        x = x.permute(1, 0, 2)                             # [B, S', d]，注意 S' ≤ 1+grid^2+L


        L = self.num_latent_tokens
        latent_tokens = x[:, -L:, :]                       # [B, L, d] —— 始终是尾部 L 个
        latent_tokens = self.ln_post(latent_tokens)

        # fake 2D & head
        if self.is_legacy:
            latent_tokens = latent_tokens.reshape(B, self.width, L, 1)
        else:
            latent_tokens = latent_tokens.reshape(B, L, self.width, 1).permute(0, 2, 1, 3)

        latent_tokens = self.conv_out(latent_tokens)       # [B, token_size, 1, L]
        latent_tokens = latent_tokens.reshape(B, self.token_size, 1, L)
        return latent_tokens

