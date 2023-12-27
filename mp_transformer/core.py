import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from einops import rearrange
from torch import einsum
import attr

from mp_transformer.misc import init_linear, apply_rotary_pos_emb


@attr.s
class TransformerBlockConfig:
    depth: int = attr.ib()
    width: int = attr.ib()
    mlp_mult: int = attr.ib(default=4)
    fp16: bool = attr.ib(default=True)

    # Attention options
    sink_enabled: bool = attr.ib(default=True)
    causal: bool = attr.ib(default=False)


def mag_norm(x: torch.Tensor, input: torch.Tensor):
    with torch.no_grad():
        desired_mags = torch.linalg.norm(input, dim=-1).mean()
        current_mags = torch.linalg.norm(x, dim=-1).mean()
    return x * (desired_mags / current_mags)


def pnormalize(x: torch.Tensor, partitions: int, eps: float = .0001):
    partitioned = x.reshape(partitions, -1, *x.shape[1:])
    dim = list(range(2, partitioned.ndim))
    n = torch.linalg.vector_norm(partitioned, dim=dim, keepdim=True)
    alpha = np.sqrt(n.numel() / partitioned.numel())
    norm_p = partitioned / torch.add(eps, n, alpha=alpha)
    return norm_p.reshape(x.shape)


class MPLinear(nn.Linear):
    """
    Dense, learnable matmul which approximates a unitary magnitude over an ema horizon.
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 config: TransformerBlockConfig,
                 n_partitions: int = 1,
                 *args, **kwargs):
        super().__init__(in_features, out_features, bias=False, *args, **kwargs)
        self.c = config
        self.n_partitions = n_partitions
        nn.init.normal_(self.weight, std=1.0)
        if hasattr(self, 'bias') and not self.bias is None:
            nn.init.zeros_(self.bias)

    def pre_step(self):
        with torch.no_grad():
            self.weight.copy_(pnormalize(self.weight, self.n_partitions))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fan_in = self.weight[0].numel()
        w_norm = pnormalize(self.weight, self.n_partitions) / fan_in ** 0.5
        return F.linear(x, w_norm, self.bias)


def vp_sum(a: torch.Tensor, b: torch.Tensor, alpha: float = 0.5):
    sum = (1 - alpha) * a + alpha * b
    return sum / np.sqrt(alpha ** 2 + (1 - alpha) ** 2)


class RotaryEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        Dim = 32
        inv_freq = 1. / (10000 ** (torch.arange(0, Dim, 2).float() / Dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, max_seq_len, device):
        t = torch.arange(max_seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i, j -> i j', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        # TODO: should we be normalizing the output of this?
        return rearrange(emb, 'n d -> () () n d')


class Attention(nn.Module):
    def __init__(self, config: TransformerBlockConfig, attn_mask):
        super().__init__()
        self.c = config
        ch = self.c.width
        self.p_qkv = MPLinear(ch, ch * 3, config=config, n_partitions=3)
        self.p_out = MPLinear(ch, ch, config=config)
        if self.c.sink_enabled:
            self.sink = nn.Parameter(torch.randn(1, 1, ch))
        self.attn_mask = attn_mask
        self.fp16 = self.c.fp16

    def get_attention_logits(self, q, k, re):
        B, H, S, _ = q.shape
        C = H * 64

        if self.c.sink_enabled:
            q_s, k_s = q[:, :, -1:], k[:, :, -1:]
            q, k = q[:, :, :-1], k[:, :, :-1]

        # Rotary embedding.
        l = re.shape[-1]
        (ql, qr), (kl, kr) = map(lambda t: (t[..., :l], t[..., l:]), (q, k))
        ql, kl = map(lambda t: apply_rotary_pos_emb(t, re), (ql, kl))
        q, k = map(lambda t: torch.cat(t, dim=-1), ((ql, qr), (kl, kr)))

        if self.c.sink_enabled:
            q, k = map(lambda t, t_s: torch.cat([t, t_s], dim=-2), (q, k), (q_s, k_s))

        # Cosine attention.
        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)
        M = einsum('b h i d, b h j d -> b h i j', q, k)
        if self.attn_mask is not None:
            mask = self.attn_mask[None, None, :S, :S].bool().logical_not().repeat(B, C // 64, 1, 1)
            M[mask] = float('-inf')

        return M

    def softmax_attention_mp_scale(self, n_seq):
        """
        Note: this is currently wrong somehow: for the first layer with random normal inputs its fine but it doesn't
        preserve magnitudes after that layer.
        TODO: derive this analytically if we can (might be difficult to compensate for different masks, but maybe we don't have to?
        """
        if not hasattr(self, 'attn_magnitude_scale_by_seq_len'):
            self.attn_magnitude_scale_by_seq_len = {}
        if n_seq not in self.attn_magnitude_scale_by_seq_len:
            re = RotaryEmbedding()(n_seq, device="cpu")
            n_heads = self.c.width // 64
            v_unsplit = torch.randn(16, n_seq, self.c.width)
            v = rearrange(v_unsplit, 'b n (h d) -> b h n d', h=n_heads)
            q, k = torch.randn(16, n_heads, n_seq, 64), torch.randn(16, n_heads, n_seq, 64)
            M = self.get_attention_logits(q, k, re)
            M = M.softmax(dim=-1)
            h = einsum('b h i j, b h j d -> b h i d', M, v)
            h = rearrange(h, 'b h n d -> b n (h d)')
            self.attn_magnitude_scale_by_seq_len[n_seq] = torch.linalg.norm(v_unsplit, dim=-1).mean() / torch.linalg.norm(h, dim=-1).mean()
        return self.attn_magnitude_scale_by_seq_len[n_seq]

    def forward(self, x, re):
        B, S, C = x.shape
        with torch.autocast(x.device.type, enabled=self.c.fp16):
            if self.c.sink_enabled:
                x = torch.cat([x, self.sink.repeat(B, 1, 1)], dim=1)

            q, k, v_unsplit = self.p_qkv(x).chunk(3, dim=-1)
            scale = 64 ** -.25  # fixed head_dim=64
            q, k = q * scale, k * scale

            # Split heads.
            q = rearrange(q, 'b n (h d) -> b h n d', h=C // 64)
            k = rearrange(k, 'b n (h d) -> b h n d', h=C // 64)
            v = rearrange(v_unsplit, 'b n (h d) -> b h n d', h=C // 64)

            M = self.get_attention_logits(q, k, re)
            M = M.softmax(dim=-1)
            h = einsum('b h i j, b h j d -> b h i d', M, v)

            if self.c.sink_enabled:
                h = h[:, :, :-1]

            # Merge heads and project.
            h = rearrange(h, 'b h n d -> b n (h d)')
            h = mag_norm(h, input=x)
            h = self.p_out(h)
        return h.float()


class MLP(nn.Module):
    def __init__(self, config: TransformerBlockConfig):
        super().__init__()
        self.c = config
        ch = self.c.width
        self.p_in = MPLinear(ch, ch * self.c.mlp_mult * 2, config=config, n_partitions=2)
        self.p_out = MPLinear(ch * 4, ch, config=config)

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=self.c.fp16):
            h = self.p_in(x)
            res, act = h.chunk(2, dim = -1)
            vp_act = F.silu(act) / .596
            h = vp_sum(vp_act, res)
            h = self.p_out(h)
        return h.float()


class Block(nn.Module):
    def __init__(self, config: TransformerBlockConfig, attn_mask: torch.Tensor):
        super().__init__()
        self.attn = Attention(config, attn_mask)
        self.mlp = MLP(config)

    def forward(self, x, re):
        h = vp_sum(x, self.attn(x, re))
        h = vp_sum(h, self.mlp(h))
        return h


class Transformer(nn.Module):
    def __init__(self, ch_in: int = 64, ch_out: int = 64, **config_kwargs):
        super().__init__()
        self.config = TransformerBlockConfig(**config_kwargs)
        self.proj_in = MPLinear(ch_in, self.config.width, self.config)  # does VP make sense here?
        self.rotary_embedding_enc = RotaryEmbedding()
        self.blocks = nn.ModuleList([Block(self.config, None) for _ in range(self.config.depth)])
        self.proj_out = nn.Linear(self.config.width, ch_out)
        self.final_gain = nn.Parameter(torch.ones(1,))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        h = self.proj_in(x)
        re = self.rotary_embedding_enc(x.shape[1], x.device)
        for block in self.blocks:
            h = block(h, re)
        h = self.proj_out(h) * self.final_gain
        return h


def run():
    model = Transformer(depth=8, width=512)
    tst_in = torch.randn(1, 64, 128)
    tst_out = model(tst_in)
    print(tst_out.mean(), tst_out.std())


if __name__ == '__main__':
    import fire
    fire.Fire()
