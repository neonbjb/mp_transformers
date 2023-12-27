import torch
import torch.nn as nn
from einops import rearrange


def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device

    if num_samples >= num:
        indices = torch.randperm(num_samples, device = device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device = device)

    return samples[indices]


def build_relative_mask(maximum_extent=4096, local_context=64):
    """
    Returns a mask for relative attention where 1=unmasked, 0=masked.
    """
    mask = torch.zeros(maximum_extent, maximum_extent)
    for i in range(maximum_extent):
        mask[i, max(0, i-local_context):i+local_context] = 1
    return mask


def build_encoder_mask(maximum_extent=4096, local_context=64):
    mask = build_relative_mask(maximum_extent, local_context)
    # Encoder gets a causal mask.
    return torch.tril(torch.ones_like(mask)) * mask


def init_linear(lin, ch, depth=1, scale_multiplier=1.0):
    nn.init.normal_(lin.weight, std=scale_multiplier / (ch * depth) ** 0.5)
    if hasattr(lin, 'bias') and not lin.bias is None:
        nn.init.zeros_(lin.bias)


def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs):
    seq_len = t.shape[-2]
    freqs = freqs[:, :, -seq_len:]
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())