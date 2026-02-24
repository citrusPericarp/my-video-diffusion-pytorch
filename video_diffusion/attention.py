### Attention 模块的实现

import math
import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange
from einops_exts import rearrange_many

from utils import exists

# 相对位置偏置
# 用于时间注意力机制，减少长距离注意力
class RelativePositionBias(nn.Module):
    def __init__(
        self,
        heads = 8,
        num_buckets = 32,
        max_distance = 128
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod  # 静态方法，不需要实例化即可调用
    def _relative_position_bucket(relative_position, num_buckets = 32, max_distance = 128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    # n: 时间步数
    def forward(self, n, device):
        q_pos = torch.arange(n, dtype = torch.long, device = device)
        k_pos = torch.arange(n, dtype = torch.long, device = device)
        rel_pos = rearrange(k_pos, 'j -> 1 j') - rearrange(q_pos, 'i -> i 1')
        rp_bucket = self._relative_position_bucket(rel_pos, num_buckets = self.num_buckets, max_distance = self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, 'i j h -> h i j')

class SpatialLinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        head_dim = dim_head * heads
        self.to_kqv = nn.Conv2d(dim, head_dim * 3, 1, bias=False)  # 这里用的卷积
        self.to_out = nn.Conv2d(head_dim, dim, 1)
        
    def forward(self, x):
        b, c, f, h, w = x.shape
        x = rearrange(x, 'b c f h w -> (b f) c h w')  # 因为在空间维度上做注意力机制

        qkv = self.to_kqv(x).chunk(3, dim=1)
        q, k, v = rearrange_many(qkv, 'b (h d) x y -> b h d (x y)', h=self.heads)  # 拆分成 heads 个头

        # Linear Attention 的实现
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h d (x y) -> b (h d) x y', h = self.heads, x = h, y = w)
        out = self.to_out(out)
        return rearrange(out, '(b f) c h w -> b c f h w', b = b)

# 用于时空注意力机制的包装
# 传参 from_einops, to_einops 表示输入和输出的 einops 格式，fn 表示要包装的模块
class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x.shape
        reconstitue_kwargs = dict(tuple(zip(self.from_einops.split(' '), shape)))
        x = rearrange(x, f'{self.from_einops} -> {self.to_einops}')
        x = self.fn(x, **kwargs)
        x = rearrange(x, f'{self.to_einops} -> {self.from_einops}', **reconstitue_kwargs)  # 显示提供重构参数
        return x

# 标准多头注意力机制
class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, rotary_emb=None):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        head_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, head_dim * 3, bias=False)
        self.to_out = nn.Linear(head_dim, dim)
        self.rotary_emb = rotary_emb

    def forward(self, x, pos_bias=None, focus_present_mask=None):
        n, device = x.shape[-2], x.device

        qkv = self.to_qkv(x).chunk(3, dim = -1)

        if exists(focus_present_mask) and focus_present_mask.all():
            # 所有样本都关注当前帧
            # 等价于将该token的值通过到输出
            values = qkv[-1]
            return self.to_out(values)

        # 拆分成 heads 个头
        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h = self.heads)

        # scale
        q = q * self.scale

        # 将位置编码旋转到 queries 和 keys 中，用于时间注意力
        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # similarity
        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)

        # 相对位置偏置
        if exists(pos_bias):
            sim = sim + pos_bias

        if exists(focus_present_mask) and not (~focus_present_mask).all():
            attend_all_mask = torch.ones((n, n), device = device, dtype = torch.bool)
            attend_self_mask = torch.eye(n, device = device, dtype = torch.bool)  # 对角线为1，其余为0

            mask = torch.where(
                rearrange(focus_present_mask, 'b -> b 1 1 1 1'),
                rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
                rearrange(attend_all_mask, 'i j -> 1 1 1 i j'),
            )

            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)  # float info

        sim = sim - sim.amax(dim = -1, keepdim = True).detach() # detach() 表示不计算梯度
        attn = sim.softmax(dim = -1)

        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        return self.to_out(out)