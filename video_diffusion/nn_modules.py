### 一些 small helper modules 类的实现
import math
import torch
import torch.nn as nn

# EMA (Exponential Moving Average)
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

# Residual
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

# 正弦位置编码
# 用于后续时间维度上的位置编码
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]   # [B, L, 1] * [1, D/2] = [B, L, D/2]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # [B, L, D/2] * 2 = [B, L, D]
        return emb

# 3D上采样
def Upsample(dim):
    return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))

# 3D下采样
def Downsample(dim):
    return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))

# Layer Norm
# 通道维度上的归一化
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps  # 用于防止分母为0
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))  # scale
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))  # shift
    
    # x: [B, C, T, H, W]
    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)  # unbiased为False表示使用偏估计
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma + self.beta

# Pre LN 包装
# 在进入注意力机制之前进行归一化
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)

