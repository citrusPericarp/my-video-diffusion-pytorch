### 调度器和扩散模型的实现
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops_exts import check_shape
from tqdm import tqdm

from utils import default, exists, is_list_str, normalize_img, unnormalize_img, extract
from text import bert_embed, tokenize

# cosine beta 调度器
def cosine_beta_schedule(timesteps, s = 0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos((x / timesteps + s) / (1 + s) * math.pi * 0.5) ** 2  # 从1到0的余弦衰减
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)  # 裁剪到0.0001到0.9999之间，防止溢出

# 高斯扩散模型
class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,  # *代表后续参数只能通过关键字参数传递
        image_size,
        num_frames,
        text_use_bert_cls = False,
        channels = 3,
        timesteps = 1000,
        loss_type = 'l1',
        use_dynamic_thres = False,  # 用于动态阈值
        dynamic_thres_percentile = 0.9  # 动态阈值的百分位数
    ):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.image_size = image_size
        self.num_frames = num_frames
        self.channels = channels

        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim = 0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[: -1], (1, 0), value = 1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # 注册 buffer 的 helper function，同时将 float64 类型的张量转换为 float32 类型
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # 计算扩散过程 q(x_t | x_{t-1}) 和其他相关量

        # sqrt(alphas_cumprod)
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        # sqrt(1 - alphas_cumprod)
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        # log(1 - alphas_cumprod)
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        # 1 / sqrt(alphas_cumprod)
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        # 1 / sqrt(alphas_cumprod) - 1
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # 计算后验分布 q(x_{t-1} | x_t, x_0)
        # q(x_{t-1} | x_t, x_0) = N(x_{t-1}; mu_q(x_t, x_0), sigma_q(x_t, x_0))
       
        # 方差：sigma_q(x_t, x_0) = beta_t * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)
        # 因为后验方差在扩散开始时为0，故需要裁剪
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        
        # 均值：mu_q(x_t, x_0) = betas * torch.sqrt(alphas_cumprod_prev) / (1 - alphas_cumprod) * x_0 
                               # + (1 - alpha_cumprod_prev) * torch.sqrt(alphas) / (1 - alphas_cumprod) * x_t
        # x_0 一项的系数
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        # x_t 一项的系数
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # 文本条件化参数
        self.text_use_bert_cls = text_use_bert_cls

        # 采样时使用动态阈值
        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

    # 计算扩散过程 q(x_t | x_0) 的均值和方差
    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    # 从噪声预测 x_0
    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    # 计算后验分布 q(x_{t-1} | x_t, x_0) 的均值和方差
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # 计算预测分布 p(x_{t-1} | x_t) 的均值和方差
    # 从而计算损失函数
    def p_mean_variance(self, x, t, clip_denoised: bool, cond = None, cond_scale = 1.):
        x_recon = self.predict_start_from_noise(x, t=t, noise = self.denoise_fn.forward_with_cond_scale(x, t, cond = cond, cond_scale = cond_scale))

        # 裁剪去噪
        if clip_denoised:
            s = 1.
            # 使用动态阈值
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim = -1
                )

                s.clamp_(min = 1.)
                s = s.view(-1, *((1,) * (len(x_recon.shape) - 1)))  # [B, 1, 1, 1]

            # 裁剪到阈值范围内
            x_recon = x_recon.clamp(-s, s) / s

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance  # 注意命名的细节，mean是模型预测出来的，而方差是固定的

    # 推理采样

    # 单步反向过程，利用模型预测的噪声，x_t -> x_{t-1}
    @torch.inference_mode()
    def p_sample(self, x, t, cond = None, cond_scale = 1., clip_denoised = True):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x = x, t = t, clip_denoised = clip_denoised, cond = cond, cond_scale = cond_scale)
        noise = torch.randn_like(x)
        # t = 0 时，不加噪声
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    # 多步反向过程，利用模型预测的噪声，x_T -> x_{T-1} -> ... -> x_0
    @torch.inference_mode()
    def p_sample_loop(self, shape, cond = None, cond_scale = 1.):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), cond = cond, cond_scale = cond_scale)

        return unnormalize_img(img)

    # 采样,调用 P_sample_loop
    @torch.inference_mode()
    def sample(self, cond = None, cond_scale = 1., batch_size = 16):
        device = next(self.denoise_fn.parameters()).device

        if is_list_str(cond):
            cond = bert_embed(tokenize(cond)).to(device)

        batch_size = cond.shape[0] if exists(cond) else batch_size
        image_size = self.image_size
        channels = self.channels
        num_frames = self.num_frames
        return self.p_sample_loop((batch_size, channels, num_frames, image_size, image_size), cond = cond, cond_scale = cond_scale)

    # 插值采样，x_0 -> x_t
    @torch.inference_mode()
    def interpolate(self, x1, x2, t = None, lam = 0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    # 前向过程，不用模型预测，x_{t-1} -> x_t
    def q_sample(self, x_start, t, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    # 计算损失函数
    def p_losses(self, x_start, t, cond = None, noise = None, **kwargs):
        b, c, f, h, w, device = *x_start.shape, x_start.device
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        if is_list_str(cond):
            cond = bert_embed(tokenize(cond), return_cls_repr = self.text_use_bert_cls)
            cond = cond.to(device)

        pred_noise = self.denoise_fn(x_noisy, t, cond = cond, **kwargs)

        if self.loss_type == 'l1':
            loss = F.l1_loss(pred_noise, noise)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(pred_noise, noise)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x, *args, **kwargs):
        b, device, img_size, = x.shape[0], x.device, self.image_size
        check_shape(x, 'b c f h w', c = self.channels, f = self.num_frames, h = img_size, w = img_size)
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        x = normalize_img(x)
        return self.p_losses(x, t, *args, **kwargs)

