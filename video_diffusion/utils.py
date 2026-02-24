### 一些 helper functions 的实现
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

# 判断一个值是否存在
def exists(val):
    return val is not None

# do nothing
# 
def noop(*args, **kwargs):
    pass

# 判断 kenerl size 是否为奇数
def is_odd(n):
    return (n % 2) == 1

# 用于设置默认值
# 如果 val 存在，则返回 val
# 否则，如果 d 是可调用的，则调用 d 并返回结果，否则返回 d 本身
def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# 循环遍历数据集
# dl： data loader
def cycle(dl):
    while True:
        for data in dl:
            yield data

# 将一个数字分成若干个组
# 
def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

# 生成一个概率掩码
# 用于后续 attention mask 的生成
def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob

# 判断一个列表是否为字符串列表
def is_list_str(x):
    return isinstance(x, (list, tuple)) and all([type(el) == str for el in x])

# 提取参数
# a: 参数(alphas, betas, etc.) -> [T]
# t: 时间步 -> [B]
# x_shape: [B, C, T, H, W]
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# 将通道数转换为模式
CHANNELS_TO_MODE = {
    1 : 'L',
    3 : 'RGB',
    4 : 'RGBA'
}

# 读取 gif 文件，依次返回每一帧
def seek_all_images(img, channels = 3):
    assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
    mode = CHANNELS_TO_MODE[channels]

    i = 0
    while True:
        try:
            img.seek(i)  # gif 的第 i 帧
            yield img.convert(mode)  # 转换为指定模式
        except EOFError:  # EOFError: End of file reached
            break
        i += 1

# [c, f, h, w] tensor -> gif
def video_tensor_to_gif(tensor, path, duration = 120, loop = 0, optimize = True):
    images = map(T.ToPILImage(), tensor.unbind(dim = 1))  # 沿帧数维度拆分并转换为 PIL 图像
    first_img, *rest_imgs = images
    first_img.save(path, save_all = True, append_images = rest_imgs, duration = duration, loop = loop, optimize = optimize)
    return images

# gif -> [c, f, h, w] tensor
def gif_to_tensor(path, channels = 3, transform = T.ToTensor()):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, channels = channels)))
    return torch.stack(tensors, dim = 1)  # 沿帧数维度堆叠

# 恒等函数
def identity(t, *args, **kwargs):
    return t

# 归一化图像到 [-1, 1]
def normalize_img(t):
    return t * 2 - 1

# 反归一化图像到 [0, 1]
def unnormalize_img(t):
    return (t + 1) * 0.5

# 将帧数调整为指定帧数
# t: [c, f, h, w]
def cast_num_frames(t, *, frames):
    f = t.shape[1]

    if f == frames:
        return t

    if f > frames:
        return t[:, :frames]  # 截取前 frames 帧

    return F.pad(t, (0, 0, 0, 0, 0, frames - f))  # 填充到指定帧数
