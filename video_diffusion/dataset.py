from functools import partial
from pathlib import Path
import numpy as np
import torch
from torch.utils import data
import torchvision.transforms as T
import torchvision.datasets as D

from utils import cast_num_frames, gif_to_tensor, identity


# Moving MNIST 数据集（无条件视频）：在画布上 1～2 个数字随机运动、反弹
class MovingMNISTDataset(data.Dataset):
    def __init__(
        self,
        num_frames=16,
        image_size=64,
        num_digits=2,
        root="./data/mnist",
        train=True,
        num_samples=10000,
        seed=42,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.image_size = image_size
        self.num_digits = min(max(1, num_digits), 2)
        self.num_samples = num_samples
        self.rng = np.random.default_rng(seed)
        # 使用 MNIST 数字，28x28
        self.mnist = D.MNIST(root=root, train=train, download=True, transform=T.ToTensor())
        self.digit_size = 28

    def __len__(self):
        return self.num_samples

    def _render_frame(self, canvas, digit_imgs, positions):
        # canvas: (1, H, W), digit_imgs: (n, 28, 28)
        for img, (y, x) in zip(digit_imgs, positions):
            h, w = img.shape[-2], img.shape[-1]
            y, x = int(y), int(x)
            y1, y2 = max(0, y), min(self.image_size, y + h)
            x1, x2 = max(0, x), min(self.image_size, x + w)
            sy1 = max(0, -y)
            sy2 = h - max(0, y + h - self.image_size)
            sx1 = max(0, -x)
            sx2 = w - max(0, x + w - self.image_size)
            if y2 > y1 and x2 > x1:
                patch = img[sy1:sy2, sx1:sx2]
                canvas[0, y1:y2, x1:x2] = torch.maximum(
                    canvas[0, y1:y2, x1:x2],
                    patch,
                )
        return canvas

    def __getitem__(self, index):
        # 固定种子便于复现
        rng = np.random.default_rng(self.rng.integers(0, 2**31))
        n = self.num_digits
        # 随机选 n 个数字
        indices = rng.integers(0, len(self.mnist), size=n)
        digits = torch.stack([self.mnist[i][0].squeeze(0) for i in indices])  # [n, 28, 28]
        # 初始位置与速度（像素/帧）
        positions = rng.uniform(0, self.image_size - self.digit_size, size=(n, 2))  # [n, (y,x)]
        velocities = rng.uniform(-4, 4, size=(n, 2))
        out = torch.zeros(1, self.num_frames, self.image_size, self.image_size)
        for f in range(self.num_frames):
            frame = torch.zeros(1, self.image_size, self.image_size)
            self._render_frame(frame, digits, positions)
            out[0, f] = frame.squeeze(0)
            positions += velocities
            for i in range(n):
                for j in range(2):
                    if positions[i, j] < 0:
                        positions[i, j] = -positions[i, j]
                        velocities[i, j] = -velocities[i, j]
                    elif positions[i, j] > self.image_size - self.digit_size:
                        positions[i, j] = 2 * (self.image_size - self.digit_size) - positions[i, j]
                        velocities[i, j] = -velocities[i, j]
        # 返回 [C, F, H, W]
        return out


# 数据集类（GIF 文件夹）
class Dataset(data.Dataset):
    def __init__(
        self,
        folder,
        image_size,
        channels = 3,
        num_frames = 16,
        horizontal_flip = False,
        force_num_frames = True,
        exts = ['gif']
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.channels = channels
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.cast_num_frames_fn = partial(cast_num_frames, frames = num_frames) if force_num_frames else identity

        self.transform = T.Compose([
            T.Resize(image_size),
            T.RandomHorizontalFlip() if horizontal_flip else T.Lambda(identity),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        tensor = gif_to_tensor(path, self.channels, transform = self.transform)
        return self.cast_num_frames_fn(tensor)
