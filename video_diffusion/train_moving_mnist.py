# 无条件 Moving MNIST 训练入口
# 在项目根目录运行: python -m video_diffusion.train_moving_mnist
from pathlib import Path
import torch
from dataset import MovingMNISTDataset
from diffusion import GaussianDiffusion
from unet3d import UNet3D
from trainer import Trainer

# --------------- 可调参数 ---------------
IMAGE_SIZE = 32
NUM_FRAMES = 8
CHANNELS = 1

NUM_SAMPLES = 10000
TRAIN_BATCH_SIZE = 1
TRAIN_NUM_STEPS = 50000
GRADIENT_ACCUMULATE_EVERY = 2
SAVE_AND_SAMPLE_EVERY = 2000
RESULTS_FOLDER = "./results_moving_mnist"
MNIST_ROOT = "./data/mnist"

DIM = 32
DIM_MULTS = (1, 2, 4)
# -----------------------------------------


def main():
    # 无条件 UNet3D：不传 cond_dim，不启用 BERT 文本条件
    model = UNet3D(
        dim=DIM,
        cond_dim=None,
        use_bert_text_cond=False,
        channels=CHANNELS,
        dim_mults=DIM_MULTS,
        use_sparse_linear_attn=True,
    )

    diffusion = GaussianDiffusion(
        denoise_fn=model,
        image_size=IMAGE_SIZE,
        num_frames=NUM_FRAMES,
        channels=CHANNELS,
        timesteps=1000,
        loss_type="l1",
    ).cuda()

    dataset = MovingMNISTDataset(
        num_frames=NUM_FRAMES,
        image_size=IMAGE_SIZE,
        num_digits=2,
        root=MNIST_ROOT,
        train=True,
        num_samples=NUM_SAMPLES,
        seed=42,
    )

    trainer = Trainer(
        diffusion_model=diffusion,
        dataset=dataset,
        train_batch_size=TRAIN_BATCH_SIZE,
        train_num_steps=TRAIN_NUM_STEPS,
        gradient_accumulate_every=GRADIENT_ACCUMULATE_EVERY,
        save_and_sample_every=SAVE_AND_SAMPLE_EVERY,
        results_folder=RESULTS_FOLDER,
        num_sample_rows=4,
        train_lr=1e-4,
        amp=False,
    )

    Path(RESULTS_FOLDER).mkdir(parents=True, exist_ok=True)
    trainer.train()


if __name__ == "__main__":
    main()
