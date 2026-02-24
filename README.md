# my-video-diffusion-pytorch
这是我自己的练习repo，关于video diffusion的pytorch实现。

## 无条件 Moving MNIST 训练

1. 安装依赖：`pip install -r requirements.txt`
2. 在项目根目录（即 `my-video-diffusion-pytorch` 目录）下执行：
   ```bash
   python -m video_diffusion.train_moving_mnist
   ```
3. 首次运行会自动下载 MNIST 到 `./data/mnist`； checkpoint 与采样 GIF 保存在 `./results_moving_mnist`。
4. 可在 `video_diffusion/train_moving_mnist.py` 顶部修改 `IMAGE_SIZE`、`NUM_FRAMES`、`TRAIN_NUM_STEPS`、`SAVE_AND_SAMPLE_EVERY` 等参数。
