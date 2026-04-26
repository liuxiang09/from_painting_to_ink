# 从照片到水墨（from_painting_to_ink）

本项目目标是把自然风景照片转换成水墨风格图像，并对比两条技术路线：

- **CycleGAN（基线）**
- **条件 Diffusion（基于 Stable Diffusion 官方库）**

仓库覆盖了完整流程：**数据构建 → 训练 → 推理 → 对比评估**。

---

## 1. 项目任务定义

- 输入：自然风景照片（content）
- 输出：水墨风格图像（stylized image）
- 训练形态：无配对（unpaired）风格迁移
- 对比目标：在相同数据域下比较 CycleGAN 与 Diffusion 的内容保持与风格贴近程度

---

## 2. 数据集构建思路

### 2.1 原始数据来源

当前默认使用两部分原始数据：

```text
from_painting_to_ink/
├─ flickr8k/
│  ├─ Images/
│  └─ captions.txt
├─ ink_src/
│  ├─ *.jpg / *.png / ...
└─ src/
```

- `flickr8k`：照片域候选数据。
- `ink_src`：水墨域候选数据（当前脚本按文件名关键词识别类别，如 `shui_mo`、`shan_shui` 等）。

### 2.2 数据筛选策略

数据准备脚本 `src/data/prepare_dataset.py` 的核心思路：

1. 使用 CLIP 对 Flickr8k 全量图片打分，筛选更接近“自然风景”的照片。
2. 从 `ink_src` 中抽取指定类别（默认 `shui_mo`）的样本。
3. 统一 resize 到训练尺寸（默认 256），导出到标准训练目录。

这样做的目的：

- 尽量减少“非风景照片”对迁移目标的干扰；
- 让两个域的数据语义更聚焦，提升训练稳定性；
- 保持无配对训练（不要求一一对应样本）。

### 2.3 导出后的标准目录

```text
data/
├─ landscape/
│  ├─ *.png
│  └─ manifest.csv
└─ ink/
   └─ shui_mo/
      ├─ *.png
      └─ manifest.csv
```

- `manifest.csv` 保存来源路径、caption、category、CLIP 分数等信息，方便追溯。

### 2.4 数据准备命令

```bash
python -m src.data.prepare_dataset \
  --flickr-root flickr8k \
  --ink-root ink_src \
  --out-root data \
  --preview-root outputs/data_preview \
  --photo-count 1000 \
  --ink-count 1000 \
  --ink-category shui_mo \
  --clip-model-name openai/clip-vit-base-patch32 \
  --image-size 256
```

会额外生成：

- `outputs/data_preview/dataset_stats.json`
- 预览图（如 `flickr_landscape_preview.png`、`ink_shui_mo_preview.png`）

---

## 3. 模型构建思路

## 3.1 CycleGAN 路线

- 两个生成器：`G_A: photo -> ink`，`G_B: ink -> photo`
- 两个判别器：`D_A`、`D_B`
- 训练损失：
  - 对抗损失（GAN）
  - 循环一致性损失（Cycle）
  - 身份损失（Identity）
  - 额外艺术损失（Content / Style / TV）

对应实现：

- `src/models/cyclegan/cycle_gan_model.py`
- `src/train/train_cyclegan.py`

## 3.2 Diffusion 路线

- 底座：`CompVis/stable-diffusion-v1-4`（可替换为本地官方 `.ckpt/.safetensors`）
- 在 latent 空间训练噪声预测（DDPM scheduler）
- 条件构造：
  - `content` 图像 + `style` 图像
  - 通过 `ImageConditionProjector` 投影为 cross-attention tokens
  - 注入 UNet 做条件去噪

训练损失由三部分组成：

- noise loss
- content loss
- style loss

对应实现：

- `src/models/diffusion/stable_diffusion.py`
- `src/train/train_diffusion.py`

---

## 4. 训练流程（含可恢复训练）

配置文件：

- `src/configs/cyclegan.yaml`
- `src/configs/diffusion.yaml`

默认训练数据路径都指向：

- `data/landscape`
- `data/ink/shui_mo`

### 4.1 CycleGAN 训练

从头训练：

```bash
python -m src.train.train_cyclegan --config src/configs/cyclegan.yaml
```

断点续训：

```bash
python -m src.train.train_cyclegan \
  --config src/configs/cyclegan.yaml \
  --resume outputs/cyclegan/checkpoints/cyclegan_epoch_020.pt
```

### 4.2 Diffusion 训练

从头训练：

```bash
python -m src.train.train_diffusion --config src/configs/diffusion.yaml
```

断点续训：

```bash
python -m src.train.train_diffusion \
  --config src/configs/diffusion.yaml \
  --resume outputs/diffusion/checkpoints/diffusion_epoch_020.pt
```

### 4.3 训练输出说明

CycleGAN 输出：

- `outputs/cyclegan/checkpoints/`
- `outputs/cyclegan/samples/`
- `outputs/cyclegan/metrics/history.csv`
- `outputs/cyclegan/metrics/*.png`（损失曲线）

Diffusion 输出：

- `outputs/diffusion/checkpoints/`
- `outputs/diffusion/samples/`
- `outputs/diffusion/metrics/history.csv`

---

## 5. 推理流程

### 5.1 CycleGAN 推理

可输入单张图片或文件夹：

```bash
python -m src.eval.infer_cyclegan \
  --checkpoint outputs/cyclegan/checkpoints/cyclegan_epoch_050.pt \
  --input data/landscape \
  --output outputs/cyclegan/infer
```

### 5.2 Diffusion 推理

Diffusion 需要同时给 content 与 style：

```bash
python -m src.eval.infer_diffusion \
  --checkpoint outputs/diffusion/checkpoints/diffusion_epoch_100.pt \
  --content data/landscape/example.png \
  --style data/ink/shui_mo/style.png \
  --output outputs/diffusion/infer \
  --steps 50 \
  --strength 0.75 \
  --style-weight 0.75
```

---

## 6. 评估对比

### 6.1 指标对比（CycleGAN vs Diffusion）

```bash
python -m src.eval.eval_metrics \
  --content data/landscape \
  --style-ref data/ink/shui_mo \
  --cyclegan outputs/cyclegan/infer \
  --diffusion outputs/diffusion/infer \
  --output outputs/eval/metrics.csv
```

输出：

- `outputs/eval/metrics.csv`
- 包含 `content_l1`、`style_distance`、`simple_fid_vs_style`

---

## 8. 代码结构

```text
src/
├─ configs/   # 训练配置
├─ data/      # 数据准备与数据集读取
├─ eval/      # 推理与指标计算
├─ models/    # CycleGAN / Diffusion 模型实现
└─ train/     # 训练入口与监控
```

---

## 9. 环境依赖

```bash
pip install torch torchvision pillow pyyaml numpy matplotlib tqdm diffusers transformers safetensors accelerate
```

说明：

- 首次运行 CLIP 筛选与 SD 底座会下载模型权重，需要网络与磁盘空间。
