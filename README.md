# 案例九实验报告

## 1. 任务说明

本案例的目标是完成“自然图像到水墨风格图像”的无配对风格迁移，并比较两条生成路线：

- `CycleGAN`：以双向生成器和判别器完成跨域映射。
- `Diffusion`：以条件编码器和噪声预测 `UNet` 完成风格约束生成。

当前工程中的训练脚本统一使用 `UnpairedDataset`，因此照片域和水墨域不要求一一配对，只要求两个域各自具备可读图像即可。

## 2. 当前数据集构成

结合当前工作区目录与代码逻辑，项目使用的数据源不是单一数据集，而是三部分混合构成：

| 域 | 路径 | 当前确认规模 | 在项目中的角色 | 备注 |
| --- | --- | --- | --- | --- |
| 照片域 | `flickr8k/Images` | 8091 张图像 | 主要内容图来源 | `captions.txt` 中共有 40455 条描述，约等于每图 5 条 caption |
| 照片域补充 | `laion-art/laion-art.parquet` | 1 个 parquet 文件，约 1.28 GB | 用于补充照片/艺术图像语义多样性 | `prepare_dataset.py` 默认最多读取 20000 条，但前提是环境已安装 `pyarrow` 或 `pandas` |
| 水墨域 | `ink_src/all_after_clean` | 5798 张图像 | 主要风格图来源 | 这是当前最干净、最适合直接导出的水墨图像目录 |
| 水墨域污染项 | `ink_src/__MACOSX/all_after_clean` | 5798 个 `._*` 文件 | 当前目录扫描时会被误当作图像文件 | 这些是 macOS 打包残留资源文件，不应视为有效训练样本 |

因此，按照“当前目录真实构成”来看：

- 照片域由 `Flickr8k + LAION-art parquet` 组成。
- 水墨域目录表面上有 `11596` 个带图像后缀的文件，但其中一半来自 `__MACOSX`，属于打包残留。
- 如果直接把 `ink_src` 作为根目录递归读取，现有加载器会把 `._gong_bi_hua_xxx.jpg` 这类文件也纳入候选列表，训练时只能依靠解码失败后的重试机制跳过，效率和稳定性都会受影响。

## 3. 与当前代码实现的对应关系

数据准备脚本 `case9/data/prepare_dataset.py` 的逻辑如下：

- `flickr8k` 与 `laion-art` 会合并成照片域，最终导出到 `data/landscape`
- `ink_src` 会作为水墨域导出到 `data/ink`
- 默认划分比例为 `train:val:test = 0.8:0.1:0.1`
- 默认导出分辨率为 `256 x 256`

训练配置文件的目标目录已经固定为：

- `case9/configs/cyclegan.yaml`：`data/landscape/train` 与 `data/ink/train`
- `case9/configs/diffusion.yaml`：`data/landscape/train` 与 `data/ink/train`

这说明当前工程的标准流程并不是直接拿原始目录训练，而是先将原始数据整理到统一的 `data/` 结构下，再进入模型训练。

## 4. 当前数据集的工程性结论

基于这份工作区的现状，可以得出几个直接影响实验结果的结论：

1. `Flickr8k` 是当前最稳定、最明确可用的照片域数据，共 `8091` 张图像。
2. `LAION-art` 已经放入项目，但目前是 `parquet` 形态；只有在环境补齐 `pyarrow` 或 `pandas` 后，数据准备脚本才能真正读取它。
3. `ink_src` 中真正干净的水墨图像应以 `all_after_clean` 下的 `5798` 张为准。
4. `__MACOSX` 目录属于数据噪声。若不显式避开，报告中的“样本规模”会被虚增到 `11596`，但这并不代表真实有效样本翻倍。

如果以“当前最稳妥、最符合真实可用数据”的口径写实验设置，那么推荐的有效数据构成为：

- 照片域：`Flickr8k 8091` 张 + `LAION-art` 中后续成功解析出的样本
- 水墨域：`ink_src/all_after_clean` 中 `5798` 张

## 5. 推荐的数据整理方式

为了让导出的训练集和报告保持一致，不建议直接使用 `ink_src` 作为 `--ink-root`，而应显式指向清洗后的子目录：

```bash
python -m case9.data.prepare_dataset \
  --flickr-root flickr8k \
  --ink-root ink_src/all_after_clean \
  --laion-root laion-art \
  --out-root data \
  --preview-root outputs/data_preview \
  --image-size 256 \
  --train-ratio 0.8 \
  --val-ratio 0.1
```

在当前环境中，还需要先补齐依赖，否则脚本无法运行：

```bash
pip install -r requirements.txt
pip install pyarrow
```

如果暂时不引入 `LAION-art`，仅按当前已确认的本地图像统计，则默认划分后的规模大致为：

| 数据域 | 总量 | 训练集 | 验证集 | 测试集 |
| --- | --- | --- | --- | --- |
| Flickr8k | 8091 | 6472 | 809 | 810 |
| Ink（clean） | 5798 | 4638 | 579 | 581 |

其中照片域在引入 `LAION-art` 后会继续增大，但具体数量取决于 parquet 中可解析记录数以及是否保留脚本默认的 `--limit-laion 20000`。

## 6. 模型训练设置

### 6.1 CycleGAN

`case9/configs/cyclegan.yaml` 当前配置为：

- 图像尺寸：`256`
- 批大小：`4`
- 训练轮数：`50`
- 学习率：`2e-4`
- 风格损失权重：`0.8`
- 循环一致性权重：`10.0`
- 身份损失权重：`5.0`

训练命令：

```bash
python -m case9.train.train_cyclegan --config case9/configs/cyclegan.yaml
```

### 6.2 Diffusion

`case9/configs/diffusion.yaml` 当前配置为：

- 图像尺寸：`256`
- 批大小：`2`
- 训练轮数：`100`
- 学习率：`1e-4`
- 扩散步数：`1000`
- 风格损失权重：`0.75`

训练命令：

```bash
python -m case9.train.train_diffusion --config case9/configs/diffusion.yaml
```

## 7. 推理与评估流程

### 7.1 CycleGAN 推理

```bash
python -m case9.infer.infer_cyclegan \
  --checkpoint outputs/cyclegan/checkpoints/cyclegan_epoch_050.pt \
  --input data/landscape/test \
  --output outputs/cyclegan/infer
```

### 7.2 Diffusion 推理

```bash
python -m case9.infer.infer_diffusion \
  --checkpoint outputs/diffusion/checkpoints/diffusion_epoch_100.pt \
  --content data/landscape/test/example.png \
  --style data/ink/test/style.png \
  --output outputs/diffusion/infer
```

### 7.3 定量评估

```bash
python -m case9.eval.compare_metrics \
  --content data/landscape/test \
  --style-ref data/ink/test \
  --cyclegan outputs/cyclegan/infer \
  --diffusion outputs/diffusion/infer \
  --output outputs/eval/metrics.csv
```

当前评估脚本输出三类指标：

- `content_l1`：衡量内容保留程度
- `style_distance`：衡量与风格参考图之间的纹理/颜色统计差异
- `simple_fid_vs_style`：基于通道均值与方差的简化分布距离

### 7.4 可视化对比

```bash
python -m case9.eval.visual_panel \
  --content data/landscape/test \
  --style data/ink/test \
  --cyclegan outputs/cyclegan/infer \
  --diffusion outputs/diffusion/infer \
  --output outputs/eval/panels
```

## 8. 总结

本次案例的数据集不再适合写成“照片集 + 水墨集”的笼统描述，而应明确说明其真实构成为：

- 内容域以 `Flickr8k` 为主，`LAION-art parquet` 为补充
- 风格域来自 `ink_src/all_after_clean`
- `ink_src/__MACOSX` 是当前数据目录中的噪声来源

因此，重写后的实验报告应强调两点：一是项目采用无配对跨域训练；二是当前最重要的数据治理任务不是继续扩充模型结构，而是先确保 `LAION-art` 可读、`__MACOSX` 不再混入训练集。只有在这一前提下，后续的 CycleGAN 与 Diffusion 对比才具有可解释性和可信度。
