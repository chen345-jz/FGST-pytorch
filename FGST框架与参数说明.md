# FGST 毫米波雷达步态识别框架与参数说明

本文档说明当前项目保留的唯一方法：基于毫米波雷达点云的细粒度时空步态识别网络。当前实现文件为 `python/fgst_reid.py`，一键运行脚本为 `run_fgst_reid.bat`。

## 1. 当前框架概览

当前模型采用 `FGSTReID` 框架，主体由三个核心模块组成：

```text
DSFE -> PGBP -> LGTE
```

对应含义如下：

- `DSFE`：Dual-Stream Feature Extraction，双流点级特征提取模块。
- `PGBP`：Probability-Guided Body-Part Partition，概率引导身体部位划分模块。
- `LGTE`：Local-Global Temporal Feature Extraction，局部-全局时间特征提取模块。

整体数据流为：

```text
雷达点云序列
 -> 数据标准化与 z 轴排序
 -> DSFE 双流点级特征提取
 -> PGBP 身体部位划分
 -> LGTE 多尺度时间建模
 -> 多部位特征拼接
 -> ReID embedding
 -> Triplet 检索训练与 Rank/mAP 评估
```

当前工程已经清理旧方法，仅保留上述 DSFE/PGBP/LGTE 主线。

## 2. 输入数据格式

当前模型使用 4 维点属性：

```text
x, y, z, velocity
```

输入张量形状为：

```text
[B, T, N, C]
```

其中：

- `B`：batch size。
- `T`：每个样本的帧数，当前默认 `20`。
- `N`：每帧固定点数，当前默认 `80`。
- `C`：点属性维度，当前固定为 `4`。

因此当前默认输入形状为：

```text
[B, 20, 80, 4]
```

说明：论文原始设置包含 RCS 维度，但当前数据文件为 4 维点云，因此本项目已直接去掉第五维，不再补零、不再进行 RCS 归一化。

## 3. 数据预处理

数据预处理函数为：

```text
standardize_radar_sequence(...)
```

主要处理步骤如下：

1. 固定帧数为 `frame_num=20`。
2. 固定每帧点数为 `num_points=80`。
3. 点属性维度固定为 `feature_dim=4`。
4. 若帧数不足，则重复最后一帧；若帧数过多，则从中间截取。
5. 若点数不足，则随机重复已有点；若点数过多，则随机采样或截断。
6. 对每一帧的 `x,y,z` 坐标减去该帧点云质心，降低绝对位置影响。
7. 按 `z` 轴高度从高到低排序，为后续 PGBP 身体部位划分做准备。

预处理后的数据保持：

```text
[20, 80, 4]
```

## 4. DSFE 双流特征提取模块

DSFE 的作用是从每一帧点云中提取点级空间与运动特征。当前实现中，DSFE 分为两条分支：

```text
空间分支：x, y, z
速度分支：velocity
```

空间分支输入为 3 维：

```text
[x, y, z]
```

速度分支输入为 1 维：

```text
[velocity]
```

每条分支均使用 MLP 提取特征，通道配置为：

```text
32 -> 64 -> 128
```

速度分支会在中间层融合到空间分支中：

```text
velocity feature -> concat -> spatial feature
```

最终两条分支输出拼接后，通过融合 MLP 映射到 512 维点特征：

```text
[B, T, N, 4]
 -> DSFE
[B, T, N, 512]
```

当前 DSFE 关键参数：

| 参数 | 当前值 | 说明 |
|---|---:|---|
| 空间输入维度 | `3` | `x,y,z` |
| 速度输入维度 | `1` | `velocity` |
| MLP 通道 | `32, 64, 128` | 点级特征提取 |
| 融合输出通道 | `512` | DSFE 输出点特征维度 |
| 激活函数 | `LeakyReLU(0.1)` | 负斜率为 0.1 |
| 归一化 | `BatchNorm1d` | 用于 MLP 输出 |

## 5. PGBP 概率引导身体部位划分模块

PGBP 的作用是将一帧内的点级特征划分为多个身体部位特征。

由于毫米波雷达点云稀疏，不同身体部位反射点数量并不均匀。例如躯干部位通常反射点更多，头部、手臂、脚部可能更少。因此当前实现不采用简单均匀切分，而是根据训练集高度分布估计身体部位概率。

概率估计函数为：

```text
estimate_pgbp_probabilities(...)
```

默认身体部位数量为：

```text
num_parts = 16
```

训练开始时会输出类似：

```text
[fgst] PGBP probabilities: [...]
```

PGBP 的处理流程：

1. 输入点特征已经按 `z` 轴从高到低排序。
2. 根据训练集高度分布概率，将 80 个点划分成 16 个身体区域。
3. 每个区域至少分到 1 个点，避免小概率区域为空。
4. 对每个区域执行：

```text
Global Max Pooling + Global Average Pooling
```

输出形状为：

```text
[B, T, N, 512]
 -> PGBP
[B, T, K, 512]
```

其中：

```text
K = 16
```

当前 PGBP 关键参数：

| 参数 | 当前值 | 说明 |
|---|---:|---|
| 身体部位数 | `16` | 默认将人体高度方向划分为 16 个 part |
| 输入点数 | `80` | 每帧点数 |
| 输入通道 | `512` | DSFE 输出维度 |
| 区域聚合 | `max + mean` | 同时保留显著特征与平均分布 |
| 划分依据 | 高度分布概率 | 由训练集估计 |

## 6. LGTE 局部-全局时间建模模块

LGTE 的作用是对每个身体部位沿时间维度建模。PGBP 输出后，每个身体部位都有一条时间序列：

```text
[B, T, 512]
```

当前模型为每个身体部位使用独立的 LGTE 模块，即 16 个 part 对应 16 个独立时间建模器。这样可以让腿部、躯干、手臂等不同部位学习不同的运动模式。

每个 LGTE 模块包含 4 条 temporal convolution 分支：

```text
dilation = 1, 2, 3, 4
```

每条分支输出 128 通道：

```text
128 * 4 = 512
```

随后使用 attention 模块计算每帧、每通道的重要性权重，并进行加权时间汇聚：

```text
weighted temporal summation
```

LGTE 输出后，每个 part 的 512 维时间特征会经过全连接层映射到 256 维：

```text
[B, T, 16, 512]
 -> LGTE
[B, 16, 256]
```

当前 LGTE 关键参数：

| 参数 | 当前值 | 说明 |
|---|---:|---|
| temporal kernel size | `3` | 时间卷积核大小 |
| dilation | `1,2,3,4` | 多尺度时间感受野 |
| 每分支输出通道 | `128` | 四个分支合计 512 |
| attention kernel size | `3` | 时间注意力卷积 |
| part 输出维度 | `256` | 每个身体部位最终特征维度 |
| part 模块共享 | 不共享 | 每个 part 独立 LGTE |

## 7. 最终 ReID 特征

LGTE 输出为：

```text
[B, 16, 256]
```

随后将 16 个身体部位特征拼接：

```text
16 * 256 = 4096
```

最终 ReID embedding 形状为：

```text
[B, 4096]
```

模型训练时返回：

```text
feature, logits, parts
```

评估与预测时返回：

```text
feature
```

当前最终特征参数：

| 参数 | 当前值 | 说明 |
|---|---:|---|
| part 数量 | `16` | PGBP 输出 |
| 每个 part 特征维度 | `256` | LGTE 后 FC 输出 |
| 最终 embedding 维度 | `4096` | 16 个 part 拼接 |
| 归一化 | `LayerNorm` | 对最终 embedding 归一化 |

## 8. 损失函数设置

当前损失函数为：

```text
FGSTLoss
```

包含两部分：

```text
global Batch-all Triplet Loss
part-level Batch-all Triplet Loss
```

总损失为：

```text
L = L_global_triplet + part_weight * L_part_triplet + ce_weight * L_ce
```

当前默认参数为：

```text
margin = 0.2
part_weight = 0.5
ce_weight = 0.0
```

说明：

- `global triplet` 约束最终 4096 维身份特征。
- `part-level triplet` 约束每个身体部位的 256 维局部特征。
- `ce_weight=0.0` 表示默认不启用分类交叉熵损失，更偏向检索式 metric learning。

损失函数参数：

| 参数 | 当前值 | 说明 |
|---|---:|---|
| triplet margin | `0.2` | 正负样本距离间隔 |
| part_weight | `0.5` | 局部 part triplet 权重 |
| ce_weight | `0.0` | 默认关闭 CE 分类损失 |
| loss 类型 | Batch-all Triplet | 使用 batch 内所有有效三元组 |

## 9. 训练参数设置

当前默认训练参数来自 `HParams` 和命令行参数。

| 参数 | 当前值 | 说明 |
|---|---:|---|
| `radar_base_dir` | `.\2s` | 数据集目录 |
| `split_ratio` | `0.7` | 按人员 ID 划分训练/测试 |
| `num_epochs` | `50` | 训练轮数 |
| `batch_size` | `8` | 物理 batch size |
| `accumulation_steps` | `2` | 梯度累积步数 |
| `learning_rate` | `0.0001` | AdamW 初始学习率 |
| `weight_decay` | `0.0001` | L2 正则 |
| `clip_grad_norm` | `1.0` | 梯度裁剪 |
| `eval_interval` | `1` | 每轮评估一次 |
| `early_stopping_patience` | `20` | 早停耐心值 |
| `seed` | `42` | 随机种子 |
| `frame_num` | `20` | 每个样本帧数 |
| `num_points` | `80` | 每帧点数 |
| `feature_dim` | `4` | 输入点属性维度 |
| `num_parts` | `16` | PGBP 身体部位数 |
| `feat_dim` | `256` | 每个 part 输出维度 |

优化器设置：

```text
AdamW
learning_rate = 1e-4
betas = (0.9, 0.999)
weight_decay = 1e-4
```

学习率调度：

```text
warmup + cosine decay
```

其中 warmup epoch 为：

```text
int(num_epochs * 0.1)
```

在当前 `50` epoch 设置下：

```text
warmup_epochs = 5
```

## 10. Batch 采样策略

训练使用 `BalancedBatchSampler`，用于保证每个 batch 中包含多个身份和每个身份的多个样本。

当前设置：

```text
n_classes_per_batch = min(2, len(train_ids))
n_samples_per_class = batch_size // n_classes_per_batch
```

在默认 `batch_size=8` 时：

```text
n_classes_per_batch = 2
n_samples_per_class = 4
```

因此每个 batch 约为：

```text
2 个身份 * 每个身份 4 个样本 = 8 个样本
```

这与 Triplet Loss 的训练需求一致，因为 triplet 需要同身份正样本和不同身份负样本。

## 11. 训练/测试划分与 ReID 评估协议

当前数据按人员 ID 划分：

```text
train IDs: 70%
test IDs: 30%
```

训练身份和测试身份不重叠，符合 ReID 检索任务设定。

评估时，对测试身份样本继续划分为：

```text
gallery: 70%
query: 30%
```

评估流程：

1. 提取 gallery embedding。
2. 提取 query embedding。
3. 对 embedding 做 L2 normalize。
4. 计算 query 与 gallery 的相似度矩阵。
5. 输出 Rank-1、Rank-3、Rank-5 和 mAP。

主要指标：

| 指标 | 含义 |
|---|---|
| Rank-1 | query 的最近 gallery 样本是否为正确身份 |
| Rank-3 | 正确身份是否出现在前 3 个检索结果中 |
| Rank-5 | 正确身份是否出现在前 5 个检索结果中 |
| mAP | 平均检索精度 |

## 12. 当前运行命令

一键训练：

```powershell
run_fgst_reid.bat
```

手动训练：

```powershell
D:\cppsoft\venvs\mmwave_pt\Scripts\python.exe .\python\fgst_reid.py train --radar_base_dir ".\2s" --split_ratio 0.7 --num_epochs 50
```

评估：

```powershell
D:\cppsoft\venvs\mmwave_pt\Scripts\python.exe .\python\fgst_reid.py eval --radar_base_dir ".\2s" --split_ratio 0.7 --seed 42
```

单样本预测：

```powershell
D:\cppsoft\venvs\mmwave_pt\Scripts\python.exe .\python\fgst_reid.py predict .\2s\p_1\0.npy --radar_base_dir ".\2s" --split_ratio 0.7 --seed 42
```

批量随机预测：

```powershell
D:\cppsoft\venvs\mmwave_pt\Scripts\python.exe .\python\fgst_reid.py predict_batch --radar_base_dir ".\2s" --split_ratio 0.7 --seed 42 --num_samples 50
```

## 13. 当前 4 维版本完整训练结果

当前 4 维输入版本已完成 50 epoch 完整训练。

最佳测试结果：

```text
best test Rank-1 = 93.33%
best test mAP    = 76.80%
```

最终汇总结果：

```text
final train Rank-1 = 96.19%
final train mAP    = 86.85%

final test Rank-1  = 93.33%
final test mAP     = 76.80%
final test R3      = 97.78%
final test R5      = 99.26%
```

模型产物：

```text
checkpoints_fgst/best_model_fgst.pth
checkpoints_fgst/training_progress_fgst.png
```

## 14. 当前工程保留内容

当前项目中与模型相关的核心内容为：

```text
python/fgst_reid.py
run_fgst_reid.bat
checkpoints_fgst/
README.md
python/README_pt.md
```

旧的 PointNet-LSTM baseline 已经删除，当前工程只保留 DSFE/PGBP/LGTE 细粒度时空步态识别框架。
