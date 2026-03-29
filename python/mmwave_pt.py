import argparse
import csv
import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# 简单的 key=value 配置读取器（兼容当前 cfg 文件格式）。
def load_cfg(path: Path) -> Dict[str, str]:
    cfg: Dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        cfg[k.strip()] = v.strip()
    return cfg


def cfg_int(cfg: Dict[str, str], key: str, default: int) -> int:
    try:
        return int(cfg.get(key, str(default)))
    except ValueError:
        return default


def cfg_float(cfg: Dict[str, str], key: str, default: float) -> float:
    try:
        return float(cfg.get(key, str(default)))
    except ValueError:
        return default


def cfg_str(cfg: Dict[str, str], key: str, default: str) -> str:
    return cfg.get(key, default)


def cfg_int_list(cfg: Dict[str, str], key: str, default: List[int]) -> List[int]:
    raw = cfg.get(key, "")
    if not raw:
        return default
    out: List[int] = []
    for x in raw.split(","):
        x = x.strip()
        if x and (x.lstrip("-").isdigit()):
            out.append(int(x))
    return out if out else default


def resolve_cfg_path(raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    return (Path.cwd() / p).resolve()


    # 训练相关超参数
@dataclass
class FGSTConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_frames: int
    max_points_per_frame: int
    num_body_parts: int
    point_feature_dim: int
    temporal_feature_dim: int
    temporal_dilations: List[int]
    negative_slope: float
    dropout: float


    # 训练增强与优化策略开关
@dataclass
class TrainConfig:
    use_class_weight: bool
    lr_decay_milestones: List[int]
    lr_decay_gamma: float
    save_best_only: bool
    use_triplet_loss: bool
    triplet_margin: float
    triplet_weight: float
    use_part_triplet: bool
    part_triplet_weight: float
    adam_beta1: float
    adam_beta2: float


    # 数据增强参数
@dataclass
class AugmentConfig:
    enable: bool
    jitter_std: float
    point_dropout: float
    time_shift: int


class NpyGaitDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[Path, int]],
        fgst_cfg: FGSTConfig,
        is_train: bool = False,
        augment_cfg: AugmentConfig | None = None,
    ):
        self.samples = samples
        self.fgst_cfg = fgst_cfg
        self.is_train = is_train
        self.augment_cfg = augment_cfg

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, out: np.ndarray):
        if not self.augment_cfg or not self.augment_cfg.enable:
            return out

        valid = out[:, :, 4] > 0.0

        # 空间抖动：仅对有效点添加 xyz 高斯噪声
        if self.augment_cfg.jitter_std > 0:
            noise = np.random.normal(
                loc=0.0,
                scale=self.augment_cfg.jitter_std,
                size=out[:, :, :3].shape,
            ).astype(np.float32)
            out[:, :, :3] = out[:, :, :3] + noise * valid[:, :, None]

        # 点丢弃：模拟雷达点云稀疏与随机缺失
        if self.augment_cfg.point_dropout > 0:
            drop = np.random.rand(*valid.shape) < self.augment_cfg.point_dropout
            drop = np.logical_and(drop, valid)
            out[drop] = 0.0

        # 时间平移：在帧维度上随机平移，提升时序鲁棒性
        if self.augment_cfg.time_shift > 0:
            shift = random.randint(-self.augment_cfg.time_shift, self.augment_cfg.time_shift)
            if shift != 0:
                rolled = np.roll(out, shift=shift, axis=0)
                if shift > 0:
                    rolled[:shift, :, :] = 0.0
                else:
                    rolled[shift:, :, :] = 0.0
                out = rolled
        return out

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        arr = np.load(path)
        # 期望输入: [T, P, C]，且 C>=4（x,y,z,doppler）
        if arr.ndim != 3 or arr.shape[2] < 4:
            raise ValueError(f"Invalid npy shape for {path}: {arr.shape}")

        t_lim = min(self.fgst_cfg.max_frames, arr.shape[0])
        p_lim = min(self.fgst_cfg.max_points_per_frame, arr.shape[1])
        out = np.zeros(
            (self.fgst_cfg.max_frames, self.fgst_cfg.max_points_per_frame, 5),
            dtype=np.float32,
        )
        # 第5维补常量 SNR，保持与原工程特征维度一致（5维）
        out[:t_lim, :p_lim, :4] = arr[:t_lim, :p_lim, :4].astype(np.float32)
        out[:t_lim, :p_lim, 4] = 10.0  # snr constant as in C++ pipeline

        # 帧内中心化：减去每一帧有效点的xyz质心，减小绝对距离影响。
        for t in range(self.fgst_cfg.max_frames):
            valid = out[t, :, 4] > 0.0
            if not np.any(valid):
                continue
            centroid = out[t, valid, :3].mean(axis=0, keepdims=True)
            out[t, valid, :3] = out[t, valid, :3] - centroid

        if self.is_train:
            out = self._augment(out)
        return torch.from_numpy(out), int(label), str(path)


class MultiScaleTemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilations: List[int], negative_slope: float = 0.1):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=d, dilation=d),
                    nn.LeakyReLU(negative_slope=negative_slope),
                )
                for d in dilations
            ]
        )
        hidden = max(16, out_ch // 2)
        self.scale_attn = nn.Sequential(
            nn.Linear(len(dilations) * out_ch, hidden),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Linear(hidden, len(dilations)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [b(x) for b in self.branches]  # each [B, C, T]
        # 先得到每个尺度的全局描述，再预测尺度权重。
        pooled = [f.max(dim=2).values for f in feats]  # each [B, C]
        attn_logits = self.scale_attn(torch.cat(pooled, dim=1))  # [B, S]
        attn = torch.softmax(attn_logits, dim=1)
        out = 0.0
        for i, f in enumerate(feats):
            out = out + f * attn[:, i].view(-1, 1, 1)
        return out


class FgstNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        point_dim: int,
        temporal_dim: int,
        num_parts: int,
        num_classes: int,
        temporal_dilations: List[int],
        negative_slope: float,
        dropout: float,
    ):
        super().__init__()
        self.num_parts = num_parts
        self.point_dim = point_dim
        half_dim = max(16, point_dim // 2)

        # DSFE: 空间-RCS流 与 空间-速度流，最后做融合。
        self.spatial_rcs_mlp = nn.Sequential(
            nn.Linear(4, half_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Linear(half_dim, half_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.spatial_vel_mlp = nn.Sequential(
            nn.Linear(4, half_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Linear(half_dim, half_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
        )
        self.point_fuse = nn.Sequential(
            nn.Linear(half_dim * 2, point_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
        )

        self.part_prob = nn.Linear(point_dim, num_parts)
        self.part_temporal = MultiScaleTemporalBlock(
            in_ch=point_dim,
            out_ch=temporal_dim,
            dilations=temporal_dilations,
            negative_slope=negative_slope,
        )
        self.global_temporal = MultiScaleTemporalBlock(
            in_ch=point_dim,
            out_ch=temporal_dim,
            dilations=temporal_dilations,
            negative_slope=negative_slope,
        )
        self.embedding_head = nn.Sequential(
            nn.Linear((num_parts + 1) * temporal_dim, temporal_dim),
            nn.LeakyReLU(negative_slope=negative_slope),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(temporal_dim, num_classes)

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        # 输入 x: [B, T, P, C]
        b, t, p, c = x.shape
        xp = x.reshape(b * t * p, c)

        xyz = xp[:, 0:3]
        doppler = xp[:, 3:4]
        snr = xp[:, 4:5]
        feat_rcs = self.spatial_rcs_mlp(torch.cat([xyz, snr], dim=1))
        feat_vel = self.spatial_vel_mlp(torch.cat([xyz, doppler], dim=1))
        point_feat = self.point_fuse(torch.cat([feat_rcs, feat_vel], dim=1))  # [B*T*P, F]

        prob = torch.softmax(self.part_prob(point_feat), dim=1)  # [B*T*P, K]
        point_feat = point_feat.reshape(b, t, p, -1)  # [B, T, P, F]
        prob = prob.reshape(b, t, p, self.num_parts)  # [B, T, P, K]

        # 全局时序分支：先按点做全局池化，再做时序卷积
        global_frame = point_feat.max(dim=2).values  # [B, T, F]
        global_ts = self.global_temporal(global_frame.transpose(1, 2))  # [B, H, T]
        global_vec = global_ts.max(dim=2).values  # [B, H]

        # 局部时序分支：按 part 概率加权聚合后建模时序
        part_vecs = []
        for k in range(self.num_parts):
            wk = prob[:, :, :, k].unsqueeze(-1)  # [B, T, P, 1]
            weighted = point_feat * wk
            pooled = weighted.sum(dim=2) / (wk.sum(dim=2) + 1e-6)  # [B, T, F]
            ts = self.part_temporal(pooled.transpose(1, 2))  # [B, H, T]
            part_vecs.append(ts.max(dim=2).values)  # [B, H]

        # [B, K, H]，每个 body part 的时序嵌入
        part_stack = torch.stack(part_vecs, dim=1)
        part_cat = torch.cat(part_vecs, dim=1)
        fused = torch.cat([part_cat, global_vec], dim=1)
        embedding = self.embedding_head(fused)
        logits = self.classifier(embedding)
        if return_embedding:
            return logits, embedding, part_stack
        return logits


def batch_hard_triplet_loss(embeddings: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    # embeddings: [B, D], labels: [B]
    if embeddings.ndim != 2 or labels.ndim != 1:
        raise ValueError("batch_hard_triplet_loss expects embeddings [B,D] and labels [B].")
    if embeddings.size(0) < 2:
        return embeddings.new_zeros(())

    emb = F.normalize(embeddings, p=2, dim=1)
    dist = torch.cdist(emb, emb, p=2)
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    diff = ~same

    # 不把自身当作正样本
    eye = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    pos_mask = same & (~eye)
    neg_mask = diff

    # hardest positive: 同类中最远
    pos_dist = dist.masked_fill(~pos_mask, float("-inf"))
    hardest_pos = pos_dist.max(dim=1).values
    valid_pos = pos_mask.any(dim=1)

    # hardest negative: 异类中最近
    neg_dist = dist.masked_fill(~neg_mask, float("inf"))
    hardest_neg = neg_dist.min(dim=1).values
    valid_neg = neg_mask.any(dim=1)

    valid = valid_pos & valid_neg
    if not torch.any(valid):
        return embeddings.new_zeros(())

    losses = F.relu(hardest_pos[valid] - hardest_neg[valid] + margin)
    return losses.mean()


def collect_samples(npy_root: Path) -> List[Tuple[Path, int]]:
    samples: List[Tuple[Path, int]] = []
    for pdir in sorted(npy_root.iterdir()):
        if not pdir.is_dir():
            continue
        name = pdir.name
        if not name.startswith("p_"):
            continue
        try:
            label = int(name.split("_", 1)[1])
        except ValueError:
            continue
        for npy in sorted(pdir.glob("*.npy")):
            samples.append((npy, label))
    return samples


# 按身份标签分层划分，尽量保持每类训练/测试比例一致
def split_samples(samples: List[Tuple[Path, int]], test_ratio: float, seed: int):
    rng = random.Random(seed)
    by_label: Dict[int, List[Tuple[Path, int]]] = {}
    for s in samples:
        by_label.setdefault(s[1], []).append(s)
    train, test = [], []
    for label, lst in by_label.items():
        rng.shuffle(lst)
        n_test = max(1, int(round(len(lst) * test_ratio)))
        test.extend(lst[:n_test])
        train.extend(lst[n_test:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def compute_metrics(y_true: List[int], y_pred: List[int]):
    labels = sorted(set(y_true))
    tp = {k: 0 for k in labels}
    fp = {k: 0 for k in labels}
    fn = {k: 0 for k in labels}
    for gt, pd in zip(y_true, y_pred):
        if gt == pd:
            tp[gt] += 1
        else:
            if pd in fp:
                fp[pd] += 1
            if gt in fn:
                fn[gt] += 1

    # 逐类统计 Precision / Recall / F1，并计算总体 accuracy 与 macro-F1
    per = []
    for k in labels:
        p = tp[k] / (tp[k] + fp[k]) if (tp[k] + fp[k]) else 0.0
        r = tp[k] / (tp[k] + fn[k]) if (tp[k] + fn[k]) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        per.append((k, p, r, f1))
    acc = sum(int(a == b) for a, b in zip(y_true, y_pred)) / max(1, len(y_true))
    macro_f1 = sum(x[3] for x in per) / max(1, len(per))
    return acc, macro_f1, per


def print_metrics(acc: float, macro_f1: float, per):
    print(f"top1_accuracy: {acc:.6f}")
    print(f"macro_f1: {macro_f1:.6f}")
    for k, p, r, f1 in per:
        print(f"label {k} -> P: {p:.6f}, R: {r:.6f}, F1: {f1:.6f}")


def save_metrics_csv(path: Path, acc: float, macro_f1: float, per):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["top1_accuracy", f"{acc:.6f}"])
        w.writerow(["macro_f1", f"{macro_f1:.6f}"])
        w.writerow([])
        w.writerow(["label", "precision", "recall", "f1"])
        for k, p, r, f1 in per:
            w.writerow([k, f"{p:.6f}", f"{r:.6f}", f"{f1:.6f}"])


def build_model(cfg: FGSTConfig, num_classes: int):
    return FgstNet(
        in_dim=5,
        point_dim=cfg.point_feature_dim,
        temporal_dim=cfg.temporal_feature_dim,
        num_parts=cfg.num_body_parts,
        num_classes=num_classes,
        temporal_dilations=cfg.temporal_dilations,
        negative_slope=cfg.negative_slope,
        dropout=cfg.dropout,
    )


def run_predict_epoch(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            pred = torch.argmax(logits, dim=1).cpu().tolist()
            y_pred.extend(pred)
            y_true.extend(y.tolist())
    return y_true, y_pred


def run_train(cfg: Dict[str, str], cfg_path: Path):
    fgst_cfg = FGSTConfig(
        epochs=cfg_int(cfg, "fgst.epochs", 50),
        batch_size=cfg_int(cfg, "fgst.batch_size", 16),
        learning_rate=cfg_float(cfg, "fgst.learning_rate", 1e-3),
        weight_decay=cfg_float(cfg, "fgst.weight_decay", 1e-4),
        max_frames=cfg_int(cfg, "fgst.max_frames", 20),
        max_points_per_frame=cfg_int(cfg, "fgst.max_points_per_frame", 80),
        num_body_parts=cfg_int(cfg, "fgst.num_body_parts", 4),
        point_feature_dim=cfg_int(cfg, "fgst.point_feature_dim", 128),
        temporal_feature_dim=cfg_int(cfg, "fgst.temporal_feature_dim", 128),
        temporal_dilations=cfg_int_list(cfg, "fgst.temporal_dilations", [1, 2, 3, 4]),
        negative_slope=cfg_float(cfg, "fgst.negative_slope", 0.1),
        dropout=cfg_float(cfg, "fgst.dropout", 0.2),
    )
    train_cfg = TrainConfig(
        use_class_weight=cfg_str(cfg, "train.use_class_weight", "true").lower() in ("1", "true", "yes", "on"),
        lr_decay_milestones=[
            int(x.strip())
            for x in cfg_str(cfg, "train.lr_decay_milestones", "20,35").split(",")
            if x.strip().isdigit()
        ],
        lr_decay_gamma=cfg_float(cfg, "train.lr_decay_gamma", 0.5),
        save_best_only=cfg_str(cfg, "train.save_best_only", "true").lower() in ("1", "true", "yes", "on"),
        use_triplet_loss=cfg_str(cfg, "train.use_triplet_loss", "true").lower() in ("1", "true", "yes", "on"),
        triplet_margin=cfg_float(cfg, "train.triplet_margin", 0.2),
        triplet_weight=cfg_float(cfg, "train.triplet_weight", 0.5),
        use_part_triplet=cfg_str(cfg, "train.use_part_triplet", "true").lower() in ("1", "true", "yes", "on"),
        part_triplet_weight=cfg_float(cfg, "train.part_triplet_weight", 0.5),
        adam_beta1=cfg_float(cfg, "train.adam_beta1", 0.9),
        adam_beta2=cfg_float(cfg, "train.adam_beta2", 0.999),
    )
    aug_cfg = AugmentConfig(
        enable=cfg_str(cfg, "augment.enable", "true").lower() in ("1", "true", "yes", "on"),
        jitter_std=cfg_float(cfg, "augment.jitter_std", 0.01),
        point_dropout=cfg_float(cfg, "augment.point_dropout", 0.1),
        time_shift=cfg_int(cfg, "augment.time_shift", 1),
    )

    npy_root = resolve_cfg_path(cfg_str(cfg, "data.npy_root_path", "./2s/2s"))
    samples = collect_samples(npy_root)
    if not samples:
        raise RuntimeError(f"No npy samples found in {npy_root}")

    test_ratio = cfg_float(cfg, "data.test_ratio", 0.2)
    seed = cfg_int(cfg, "data.seed", 42)
    train_samples, test_samples = split_samples(samples, test_ratio, seed)

    labels = sorted({lb for _, lb in samples})
    label_to_idx = {lb: i for i, lb in enumerate(labels)}
    idx_to_label = {i: lb for lb, i in label_to_idx.items()}

    train_idx = [(p, label_to_idx[lb]) for p, lb in train_samples]
    test_idx = [(p, label_to_idx[lb]) for p, lb in test_samples]

    train_ds = NpyGaitDataset(train_idx, fgst_cfg, is_train=True, augment_cfg=aug_cfg)
    test_ds = NpyGaitDataset(test_idx, fgst_cfg, is_train=False, augment_cfg=None)
    train_loader = DataLoader(train_ds, batch_size=fgst_cfg.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=fgst_cfg.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(fgst_cfg, num_classes=len(labels)).to(device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=fgst_cfg.learning_rate,
        weight_decay=fgst_cfg.weight_decay,
        betas=(train_cfg.adam_beta1, train_cfg.adam_beta2),
    )
    scheduler = None
    if train_cfg.lr_decay_milestones:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            opt,
            milestones=train_cfg.lr_decay_milestones,
            gamma=train_cfg.lr_decay_gamma,
        )

    # 可选类别重加权：按类别频次反比加权，缓解类别不均衡
    class_weight = None
    if train_cfg.use_class_weight:
        counts = np.zeros(len(labels), dtype=np.float32)
        for _, lb in train_idx:
            counts[lb] += 1.0
        counts = np.maximum(counts, 1.0)
        weight = counts.sum() / (len(labels) * counts)
        class_weight = torch.tensor(weight, dtype=torch.float32, device=device)

    # 以验证集 macro-F1 选择最优检查点
    best_epoch = -1
    best_macro_f1 = -1.0
    best_state = None

    for ep in range(fgst_cfg.epochs):
        model.train()
        ep_ce_loss = 0.0
        ep_tri_loss = 0.0
        ep_part_tri_loss = 0.0
        ep_total_loss = 0.0
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits, embedding, part_stack = model(x, return_embedding=True)
            ce_loss = F.cross_entropy(logits, y, weight=class_weight)
            global_tri_loss = (
                batch_hard_triplet_loss(embedding, y, train_cfg.triplet_margin)
                if train_cfg.use_triplet_loss
                else logits.new_zeros(())
            )
            # separate triplet: 对每个 part 的 embedding 分别施加度量约束，再做均值
            if train_cfg.use_triplet_loss and train_cfg.use_part_triplet:
                part_tri_losses = []
                for k in range(part_stack.size(1)):
                    part_tri_losses.append(
                        batch_hard_triplet_loss(part_stack[:, k, :], y, train_cfg.triplet_margin)
                    )
                part_tri_loss = torch.stack(part_tri_losses).mean() if part_tri_losses else logits.new_zeros(())
            else:
                part_tri_loss = logits.new_zeros(())

            loss = (
                ce_loss
                + train_cfg.triplet_weight * global_tri_loss
                + train_cfg.part_triplet_weight * part_tri_loss
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_ce_loss += float(ce_loss.detach().cpu().item())
            ep_tri_loss += float(global_tri_loss.detach().cpu().item())
            ep_part_tri_loss += float(part_tri_loss.detach().cpu().item())
            ep_total_loss += float(loss.detach().cpu().item())
        if scheduler is not None:
            scheduler.step()

        # 每个 epoch 在验证集上评估，用于选择最佳模型
        _, val_pred_idx = run_predict_epoch(model, test_loader, device)
        val_true_idx = [y for _, y in test_idx]
        val_true = [idx_to_label[x] for x in val_true_idx]
        val_pred = [idx_to_label[x] for x in val_pred_idx]
        _, val_mf1, _ = compute_metrics(val_true, val_pred)
        if val_mf1 > best_macro_f1:
            best_macro_f1 = val_mf1
            best_epoch = ep + 1
            best_state = copy.deepcopy(model.state_dict())

        if ep == 0 or (ep + 1) % 10 == 0:
            current_lr = opt.param_groups[0]["lr"]
            print(
                f"[fgst_pt] epoch {ep+1}/{fgst_cfg.epochs} "
                f"ce_loss={ep_ce_loss:.6f} global_tri={ep_tri_loss:.6f} part_tri={ep_part_tri_loss:.6f} "
                f"total_loss={ep_total_loss:.6f} "
                f"val_macro_f1={val_mf1:.6f} lr={current_lr:.6g}"
            )

    if best_state is not None and train_cfg.save_best_only:
        model.load_state_dict(best_state)
        print(f"[fgst_pt] best checkpoint loaded: epoch={best_epoch} macro_f1={best_macro_f1:.6f}")

    train_true, train_pred = run_predict_epoch(model, train_loader, device)
    test_true, test_pred = run_predict_epoch(model, test_loader, device)
    train_true = [idx_to_label[x] for x in train_true]
    train_pred = [idx_to_label[x] for x in train_pred]
    test_true = [idx_to_label[x] for x in test_true]
    test_pred = [idx_to_label[x] for x in test_pred]

    print(f"train samples: {len(train_true)}, test samples: {len(test_true)}")
    tr_acc, tr_mf1, tr_per = compute_metrics(train_true, train_pred)
    print("[train]")
    print_metrics(tr_acc, tr_mf1, tr_per)
    te_acc, te_mf1, te_per = compute_metrics(test_true, test_pred)
    print("[test]")
    print_metrics(te_acc, te_mf1, te_per)

    model_path = resolve_cfg_path(cfg_str(cfg, "model.path", "./model/mmwave_fgst_2s.pt"))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "labels": labels,
            "fgst_cfg": fgst_cfg.__dict__,
            "train_cfg": train_cfg.__dict__,
            "augment_cfg": aug_cfg.__dict__,
            "best_epoch": best_epoch,
            "best_macro_f1": best_macro_f1,
        },
        model_path,
    )
    report_path = resolve_cfg_path(cfg_str(cfg, "report.path", "./model/metrics_fgst_2s.csv"))
    save_metrics_csv(report_path, te_acc, te_mf1, te_per)
    print(f"metrics saved: {report_path}")
    print(f"model saved: {model_path}")


def load_model_for_eval(cfg: Dict[str, str], cfg_path: Path):
    model_path = resolve_cfg_path(cfg_str(cfg, "model.path", "./model/mmwave_fgst_2s.pt"))
    ckpt = torch.load(model_path, map_location="cpu")
    labels = ckpt["labels"]
    c = dict(ckpt["fgst_cfg"])
    # 兼容旧checkpoint：缺少新字段时补默认值
    c.setdefault("temporal_dilations", [1, 2, 3, 4])
    c.setdefault("negative_slope", 0.1)
    c.setdefault("dropout", 0.2)
    fgst_cfg = FGSTConfig(**c)
    model = build_model(fgst_cfg, num_classes=len(labels))
    model.load_state_dict(ckpt["state_dict"])
    return model, labels, fgst_cfg, model_path


def run_eval(cfg: Dict[str, str], cfg_path: Path):
    model, labels, fgst_cfg, _ = load_model_for_eval(cfg, cfg_path)
    label_to_idx = {lb: i for i, lb in enumerate(labels)}

    # eval 使用全量样本计算总体指标
    npy_root = resolve_cfg_path(cfg_str(cfg, "data.npy_root_path", "./2s/2s"))
    samples = collect_samples(npy_root)
    ds = NpyGaitDataset([(p, label_to_idx[lb]) for p, lb in samples], fgst_cfg)
    loader = DataLoader(ds, batch_size=cfg_int(cfg, "fgst.batch_size", 16), shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    y_true, y_pred = run_predict_epoch(model, loader, device)
    idx_to_label = {i: lb for i, lb in enumerate(labels)}
    y_true = [idx_to_label[x] for x in y_true]
    y_pred = [idx_to_label[x] for x in y_pred]

    acc, mf1, per = compute_metrics(y_true, y_pred)
    print(f"eval samples: {len(y_true)}")
    print_metrics(acc, mf1, per)
    report_path = resolve_cfg_path(cfg_str(cfg, "report.path", "./model/metrics_fgst_2s.csv"))
    save_metrics_csv(report_path, acc, mf1, per)
    print(f"metrics saved: {report_path}")


def run_predict(cfg: Dict[str, str], cfg_path: Path, sample_npy: Path):
    model, labels, fgst_cfg, _ = load_model_for_eval(cfg, cfg_path)
    label_to_idx = {lb: i for i, lb in enumerate(labels)}
    # predict 只需要输入样本，这里给一个占位标签以复用数据读取逻辑
    ds = NpyGaitDataset([(sample_npy.resolve(), label_to_idx[labels[0]])], fgst_cfg)
    x, _, _ = ds[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    with torch.no_grad():
        logits = model(x.unsqueeze(0).to(device))
        pred_idx = int(torch.argmax(logits, dim=1).item())
    pred_label = labels[pred_idx]
    print(f"predict label: {pred_label}")

    out_path = resolve_cfg_path(cfg_str(cfg, "predict.output_path", "./model/predict_result_2s.csv"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_npy", "predict_label"])
        w.writerow([str(sample_npy.resolve()), pred_label])
    print(f"predict saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "eval", "predict"])
    parser.add_argument("config_path")
    parser.add_argument("sample_npy", nargs="?")
    args = parser.parse_args()

    cfg_path = Path(args.config_path).resolve()
    cfg = load_cfg(cfg_path)
    if args.mode == "train":
        run_train(cfg, cfg_path)
    elif args.mode == "eval":
        run_eval(cfg, cfg_path)
    else:
        if not args.sample_npy:
            raise SystemExit("predict mode requires sample_npy")
        run_predict(cfg, cfg_path, Path(args.sample_npy))


if __name__ == "__main__":
    main()
