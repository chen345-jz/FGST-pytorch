import argparse
import math
import os
import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm


# =========================
# Utils
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f"[fgst] global random seed set to {seed}")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


def _resample_axis(arr: np.ndarray, size: int, axis: int, rng: np.random.Generator | None = None) -> np.ndarray:
    cur = arr.shape[axis]
    if cur == size:
        return arr
    if cur == 0:
        shape = list(arr.shape)
        shape[axis] = size
        return np.zeros(shape, dtype=np.float32)
    if cur > size:
        start = (cur - size) // 2 if axis == 0 else 0
        indices = np.arange(start, start + size) if axis == 0 else np.sort((rng or np.random.default_rng()).choice(cur, size, replace=False))
    else:
        base = np.arange(cur)
        pad = (rng or np.random.default_rng()).choice(cur, size - cur, replace=True)
        indices = np.concatenate([base, pad])
    return np.take(arr, indices, axis=axis)


def standardize_radar_sequence(arr, frame_count, num_points, feature_dim=4, random_points=False):
    """Return [T,N,4] as x,y,z,velocity and apply per-frame coordinate normalization."""
    out = np.zeros((frame_count, num_points, feature_dim), dtype=np.float32)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3:
        return out

    rng = np.random.default_rng() if random_points else None
    arr = _resample_axis(arr, frame_count, axis=0, rng=rng)
    arr = _resample_axis(arr, num_points, axis=1, rng=rng)

    if arr.shape[2] >= feature_dim:
        arr = arr[:, :, :feature_dim]
    else:
        pad = np.zeros((arr.shape[0], arr.shape[1], feature_dim - arr.shape[2]), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=2)

    valid = np.abs(arr).sum(axis=2) > 0
    xyz = arr[:, :, :3]
    for t in range(arr.shape[0]):
        m = valid[t]
        if m.any():
            centroid = xyz[t, m].mean(axis=0, keepdims=True)
            xyz[t] = xyz[t] - centroid
        order = np.argsort(-xyz[t, :, 2])
        arr[t] = arr[t, order]

    out[:] = arr
    return out


class RadarDataset(Dataset):
    def __init__(
        self,
        radar_base_dir,
        person_ids,
        session_ranges=None,
        frame_count=20,
        num_points=80,
        feature_dim=4,
        random_points=False,
    ):
        self.radar_base_dir = radar_base_dir
        self.person_ids = sorted(person_ids)
        self.session_ranges = session_ranges if session_ranges is not None else range(50)
        self.frame_count = frame_count
        self.num_points = num_points
        self.feature_dim = feature_dim
        self.random_points = random_points
        self.id_to_label = {pid: i for i, pid in enumerate(self.person_ids)}
        self.samples = []
        for pid in self.person_ids:
            d = os.path.join(self.radar_base_dir, f"p_{pid}")
            if not os.path.isdir(d):
                continue
            for s in self.session_ranges:
                p = os.path.join(d, f"{s}.npy")
                if os.path.exists(p):
                    self.samples.append((pid, s, self.id_to_label[pid]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pid, sess, label = self.samples[idx]
        p = os.path.join(self.radar_base_dir, f"p_{pid}", f"{sess}.npy")
        x = np.zeros((self.frame_count, self.num_points, self.feature_dim), dtype=np.float32)
        try:
            arr = np.load(p)
            x = standardize_radar_sequence(
                arr,
                self.frame_count,
                self.num_points,
                feature_dim=self.feature_dim,
                random_points=self.random_points,
            )
        except Exception:
            pass
        return torch.from_numpy(x), label


# =========================
# Fine-Grained Spatial-Temporal Gait Network
# =========================
class PointMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        b, n, _ = x.shape
        y = self.fc(x)
        y = self.bn(y.reshape(b * n, -1)).reshape(b, n, -1)
        return F.leaky_relu(y, negative_slope=0.1)


class DSFEModule(nn.Module):
    """Dual-stream feature extraction: spatial stream + Doppler-velocity stream."""

    def __init__(self, out_channels=512):
        super().__init__()
        self.vel1 = PointMLP(1, 32)
        self.vel2 = PointMLP(32, 64)
        self.vel3 = PointMLP(64, 128)

        self.spa1 = PointMLP(3, 32)
        self.spa2 = PointMLP(32 + 32, 64)
        self.spa3 = PointMLP(64 + 64, 128)

        self.fuse = PointMLP(128 + 128, out_channels)

    def forward(self, points):
        spatial = points[..., :3]
        velocity = points[..., 3:4]

        v1 = self.vel1(velocity)
        s1 = self.spa1(spatial)
        v2 = self.vel2(v1)
        s2 = self.spa2(torch.cat([s1, v1], dim=-1))
        v3 = self.vel3(v2)
        s3 = self.spa3(torch.cat([s2, v2], dim=-1))
        return self.fuse(torch.cat([s3, v3], dim=-1))


class TimeDistributedDSFE(nn.Module):
    def __init__(self, out_channels=512):
        super().__init__()
        self.dsfe = DSFEModule(out_channels=out_channels)

    def forward(self, x):
        b, t, n, c = x.shape
        y = self.dsfe(x.reshape(b * t, n, c))
        return y.reshape(b, t, n, -1)


class PGBPModule(nn.Module):
    """Probability-guided body-part partition with max+average region pooling."""

    def __init__(self, num_parts=16, probabilities: Sequence[float] | None = None):
        super().__init__()
        if probabilities is None:
            probabilities = [1.0 / num_parts] * num_parts
        prob = torch.as_tensor(probabilities, dtype=torch.float32)
        prob = prob.clamp(min=1e-6)
        prob = prob / prob.sum()
        self.num_parts = int(num_parts)
        self.register_buffer("probabilities", prob)

    def _boundaries(self, num_points: int) -> List[Tuple[int, int]]:
        if num_points < self.num_parts:
            raise ValueError(f"num_points ({num_points}) must be >= num_parts ({self.num_parts}) for PGBP")
        remaining = num_points - self.num_parts
        raw_extra = self.probabilities * remaining
        counts = torch.floor(raw_extra).long() + 1
        deficit = num_points - int(counts.sum().item())
        if deficit > 0:
            frac = raw_extra - torch.floor(raw_extra)
            order = torch.argsort(frac, descending=True)
            counts[order[:deficit]] += 1
        elif deficit < 0:
            removable = counts - 1
            order = torch.argsort(raw_extra - torch.floor(raw_extra), descending=False)
            need = -deficit
            for idx in order.tolist():
                take = min(int(removable[idx].item()), need)
                counts[idx] -= take
                need -= take
                if need == 0:
                    break
        ends = torch.cumsum(counts, dim=0).long().tolist()
        bounds = []
        start = 0
        for i, end in enumerate(ends):
            if i == self.num_parts - 1:
                end = num_points
            end = min(int(end), num_points)
            bounds.append((start, end))
            start = end
        return bounds

    def forward(self, point_features):
        parts = []
        n = point_features.shape[2]
        for start, end in self._boundaries(n):
            region = point_features[:, :, start:end, :]
            pooled = region.amax(dim=2) + region.mean(dim=2)
            parts.append(pooled)
        return torch.stack(parts, dim=2)


class TemporalConvBranch(nn.Module):
    def __init__(self, channels: int, out_channels: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(
            channels,
            out_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        return F.leaky_relu(y, negative_slope=0.1)


class LGTEModule(nn.Module):
    """Independent local-global temporal extractor for one body part."""

    def __init__(self, channels=512, branch_channels=128, dilations=(1, 2, 3, 4)):
        super().__init__()
        self.branches = nn.ModuleList([TemporalConvBranch(channels, branch_channels, d) for d in dilations])
        out_channels = branch_channels * len(dilations)
        self.attention = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.1),
            nn.Conv1d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, part_sequence):
        x = part_sequence.transpose(1, 2)
        local = torch.cat([branch(x) for branch in self.branches], dim=1)
        weight = self.attention(local)
        pooled = (local * weight).sum(dim=2) / weight.sum(dim=2).clamp(min=1e-6)
        return pooled


class FineGrainedSpatialTemporalNet(nn.Module):
    def __init__(self, num_classes, num_parts=16, feature_dim=256, pgbp_probabilities=None):
        super().__init__()
        self.num_parts = num_parts
        self.dsfe = TimeDistributedDSFE(out_channels=512)
        self.pgbp = PGBPModule(num_parts=num_parts, probabilities=pgbp_probabilities)
        self.lgte = nn.ModuleList([LGTEModule(channels=512, branch_channels=128) for _ in range(num_parts)])
        self.part_fc = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(512, feature_dim, bias=False),
                    nn.BatchNorm1d(feature_dim),
                    nn.LeakyReLU(0.1),
                )
                for _ in range(num_parts)
            ]
        )
        self.classifier = nn.Linear(feature_dim * num_parts, num_classes)
        self.norm = nn.LayerNorm(feature_dim * num_parts)

    def forward(self, radar_data, is_training=True, return_parts=False):
        point_features = self.dsfe(radar_data)
        part_sequences = self.pgbp(point_features)
        part_features = []
        for k in range(self.num_parts):
            temporal = self.lgte[k](part_sequences[:, :, k, :])
            part_features.append(self.part_fc[k](temporal))
        parts = torch.stack(part_features, dim=1)
        feature = self.norm(parts.flatten(1))
        logits = self.classifier(feature)
        if return_parts:
            return feature, logits, parts
        if is_training:
            return feature, logits
        return feature


class FGSTReID(FineGrainedSpatialTemporalNet):
    pass


# =========================
# Losses
# =========================
class BatchAllTripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin

    def forward(self, features, labels):
        if features is None or labels is None or features.size(0) < 2:
            return torch.tensor(0.0, device=features.device)
        dist = torch.cdist(features, features, p=2)
        labels = labels.view(-1)
        pos = labels[:, None].eq(labels[None, :])
        neg = ~pos
        pos.fill_diagonal_(False)
        if not (pos.any() and neg.any()):
            return torch.tensor(0.0, device=features.device)
        triplets = dist[:, :, None] - dist[:, None, :] + self.margin
        mask = pos[:, :, None] & neg[:, None, :]
        return F.relu(triplets[mask]).mean()


class FGSTLoss(nn.Module):
    def __init__(self, margin=0.2, ce_weight=0.0, part_weight=0.5):
        super().__init__()
        self.trip = BatchAllTripletLoss(margin)
        self.ce = nn.CrossEntropyLoss()
        self.ce_weight = ce_weight
        self.part_weight = part_weight

    def forward(self, feat, logits, y, parts=None):
        l_trip = self.trip(feat, y)
        l_part = torch.tensor(0.0, device=feat.device)
        if parts is not None and self.part_weight > 0:
            part_losses = [self.trip(parts[:, k, :], y) for k in range(parts.shape[1])]
            l_part = torch.stack(part_losses).mean()
        l_ce = self.ce(logits, y) if self.ce_weight > 0 else torch.tensor(0.0, device=feat.device)
        total = l_trip + self.part_weight * l_part + self.ce_weight * l_ce
        return total, {
            "total": total.item(),
            "Trip": l_trip.item(),
            "Trip_P": l_part.item(),
            "CE": l_ce.item(),
        }


class BalancedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, n_classes, n_samples):
        self.dataset, self.n_classes, self.n_samples = dataset, n_classes, n_samples
        self.batch_size = n_classes * n_samples
        parent = dataset.dataset if isinstance(dataset, Subset) else dataset
        indices = dataset.indices if isinstance(dataset, Subset) else range(len(parent.samples))
        self.id_to_indices = {}
        for idx in indices:
            pid = parent.samples[idx][0]
            self.id_to_indices.setdefault(pid, []).append(idx)
        self.ids = list(self.id_to_indices.keys())
        self.num_batches = len(list(indices)) // self.batch_size if self.batch_size > 0 else 0

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = []
            available = [pid for pid in self.ids if self.id_to_indices.get(pid)]
            if len(available) < self.n_classes:
                continue
            selected = np.random.choice(available, self.n_classes, replace=(len(available) < self.n_classes))
            for pid in selected:
                idxs = self.id_to_indices[pid]
                batch.extend(np.random.choice(idxs, self.n_samples, replace=(len(idxs) < self.n_samples)))
            if batch:
                random.shuffle(batch)
                yield batch

    def __len__(self):
        return self.num_batches


# =========================
# Eval helpers
# =========================
def evaluate_reid(model, gallery_loader, query_loader, device, title="ReID"):
    model.eval()
    g_feat, g_lab, q_feat, q_lab = [], [], [], []
    with torch.no_grad():
        for x, y in tqdm(gallery_loader, desc="Extract Gallery", leave=False):
            feat = model(radar_data=x.to(device), is_training=False)
            g_feat.append(F.normalize(feat, p=2, dim=1).cpu())
            g_lab.append(y.cpu())
        for x, y in tqdm(query_loader, desc="Extract Query", leave=False):
            feat = model(radar_data=x.to(device), is_training=False)
            q_feat.append(F.normalize(feat, p=2, dim=1).cpu())
            q_lab.append(y.cpu())

    if not g_feat or not q_feat:
        return 0.0, 0.0, 0.0, 0.0

    g_feat = torch.cat(g_feat)
    g_lab = torch.cat(g_lab)
    q_feat = torch.cat(q_feat)
    q_lab = torch.cat(q_lab)
    if q_feat.size(0) == 0 or g_feat.size(0) == 0:
        return 0.0, 0.0, 0.0, 0.0

    sim = torch.mm(q_feat, g_feat.t())
    num_q = q_lab.size(0)
    cmc = torch.zeros(10)
    aps = []

    for i in range(num_q):
        rel = g_lab == q_lab[i]
        if not rel.any():
            continue
        _, order = torch.sort(sim[i], descending=True)
        sorted_rel = rel[order]
        pos = torch.nonzero(sorted_rel, as_tuple=False).squeeze(1)
        if pos.numel() > 0 and pos[0].item() < 10:
            cmc[pos[0].item() :] += 1
        aps.append(average_precision_score(rel.numpy(), sim[i].numpy()))

    mAP = float(np.mean(aps) * 100.0) if aps else 0.0
    r1, r3, r5 = (cmc / max(num_q, 1) * 100.0)[[0, 2, 4]]
    print(f"[{title}] Rank-1={float(r1):.2f}% | mAP={mAP:.2f}% (R3={float(r3):.2f}%, R5={float(r5):.2f}%)")
    return float(r1), float(r3), float(r5), mAP


def build_reid_loaders_for_ids(dataset, person_ids, batch_size, split_ratio=0.7, seed=42):
    id_set = set(person_ids)
    indices = [i for i, (pid, _, _) in enumerate(dataset.samples) if pid in id_set]
    by_pid = {}
    for idx in indices:
        pid = dataset.samples[idx][0]
        by_pid.setdefault(pid, []).append(idx)

    g_idx, q_idx = [], []
    rng = random.Random(seed)
    for pid, idxs in by_pid.items():
        idxs = idxs[:]
        rng.shuffle(idxs)
        sp = max(1, int(len(idxs) * split_ratio))
        g_idx.extend(idxs[:sp])
        q_idx.extend(idxs[sp:])
    if not q_idx and g_idx:
        q_idx.append(g_idx.pop())
    if not g_idx or not q_idx:
        return None, None

    gallery_loader = DataLoader(Subset(dataset, g_idx), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    query_loader = DataLoader(Subset(dataset, q_idx), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return gallery_loader, query_loader


def plot_training_progress(epochs, r1s, maps, filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, r1s, "o-", label="test Rank-1")
    ax.plot(epochs, maps, "s-", label="test mAP")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score (%)")
    ax.legend()
    ax.grid(True)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)


def estimate_pgbp_probabilities(dataset: Dataset, num_parts: int, max_samples: int = 256) -> List[float]:
    counts = np.ones(num_parts, dtype=np.float64)
    sample_count = min(len(dataset), max_samples)
    if sample_count <= 0:
        return (counts / counts.sum()).tolist()
    for idx in range(sample_count):
        x, _ = dataset[idx]
        z = x[..., 2].numpy()
        valid = np.abs(x.numpy()).sum(axis=2) > 0
        if not valid.any():
            continue
        z_valid = z[valid]
        z_min, z_max = float(z_valid.min()), float(z_valid.max())
        span = max(z_max - z_min, 1e-6)
        bins = np.floor((z_max - z_valid) / span * num_parts).astype(np.int64)
        bins = np.clip(bins, 0, num_parts - 1)
        counts += np.bincount(bins, minlength=num_parts)
    prob = counts / counts.sum()
    return prob.tolist()


# =========================
# Train / Eval / Predict
# =========================
@dataclass
class HParams:
    radar_base_dir: str
    checkpoint_dir: str
    seed: int = 42
    split_ratio: float = 0.7
    num_epochs: int = 50
    batch_size: int = 16
    accumulation_steps: int = 2
    learning_rate: float = 1e-4
    eval_interval: int = 1
    early_stopping_patience: int = 20
    frame_num: int = 20
    num_points: int = 80
    feat_dim: int = 256
    num_parts: int = 16
    ce_weight: float = 0.0
    part_weight: float = 0.5
    margin: float = 0.2
    weight_decay: float = 1e-4
    clip_grad_norm: float = 1.0
    feature_dim: int = 4


def discover_person_ids(radar_root):
    out = []
    if os.path.isdir(radar_root):
        for n in os.listdir(radar_root):
            if n.startswith("p_"):
                try:
                    out.append(int(n.split("_", 1)[1]))
                except Exception:
                    pass
    return sorted(out)


def split_person_ids(all_ids, split_ratio, seed):
    rng = random.Random(seed)
    ids = all_ids[:]
    rng.shuffle(ids)
    sp = max(1, min(len(ids) - 1, int(round(len(ids) * split_ratio))))
    return sorted(ids[:sp]), sorted(ids[sp:])


def make_model(num_classes, hp: HParams, pgbp_probabilities=None, device=None):
    model = FGSTReID(
        num_classes=num_classes,
        num_parts=hp.num_parts,
        feature_dim=hp.feat_dim,
        pgbp_probabilities=pgbp_probabilities,
    )
    if device is not None:
        model = model.to(device)
    return model


def train_and_report(hp: HParams, checkpoint_path_override=""):
    set_seed(hp.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_ids = discover_person_ids(hp.radar_base_dir)
    if len(all_ids) < 2:
        raise RuntimeError(f"No valid p_<id> folders under {hp.radar_base_dir}")
    train_ids, test_ids = split_person_ids(all_ids, hp.split_ratio, hp.seed)
    print(f"[fgst] train IDs ({len(train_ids)}): {train_ids}")
    print(f"[fgst] test IDs  ({len(test_ids)}): {test_ids}")

    ds = RadarDataset(hp.radar_base_dir, all_ids, frame_count=hp.frame_num, num_points=hp.num_points, feature_dim=hp.feature_dim)
    train_ds = RadarDataset(
        hp.radar_base_dir,
        train_ids,
        frame_count=hp.frame_num,
        num_points=hp.num_points,
        feature_dim=hp.feature_dim,
        random_points=True,
    )
    pgbp_prob = estimate_pgbp_probabilities(train_ds, hp.num_parts)
    print(f"[fgst] PGBP probabilities: {[round(p, 4) for p in pgbp_prob]}")

    n_classes_per_batch = min(2, len(train_ids))
    n_samples_per_class = max(2, hp.batch_size // max(n_classes_per_batch, 1))
    sampler = BalancedBatchSampler(train_ds, n_classes=n_classes_per_batch, n_samples=n_samples_per_class)

    g = torch.Generator()
    g.manual_seed(hp.seed)
    train_loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    model = make_model(len(train_ids), hp, pgbp_probabilities=pgbp_prob, device=device)
    criterion = FGSTLoss(margin=hp.margin, ce_weight=hp.ce_weight, part_weight=hp.part_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp.learning_rate, weight_decay=hp.weight_decay, betas=(0.9, 0.999))

    warmup_epochs = max(1, int(hp.num_epochs * 0.1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda e: e / warmup_epochs
        if e < warmup_epochs
        else 0.5 * (1 + math.cos(math.pi * (e - warmup_epochs) / max(hp.num_epochs - warmup_epochs, 1))),
    )

    os.makedirs(hp.checkpoint_dir, exist_ok=True)
    best_path = checkpoint_path_override or os.path.join(hp.checkpoint_dir, "best_model_fgst.pth")

    ep_rec, r1s, maps = [], [], []
    best_map, best_r1, no_up = 0.0, 0.0, 0

    print("[fgst] start training")
    for epoch in range(hp.num_epochs):
        model.train()
        pb = tqdm(train_loader, desc=f"[fgst] epoch {epoch + 1}/{hp.num_epochs}", leave=False)
        optimizer.zero_grad()
        for bi, (x, y) in enumerate(pb):
            x = x.to(device)
            y = y.to(device)
            feat, logits, parts = model(radar_data=x, is_training=True, return_parts=True)
            loss, ld = criterion(feat, logits, y, parts=parts)
            (loss / hp.accumulation_steps).backward()
            if (bi + 1) % hp.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), hp.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
            pb.set_postfix(loss=f"{ld['total']:.3f}", trip=f"{ld['Trip']:.3f}", part=f"{ld['Trip_P']:.3f}")
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        if (epoch + 1) % hp.eval_interval == 0 or epoch == hp.num_epochs - 1:
            test_g, test_q = build_reid_loaders_for_ids(ds, test_ids, hp.batch_size, split_ratio=0.7, seed=hp.seed)
            if test_g is not None:
                r1, _, _, m = evaluate_reid(model, test_g, test_q, device, title=f"test@epoch{epoch + 1}")
                ep_rec.append(epoch + 1)
                r1s.append(r1)
                maps.append(m)
                plot_training_progress(ep_rec, r1s, maps, os.path.join(hp.checkpoint_dir, "training_progress_fgst.png"))
                if m > best_map:
                    best_map, best_r1, no_up = m, r1, 0
                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "pgbp_probabilities": pgbp_prob,
                            "hparams": hp.__dict__,
                        },
                        best_path,
                    )
                    print(f"[fgst] new best saved: mAP={m:.2f} Rank-1={r1:.2f}")
                else:
                    no_up += 1
                    print(f"[fgst] no improve: best mAP={best_map:.2f}, patience={no_up}/{hp.early_stopping_patience}")
                if no_up >= hp.early_stopping_patience:
                    print("[fgst] early stopping triggered")
                    break

    print(f"[fgst] training done. best test Rank-1={best_r1:.2f}% mAP={best_map:.2f}%")

    if os.path.exists(best_path):
        model = load_model_for_infer(hp, len(train_ids), best_path, device)
    train_g, train_q = build_reid_loaders_for_ids(ds, train_ids, hp.batch_size, split_ratio=0.7, seed=hp.seed)
    test_g, test_q = build_reid_loaders_for_ids(ds, test_ids, hp.batch_size, split_ratio=0.7, seed=hp.seed)

    tr_r1 = tr_map = te_r1 = te_map = 0.0
    if train_g is not None:
        tr_r1, _, _, tr_map = evaluate_reid(model, train_g, train_q, device, title="train")
    if test_g is not None:
        te_r1, _, _, te_map = evaluate_reid(model, test_g, test_q, device, title="test")

    print("[fgst] final summary")
    print(f"[fgst] train  Rank-1={tr_r1:.2f}% mAP={tr_map:.2f}%")
    print(f"[fgst] test   Rank-1={te_r1:.2f}% mAP={te_map:.2f}%")
    print(f"[fgst] best checkpoint: {best_path}")

    return {
        "train_rank1": tr_r1,
        "train_map": tr_map,
        "test_rank1": te_r1,
        "test_map": te_map,
        "checkpoint": best_path,
        "train_ids": train_ids,
        "test_ids": test_ids,
    }


def _load_checkpoint(checkpoint_path, device):
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        return state["model_state"], state.get("pgbp_probabilities")
    return state, None


def load_model_for_infer(hp: HParams, num_classes, checkpoint_path, device):
    state, pgbp_prob = _load_checkpoint(checkpoint_path, device)
    m = make_model(hp=hp, num_classes=num_classes, pgbp_probabilities=pgbp_prob, device=device)
    cur = m.state_dict()
    bad = [k for k, v in state.items() if k in cur and cur[k].shape != v.shape]
    for k in bad:
        del state[k]
    m.load_state_dict(state, strict=False)
    m.eval()
    return m


def collect_gallery_embeddings(model, loader, device):
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            f = model(radar_data=x, is_training=False)
            feats.append(F.normalize(f, dim=1).cpu())
            labels.append(y.cpu())
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


def infer_single_sample(model, dataset, gallery_emb, gallery_labels, hp: HParams, device, sample_npy):
    arr = np.load(sample_npy)
    arr = standardize_radar_sequence(arr, hp.frame_num, hp.num_points, feature_dim=hp.feature_dim)
    x = torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q = model(radar_data=x, is_training=False)
        q = F.normalize(q, dim=1).cpu()
    sim = torch.matmul(q, gallery_emb.t()).squeeze(0)
    topk = torch.topk(sim, k=min(5, sim.numel()))
    top_ids = []
    for idx in topk.indices.tolist():
        local_lab = int(gallery_labels[idx].item())
        top_ids.append(dataset.person_ids[local_lab])
    return top_ids[0], top_ids


def run_eval(hp: HParams, checkpoint_path):
    set_seed(hp.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_ids = discover_person_ids(hp.radar_base_dir)
    train_ids, test_ids = split_person_ids(all_ids, hp.split_ratio, hp.seed)

    ds = RadarDataset(hp.radar_base_dir, all_ids, frame_count=hp.frame_num, num_points=hp.num_points, feature_dim=hp.feature_dim)
    model = load_model_for_infer(hp, len(train_ids), checkpoint_path, device)

    test_g, test_q = build_reid_loaders_for_ids(ds, test_ids, hp.batch_size, split_ratio=0.7, seed=hp.seed)
    if test_g is None:
        raise RuntimeError("empty test split for eval")
    r1, r3, r5, m = evaluate_reid(model, test_g, test_q, device, title="eval(test)")
    print(f"[fgst] eval summary: Rank-1={r1:.2f}% mAP={m:.2f}% R3={r3:.2f}% R5={r5:.2f}%")


def run_predict(hp: HParams, checkpoint_path, sample_npy, gallery_scope="all"):
    set_seed(hp.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_ids = discover_person_ids(hp.radar_base_dir)
    train_ids, test_ids = split_person_ids(all_ids, hp.split_ratio, hp.seed)
    gallery_ids = train_ids if gallery_scope == "train" else test_ids if gallery_scope == "test" else all_ids

    ds = RadarDataset(hp.radar_base_dir, gallery_ids, frame_count=hp.frame_num, num_points=hp.num_points, feature_dim=hp.feature_dim)
    g_loader, _ = build_reid_loaders_for_ids(ds, gallery_ids, hp.batch_size, split_ratio=0.7, seed=hp.seed)
    if g_loader is None:
        raise RuntimeError("empty gallery for predict")

    model = load_model_for_infer(hp, len(train_ids), checkpoint_path, device)
    g_emb, g_lab = collect_gallery_embeddings(model, g_loader, device)

    pred, top5 = infer_single_sample(model, ds, g_emb, g_lab, hp, device, sample_npy)
    print(f"[fgst] predict sample: {sample_npy}")
    print(f"[fgst] predict top1 person_id: {pred}")
    print(f"[fgst] predict top5 person_id: {top5}")

    gt = None
    parent = os.path.basename(os.path.dirname(sample_npy))
    if parent.startswith("p_"):
        try:
            gt = int(parent.split("_", 1)[1])
        except Exception:
            gt = None
    if gt is not None:
        print(f"[fgst] ground truth person_id: {gt}")
        print(f"[fgst] correct: {pred == gt}")


def run_predict_batch(hp: HParams, checkpoint_path, num_samples, gallery_scope="all"):
    set_seed(hp.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_ids = discover_person_ids(hp.radar_base_dir)
    train_ids, test_ids = split_person_ids(all_ids, hp.split_ratio, hp.seed)
    gallery_ids = train_ids if gallery_scope == "train" else test_ids if gallery_scope == "test" else all_ids

    sample_paths = []
    for n in os.listdir(hp.radar_base_dir):
        if n.startswith("p_"):
            d = os.path.join(hp.radar_base_dir, n)
            if os.path.isdir(d):
                for f in os.listdir(d):
                    if f.endswith(".npy"):
                        sample_paths.append(os.path.join(d, f))
    sample_paths = sorted(sample_paths)
    if not sample_paths:
        raise RuntimeError("no npy samples")
    k = min(num_samples, len(sample_paths))
    chosen = random.sample(sample_paths, k)

    ds = RadarDataset(hp.radar_base_dir, gallery_ids, frame_count=hp.frame_num, num_points=hp.num_points, feature_dim=hp.feature_dim)
    g_loader, _ = build_reid_loaders_for_ids(ds, gallery_ids, hp.batch_size, split_ratio=0.7, seed=hp.seed)
    model = load_model_for_infer(hp, len(train_ids), checkpoint_path, device)
    g_emb, g_lab = collect_gallery_embeddings(model, g_loader, device)

    correct = 0
    print("[fgst] batch prediction details:")
    for p in chosen:
        pred, _ = infer_single_sample(model, ds, g_emb, g_lab, hp, device, p)
        gt = None
        par = os.path.basename(os.path.dirname(p))
        if par.startswith("p_"):
            try:
                gt = int(par.split("_", 1)[1])
            except Exception:
                gt = None
        ok = gt is not None and pred == gt
        if ok:
            correct += 1
        print(f"{p} | gt={gt} | pred={pred} | correct={ok}")
    acc = correct / k if k > 0 else 0.0
    print("-" * 60)
    print(f"samples: {k}")
    print(f"correct: {correct}")
    print(f"accuracy: {acc:.6f} ({acc * 100:.2f}%)")


def main():
    parser = argparse.ArgumentParser(description="FGST ReID (DSFE + PGBP + LGTE)")
    parser.add_argument("mode", nargs="?", default="train", choices=["train", "eval", "predict", "predict_batch"])
    parser.add_argument("sample_npy", nargs="?", default=None, help="required when mode=predict")
    parser.add_argument("--radar_base_dir", default=r".\2s")
    parser.add_argument("--split_ratio", type=float, default=0.7)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--early_stopping_patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_dir", default="checkpoints_fgst")
    parser.add_argument("--checkpoint_path", default="")
    parser.add_argument("--predict_gallery", choices=["train", "test", "all"], default="all")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--frame_num", type=int, default=20)
    parser.add_argument("--num_points", type=int, default=80)
    parser.add_argument("--num_parts", type=int, default=16)
    parser.add_argument("--feat_dim", type=int, default=256)
    parser.add_argument("--ce_weight", type=float, default=0.0)
    parser.add_argument("--part_weight", type=float, default=0.5)
    parser.add_argument("--margin", type=float, default=0.2)
    args = parser.parse_args()

    hp = HParams(
        radar_base_dir=args.radar_base_dir,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,
        split_ratio=args.split_ratio,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        accumulation_steps=args.accumulation_steps,
        learning_rate=args.learning_rate,
        eval_interval=args.eval_interval,
        early_stopping_patience=args.early_stopping_patience,
        frame_num=args.frame_num,
        num_points=args.num_points,
        num_parts=args.num_parts,
        feat_dim=args.feat_dim,
        ce_weight=args.ce_weight,
        part_weight=args.part_weight,
        margin=args.margin,
    )

    ckpt = args.checkpoint_path or os.path.join(hp.checkpoint_dir, "best_model_fgst.pth")

    if args.mode == "train":
        train_and_report(hp, checkpoint_path_override=ckpt)
        return
    if args.mode == "eval":
        run_eval(hp, checkpoint_path=ckpt)
        return
    if args.mode == "predict":
        if args.sample_npy is None:
            raise RuntimeError("mode=predict requires sample_npy path")
        run_predict(hp, checkpoint_path=ckpt, sample_npy=args.sample_npy, gallery_scope=args.predict_gallery)
        return
    run_predict_batch(hp, checkpoint_path=ckpt, num_samples=args.num_samples, gallery_scope=args.predict_gallery)


if __name__ == "__main__":
    main()
