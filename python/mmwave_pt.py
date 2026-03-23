import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


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


def resolve_cfg_path(raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    return (Path.cwd() / p).resolve()


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


class NpyGaitDataset(Dataset):
    def __init__(self, samples: List[Tuple[Path, int]], fgst_cfg: FGSTConfig):
        self.samples = samples
        self.fgst_cfg = fgst_cfg

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        arr = np.load(path)
        # expected: [T, P, C], C>=4: x,y,z,doppler
        if arr.ndim != 3 or arr.shape[2] < 4:
            raise ValueError(f"Invalid npy shape for {path}: {arr.shape}")

        t_lim = min(self.fgst_cfg.max_frames, arr.shape[0])
        p_lim = min(self.fgst_cfg.max_points_per_frame, arr.shape[1])
        out = np.zeros(
            (self.fgst_cfg.max_frames, self.fgst_cfg.max_points_per_frame, 5),
            dtype=np.float32,
        )
        out[:t_lim, :p_lim, :4] = arr[:t_lim, :p_lim, :4].astype(np.float32)
        out[:t_lim, :p_lim, 4] = 10.0  # snr constant as in C++ pipeline
        return torch.from_numpy(out), int(label), str(path)


class FgstNet(nn.Module):
    def __init__(self, in_dim: int, point_dim: int, temporal_dim: int, num_parts: int, num_classes: int):
        super().__init__()
        self.num_parts = num_parts
        self.point_mlp = nn.Sequential(
            nn.Linear(in_dim, point_dim),
            nn.ReLU(),
            nn.Linear(point_dim, point_dim),
            nn.ReLU(),
        )
        self.part_prob = nn.Linear(point_dim, num_parts)
        self.part_temporal = nn.Sequential(
            nn.Conv1d(point_dim, temporal_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(temporal_dim, temporal_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.global_temporal = nn.Sequential(
            nn.Conv1d(point_dim, temporal_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(temporal_dim, temporal_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear((num_parts + 1) * temporal_dim, temporal_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(temporal_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, P, C]
        b, t, p, c = x.shape
        xp = x.reshape(b * t * p, c)
        point_feat = self.point_mlp(xp)  # [B*T*P, F]
        prob = torch.softmax(self.part_prob(point_feat), dim=1)  # [B*T*P, K]
        point_feat = point_feat.reshape(b, t, p, -1)  # [B, T, P, F]
        prob = prob.reshape(b, t, p, self.num_parts)  # [B, T, P, K]

        global_frame = point_feat.max(dim=2).values  # [B, T, F]
        global_ts = self.global_temporal(global_frame.transpose(1, 2))  # [B, H, T]
        global_vec = global_ts.max(dim=2).values  # [B, H]

        part_vecs = []
        for k in range(self.num_parts):
            wk = prob[:, :, :, k].unsqueeze(-1)  # [B, T, P, 1]
            weighted = point_feat * wk
            pooled = weighted.sum(dim=2) / (wk.sum(dim=2) + 1e-6)  # [B, T, F]
            ts = self.part_temporal(pooled.transpose(1, 2))  # [B, H, T]
            part_vecs.append(ts.max(dim=2).values)  # [B, H]

        part_cat = torch.cat(part_vecs, dim=1)
        fused = torch.cat([part_cat, global_vec], dim=1)
        return self.fusion(fused)


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
    print(f"accuracy: {acc:.6f}")
    print(f"macro_f1: {macro_f1:.6f}")
    for k, p, r, f1 in per:
        print(f"label {k} -> P: {p:.6f}, R: {r:.6f}, F1: {f1:.6f}")


def save_metrics_csv(path: Path, acc: float, macro_f1: float, per):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["accuracy", f"{acc:.6f}"])
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
        point_feature_dim=cfg_int(cfg, "fgst.point_feature_dim", 64),
        temporal_feature_dim=cfg_int(cfg, "fgst.temporal_feature_dim", 128),
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

    train_ds = NpyGaitDataset(train_idx, fgst_cfg)
    test_ds = NpyGaitDataset(test_idx, fgst_cfg)
    train_loader = DataLoader(train_ds, batch_size=fgst_cfg.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=fgst_cfg.batch_size, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(fgst_cfg, num_classes=len(labels)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=fgst_cfg.learning_rate, weight_decay=fgst_cfg.weight_decay)

    for ep in range(fgst_cfg.epochs):
        model.train()
        ep_loss = 0.0
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.detach().cpu().item())
        if ep == 0 or (ep + 1) % 10 == 0:
            print(f"[fgst_pt] epoch {ep+1}/{fgst_cfg.epochs} loss={ep_loss:.6f}")

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
    c = ckpt["fgst_cfg"]
    fgst_cfg = FGSTConfig(**c)
    model = build_model(fgst_cfg, num_classes=len(labels))
    model.load_state_dict(ckpt["state_dict"])
    return model, labels, fgst_cfg, model_path


def run_eval(cfg: Dict[str, str], cfg_path: Path):
    model, labels, fgst_cfg, _ = load_model_for_eval(cfg, cfg_path)
    label_to_idx = {lb: i for i, lb in enumerate(labels)}

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
    # dummy gt label
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
