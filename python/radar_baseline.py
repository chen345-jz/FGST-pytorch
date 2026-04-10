# 1. 【核心思想】建立一个同时使用 Triplet Loss 和 Cross-Entropy Loss 的纯雷达基线。
# 2. 【模型结构】RadarFeatureExtractor 采用 "PointNet -> LSTM -> Mean Pooling" 架构。
# 3. 【训练损失】总损失为 L_Trip_S + L_CE_S，与后续所有跨模态实验的“自身监督”部分保持一致

import torch
import torch.nn as nn
import torch.nn.functional as F
# ... (所有 import 和外部模型定义保持不变) ...
import numpy as np
import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

sys.modules['tkinter'] = None
sys.modules['_tkinter'] = None
from tqdm import tqdm
import random
from sklearn.metrics import average_precision_score
import math
import argparse


# ==============================================================================
# --- 外部模型定义 (自包含) ---
# ==============================================================================
# ... (TimeDistributed, PointNetfeat 定义保持不变) ...
class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__();
        self.module = module;
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2: return self.module(x)
        x_reshape = x.contiguous().view(-1, x.size(-2), x.size(-1));
        y = self.module(x_reshape)
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))
        else:
            y = y.view(-1, x.size(1), y.size(-1))
        return y


class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, in_dim=4):
        super(PointNetfeat, self).__init__();
        self.conv1 = torch.nn.Conv1d(in_dim, 64, 1);
        self.conv2 = torch.nn.Conv1d(64, 128, 1);
        self.conv3 = torch.nn.Conv1d(128, 1024, 1);
        self.bn1 = nn.BatchNorm1d(64);
        self.bn2 = nn.BatchNorm1d(128);
        self.bn3 = nn.BatchNorm1d(1024);
        self.global_feat = global_feat

    def forward(self, x):
        n_pts = x.size()[2];
        x = F.relu(self.bn1(self.conv1(x)));
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)));
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0];
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, None, None
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), None, None


# ==============================================================================
# 0. 数据增强与全局配置 (保持不变)
# ==============================================================================
# ... (set_seed, seed_worker, VideoRadarDataset 保持不变) ...
def set_seed(seed):
    random.seed(seed);
    np.random.seed(seed);
    torch.manual_seed(seed);
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False;
    torch.backends.cudnn.deterministic = True;
    print(f"全局随机种子已设置为 {seed}。")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32;
    np.random.seed(worker_seed + worker_id);
    random.seed(worker_seed + worker_id)


class VideoRadarDataset(Dataset):
    def __init__(self, video_base_dir, radar_base_dir, person_ids, session_ranges=None, frame_count=20, num_points=80,
                 transform=None):
        self.video_base_dir = video_base_dir
        self.radar_base_dir = radar_base_dir
        self.person_ids = sorted(person_ids)
        self.session_ranges = session_ranges if session_ranges is not None else range(50)
        self.frame_count = frame_count
        self.num_points = num_points
        self.transform = transform
        self.samples, self.id_to_label = [], {pid: i for i, pid in enumerate(self.person_ids)}
        for person_id in self.person_ids:
            person_radar_dir = os.path.join(self.radar_base_dir, f"p_{person_id}")
            if not os.path.exists(person_radar_dir): continue
            for session in self.session_ranges:
                radar_path = os.path.join(person_radar_dir, f"{session}.npy")
                if os.path.exists(radar_path):
                    self.samples.append((person_id, session, self.id_to_label[person_id]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        person_id, session, label = self.samples[idx]
        radar_path = os.path.join(self.radar_base_dir, f"p_{person_id}", f"{session}.npy")
        radar_data = np.zeros((self.frame_count, self.num_points, 4), dtype=np.float32)
        try:
            loaded_radar_data = np.load(radar_path).astype(np.float32)
            actual_frames = loaded_radar_data.shape[0]
            if actual_frames > self.frame_count:
                start = (actual_frames - self.frame_count) // 2
                radar_data = loaded_radar_data[start: start + self.frame_count, :, :]
            elif actual_frames < self.frame_count:
                padding_shape = (self.frame_count - actual_frames, self.num_points, 4)
                if actual_frames == 0:
                    padding = np.zeros(padding_shape, dtype=np.float32)
                else:
                    padding = np.repeat(loaded_radar_data[-1:], padding_shape[0], axis=0)
                radar_data = np.concatenate([loaded_radar_data, padding], axis=0)
            else:
                radar_data = loaded_radar_data
        except Exception:
            pass
        dummy_video = torch.zeros(self.frame_count, 3, 256, 128)
        return dummy_video, torch.FloatTensor(radar_data), label


# ==============================================================================
# 1. 核心机制与模型定义
# ==============================================================================
class Sub_PointNet(nn.Module):
    def __init__(self):
        super(Sub_PointNet, self).__init__();
        self.pointnet = PointNetfeat(global_feat=True, in_dim=4)

    def forward(self, x): x = x.permute(0, 2, 1); out, _, _ = self.pointnet(x); return out


class RadarFeatureExtractor(nn.Module):
    def __init__(self, num_classes, frame_num=20, hidden_size=128, feature_dim=256):
        super(RadarFeatureExtractor, self).__init__()
        self.pointnet = TimeDistributed(Sub_PointNet())
        self.pointnet_proj = nn.Linear(1024, hidden_size * 2)
        self.lstm_net = nn.LSTM(input_size=hidden_size * 2, hidden_size=hidden_size, num_layers=1, bidirectional=True,
                                batch_first=True)
        lstm_output_dim = hidden_size * 2
        self.feature_fc = nn.Sequential(nn.Linear(lstm_output_dim, feature_dim), nn.BatchNorm1d(feature_dim))
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, data):
        frame_features = self.pointnet(data)
        frame_features_proj = self.pointnet_proj(frame_features)
        lstm_out, _ = self.lstm_net(frame_features_proj)
        feature_aggregated = lstm_out.mean(dim=1)
        final_feature = self.feature_fc(feature_aggregated)
        logits = self.classifier(final_feature)
        return final_feature, logits


class ImprovedTripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__();
        self.margin = margin

    def forward(self, features, labels):
        if features is None or labels is None or features.size(0) < 2: return torch.tensor(0.0, device=features.device)
        dist_mat = torch.cdist(features, features);
        N = features.size(0);
        is_pos = labels.view(N, 1).eq(labels.view(1, N));
        is_neg = ~is_pos;
        is_pos.fill_diagonal_(False)
        if not (is_pos.any() and is_neg.any()): return torch.tensor(0.0, device=features.device)
        dist_ap, _ = torch.max(dist_mat * is_pos.float(), 1);
        dist_an, _ = torch.min(dist_mat.masked_fill(~is_neg, float('inf')) + torch.finfo(dist_mat.dtype).eps, 1)
        return F.relu(dist_ap - dist_an + self.margin).mean()


# --- MODIFIED: Main Model for Radar-Only returns features AND logits ---
class RadarOnlyReID(nn.Module):
    def __init__(self, num_classes, frame_num=20, radar_feature_dim=256):
        super().__init__()
        self.radar_extractor = RadarFeatureExtractor(num_classes, frame_num, feature_dim=radar_feature_dim)
        self.final_norm = nn.LayerNorm(radar_feature_dim)

    def forward(self, radar_data, is_training=True):
        features, logits = self.radar_extractor(radar_data)
        normalized_features = self.final_norm(features)
        if is_training:
            return normalized_features, logits
        else:  # For evaluation, only need features
            return normalized_features


# --- MODIFIED: Loss Class for Radar-Only now includes CE Loss ---
class RadarOnlyLoss(nn.Module):
    def __init__(self, margin=0.3, ce_weight=0.5):
        super().__init__()
        self.triplet_loss_fn = ImprovedTripletLoss(margin=margin)
        self.ce_loss_fn = nn.CrossEntropyLoss()
        self.ce_weight = ce_weight

    def forward(self, features, logits, targets):
        losses = {}
        loss_trip = self.triplet_loss_fn(features, targets)
        loss_ce = self.ce_loss_fn(logits, targets)
        losses['Trip_S'] = loss_trip.item()
        losses['CE_S'] = loss_ce.item()
        total_loss = loss_trip + self.ce_weight * loss_ce
        losses['total'] = total_loss.item()
        return total_loss, losses


# ==============================================================================
# 2. 训练与评估脚本
# ==============================================================================
class BalancedBatchSampler(torch.utils.data.Sampler):
    # ... (code is identical)
    def __init__(self, dataset, n_classes, n_samples):
        self.dataset, self.n_classes, self.n_samples = dataset, n_classes, n_samples;
        self.batch_size = n_classes * n_samples
        parent_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset;
        indices = dataset.indices if isinstance(dataset, Subset) else range(len(parent_dataset.samples))
        self.id_to_indices = {}
        for idx in indices:
            pid = parent_dataset.samples[idx][0]
            if pid not in self.id_to_indices: self.id_to_indices[pid] = []
            self.id_to_indices[pid].append(idx)
        self.ids = list(self.id_to_indices.keys());
        self.num_batches = len(indices) // self.batch_size if self.batch_size > 0 else 0

    def __iter__(self):
        for _ in range(self.num_batches):
            batch = [];
            available_ids = [pid for pid in self.ids if self.id_to_indices.get(pid)]
            if len(available_ids) < self.n_classes: continue
            replace = len(available_ids) < self.n_classes;
            selected_ids = np.random.choice(available_ids, self.n_classes, replace=replace)
            for pid in selected_ids:
                indices = self.id_to_indices[pid];
                replace_samples = len(indices) < self.n_samples;
                batch.extend(np.random.choice(indices, self.n_samples, replace=replace_samples))
            if batch: random.shuffle(batch); yield batch

    def __len__(self):
        return self.num_batches


def evaluate_reid(model, gallery_loader, query_loader, device):
    # ... (code is identical)
    model.eval()
    gallery_features, gallery_labels, query_features, query_labels = [], [], [], []
    with torch.no_grad():
        for _, radar_data, targets in tqdm(gallery_loader, "提取 Gallery 特征", leave=False):
            features = model(radar_data=radar_data.to(device), is_training=False);
            gallery_features.append(F.normalize(features, p=2, dim=1).cpu());
            gallery_labels.append(targets.cpu())
        for _, radar_data, targets in tqdm(query_loader, "提取 Query 特征", leave=False):
            features = model(radar_data=radar_data.to(device), is_training=False);
            query_features.append(F.normalize(features, p=2, dim=1).cpu());
            query_labels.append(targets.cpu())
    if not gallery_features or not query_features: return 0.0, 0.0, 0.0, 0.0
    gallery_features, gallery_labels = torch.cat(gallery_features), torch.cat(gallery_labels);
    query_features, query_labels = torch.cat(query_features), torch.cat(query_labels)
    if query_features.size(0) == 0 or gallery_features.size(0) == 0: return 0.0, 0.0, 0.0, 0.0
    sim_matrix, num_q = torch.mm(query_features, gallery_features.t()), query_labels.size(0);
    cmc, aps = torch.zeros(10, device=query_features.device), []
    for i in range(num_q):
        is_relevant = (gallery_labels == query_labels[i])
        if not is_relevant.any(): continue
        _, sorted_indices = torch.sort(sim_matrix[i], descending=True);
        sorted_is_relevant = is_relevant[sorted_indices];
        pos_match_ranks = torch.nonzero(sorted_is_relevant, as_tuple=False).squeeze(1)
        if pos_match_ranks.numel() > 0 and pos_match_ranks[0].item() < 10: cmc[pos_match_ranks[0].item():] += 1
        aps.append(average_precision_score(is_relevant.numpy(), sim_matrix[i].numpy()))
    mAP = np.mean(aps) * 100.0 if aps else 0.0;
    r1, r3, r5 = (cmc / num_q * 100.0)[[0, 2, 4]];
    print(f"\n> ReID 结果: Rank-1={r1:.2f}%  |  mAP={mAP:.2f}%  (R3={r3:.2f}%, R5={r5:.2f}%)");
    return r1, r3, r5, mAP


def plot_training_progress(epochs, reid_rank1s, reid_mAPs, filename):
    # ... (code is identical)
    fig, ax = plt.subplots(figsize=(10, 6));
    ax.plot(epochs, reid_rank1s, 'o-', label='ReID Rank-1');
    ax.plot(epochs, reid_mAPs, 's-', label='ReID mAP');
    ax.set_xlabel('Epoch');
    ax.set_ylabel('Score (%)');
    ax.legend();
    ax.grid(True);
    ax.set_ylim(0, 100);
    plt.tight_layout();
    plt.savefig(filename);
    plt.close(fig)


def train_and_evaluate_split(train_ids, test_ids, hparams):
    device, SEED, num_epochs = hparams['device'], hparams['SEED'], hparams['num_epochs']
    print(f"\n{'=' * 20} 阶段二: Radar-Only (弱结构 + CE) 基线训练 {'=' * 20}")

    train_dataset = VideoRadarDataset(hparams['video_base_dir'], hparams['radar_base_dir'], train_ids,
                                      frame_count=hparams['frame_num'], num_points=hparams['num_points'],
                                      transform=None)
    test_dataset = VideoRadarDataset(hparams['video_base_dir'], hparams['radar_base_dir'], test_ids,
                                     frame_count=hparams['frame_num'], num_points=hparams['num_points'], transform=None)

    person_to_samples = {pid: [idx for idx, (p, _, _) in enumerate(test_dataset.samples) if p == pid] for pid in
                         test_ids};
    gallery_indices, query_indices = [], []
    for pid, indices in person_to_samples.items():
        if not indices: continue
        random.shuffle(indices);
        split_point = max(1, int(len(indices) * 0.7));
        gallery_indices.extend(indices[:split_point]);
        query_indices.extend(indices[split_point:])
    if not query_indices and gallery_indices: query_indices.append(gallery_indices.pop())
    if not gallery_indices or not query_indices: print(f"警告: 测试集为空，无法进行评估。"); return 0.0, 0.0
    gallery_dataset, query_dataset = Subset(test_dataset, gallery_indices), Subset(test_dataset, query_indices)
    n_classes_per_batch = min(8, len(train_ids));
    n_samples_per_class = max(2, hparams['physical_batch_size'] // n_classes_per_batch)
    sampler = BalancedBatchSampler(train_dataset, n_classes=n_classes_per_batch, n_samples=n_samples_per_class);
    g = torch.Generator();
    g.manual_seed(SEED)
    train_loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=4, pin_memory=True,
                              worker_init_fn=seed_worker, generator=g)
    gallery_loader = DataLoader(gallery_dataset, batch_size=hparams['physical_batch_size'], shuffle=False,
                                num_workers=4, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    query_loader = DataLoader(query_dataset, batch_size=hparams['physical_batch_size'], shuffle=False, num_workers=4,
                              pin_memory=True, worker_init_fn=seed_worker, generator=g)

    model = RadarOnlyReID(num_classes=len(train_ids), frame_num=hparams['frame_num'],
                          radar_feature_dim=hparams['radar_feature_dim']).to(device)
    criterion = RadarOnlyLoss(margin=0.3, ce_weight=hparams['ce_student_weight'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=hparams['learning_rate'], weight_decay=hparams['weight_decay'])
    warmup_epochs = int(num_epochs * hparams['warmup_epochs_ratio'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lambda e: e / warmup_epochs if e < warmup_epochs else 0.5 * (
                                                              1 + math.cos(math.pi * (e - warmup_epochs) / (
                                                                  num_epochs - warmup_epochs))))
    reid_rank1s, reid_mAPs, epochs_recorded = [], [], [];
    best_mAP_run, best_rank1_run, epochs_since_best = 0.0, 0.0, 0
    print(f"--- 开始 Radar-Only (弱结构 + CE) 基线训练 ---")
    for epoch in range(num_epochs):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False,
                            bar_format='{l_bar}{bar:10}{r_bar}')
        optimizer.zero_grad()
        for batch_idx, (_, radar_data, targets) in enumerate(progress_bar):
            radar_data, targets = radar_data.to(device), targets.to(device)
            features, logits = model(radar_data=radar_data, is_training=True)
            loss, loss_details = criterion(features, logits, targets)
            loss = loss / hparams['accumulation_steps'];
            loss.backward()
            if (batch_idx + 1) % hparams['accumulation_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), hparams['clip_grad_norm'])
                optimizer.step();
                optimizer.zero_grad()
            postfix = {'loss': f"{loss_details.get('total', 0):.3f}", 'Trip_S': f"{loss_details.get('Trip_S', 0):.3f}",
                       'CE_S': f"{loss_details.get('CE_S', 0):.3f}"}
            progress_bar.set_postfix(postfix)
        scheduler.step()
        if (epoch + 1) % hparams['eval_interval'] == 0 or epoch == num_epochs - 1:
            model.eval()
            rank1, _, _, mAP = evaluate_reid(model, gallery_loader, query_loader, device)
            epochs_recorded.append(epoch + 1);
            reid_rank1s.append(rank1.item());
            reid_mAPs.append(mAP)
            plot_filename = os.path.join(hparams['checkpoint_dir'], f'training_progress.png');
            plot_training_progress(epochs_recorded, reid_rank1s, reid_mAPs, plot_filename)
            if mAP > best_mAP_run:
                best_mAP_run, best_rank1_run, epochs_since_best = mAP, rank1.item(), 0
                best_model_filename = os.path.join(hparams['checkpoint_dir'], f'best_model.pth');
                torch.save(model.state_dict(), best_model_filename)
                print(f"** 新的最佳模型已保存! mAP: {mAP:.2f}%, Rank-1: {rank1.item():.2f}% **")
            else:
                epochs_since_best += 1
                print(
                    f"   (无性能提升。当前最佳 mAP: {best_mAP_run:.2f}%. Early stopping: {epochs_since_best}/{hparams['early_stopping_patience']})")
            if epochs_since_best >= hparams['early_stopping_patience']: print(f"\n--- 提前停止被触发。 ---"); break
    print(f'\n--- 训练结束 --- 最佳 mAP: {best_mAP_run:.2f}%, 最佳 Rank-1: {best_rank1_run:.2f}%');
    return best_rank1_run, best_mAP_run


# ==============================================================================
# 4. 主函数
# ==============================================================================
def main_ablation_runner():
    parser = argparse.ArgumentParser(description="Radar baseline (PointNet->LSTM->MeanPool) trainer")
    parser.add_argument("--radar_base_dir", default=r"C:\Users\Administrator.DESKTOP-QBVF4GM\Documents\Playground\2s")
    parser.add_argument("--video_base_dir", default="")
    parser.add_argument("--split_ratio", type=float, default=0.7, help="train ratio by person IDs")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--early_stopping_patience", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    SEED = args.seed
    set_seed(SEED)
    base_hparams = {
        'physical_batch_size': args.batch_size, 'accumulation_steps': args.accumulation_steps,
        'num_classes': 0, 'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate, 'weight_decay': 1e-4, 'clip_grad_norm': 1.0,
        'warmup_epochs_ratio': 0.1, 'eval_interval': args.eval_interval,
        'early_stopping_patience': args.early_stopping_patience,
        'radar_feature_dim': 256,
        'num_points': 80,
        'frame_num': 20,
        'ce_student_weight': 0.5,  # Weight for CE loss in this baseline
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'SEED': SEED,
        'video_base_dir': args.video_base_dir,
        'radar_base_dir': args.radar_base_dir,
        'root_checkpoint_dir': 'checkpoints_radar_only_baselines',
    }

    radar_root = base_hparams['radar_base_dir']
    all_person_ids = []
    if os.path.isdir(radar_root):
        for n in os.listdir(radar_root):
            if n.startswith("p_"):
                try:
                    all_person_ids.append(int(n.split("_", 1)[1]))
                except Exception:
                    continue
    all_person_ids = sorted(all_person_ids)
    if len(all_person_ids) < 2:
        raise RuntimeError(f"No valid person folders found under {radar_root}. Expected p_<id> folders.")
    base_hparams['num_classes'] = len(all_person_ids)

    rng = random.Random(SEED)
    shuffled_ids = all_person_ids[:]
    rng.shuffle(shuffled_ids)
    split_idx = max(1, min(len(shuffled_ids) - 1, int(round(len(shuffled_ids) * args.split_ratio))))
    train_ids = sorted(shuffled_ids[:split_idx])
    test_ids = sorted(shuffled_ids[split_idx:])

    hparams_run = base_hparams.copy()
    run_checkpoint_dir = os.path.join(base_hparams['root_checkpoint_dir'], 'run_mean_pooling_with_ce')
    hparams_run['checkpoint_dir'] = run_checkpoint_dir
    os.makedirs(run_checkpoint_dir, exist_ok=True)

    print(f"--- 训练集ID (共 {len(train_ids)} 个): {train_ids}")
    print(f"--- 测试集ID (共 {len(test_ids)} 个): {test_ids}")

    best_r1, best_mAP = train_and_evaluate_split(train_ids=train_ids, test_ids=test_ids, hparams=hparams_run)

    print("\n\n" + "=" * 60)
    print(f"          Radar-Only (弱结构 + CE) 基线训练完成")
    print("=" * 60)
    print(f"本次运行最佳 Rank-1: {best_r1:.2f}% | 最佳 mAP: {best_mAP:.2f}%")
    print("=" * 60)


if __name__ == '__main__':
    main_ablation_runner()
