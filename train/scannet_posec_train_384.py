#!/usr/bin/env python3
"""PoSeDC training loop (v2, optimized): learnable SparseConv features +
reliability-weighted 2D->3D distillation + EMA prototypes +
pose<->feature alternating refinement.

Optimizations over v1:
  1. sparse tensor voxelized ONCE and reused every forward (coords/colors fixed).
  2. prototype init + EMA update vectorized with np.add.at (no O(K*N) python loop).
  3. MiniBatchKMeans n_init=1 + max_iter + early-stop (CPU clustering is the bottleneck).
  4. depth images cached once (no repeated cv2.imread in project/refine).

Thesis (from validated oracle experiments):
  - Pose noise breaks the 2D(DINOv2)->3D projection; clean mIoU 0.83 -> 0.13 at 8deg/17cm.
  - Oracle joint refinement recovers ~99%; naive self-consistency loops fail.
  - Contribution = learnable SparseConv features + reliability weighting + EMA prototypes.

Three stages (run in order, all on the 3090):
  Stage 1 (clean):      --no-pose-noise --no-pose-refine
  Stage 2 (degradation):--pose-noise   --no-pose-refine
  Stage 3 (recovery):   --pose-noise   --pose-refine
"""
import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans

sys.path.insert(0, '/root')
import sparseconv_backbone as sbb


# ----------------------------------------------------------------------------
# helpers (self-contained; no dependency on 4090 scripts)
# ----------------------------------------------------------------------------
def normalize(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def many_to_one_metrics(pred, gt, weights, num_classes):
    """Cluster -> class mapping (present-classes-only mIoU, same as v2/D6).

    NOTE: not the standard ScanNet20 protocol (all 20 classes); convert before paper.
    """
    nproto = int(pred.max()) + 1
    hist = np.zeros((nproto, num_classes), dtype=np.float64)
    np.add.at(hist, (pred, gt), weights)
    mapping = hist.argmax(axis=1).astype(np.int64)
    mapped = mapping[pred]
    oacc = float(weights[mapped == gt].sum() / max(weights.sum(), 1))
    ious, accs = [], []
    for c in range(num_classes):
        tp = weights[(mapped == c) & (gt == c)].sum()
        fp = weights[(mapped == c) & (gt != c)].sum()
        fn = weights[(mapped != c) & (gt == c)].sum()
        gt_w = weights[gt == c].sum()
        ious.append(tp / max(tp + fp + fn, 1.0))
        accs.append(tp / max(gt_w, 1.0))
    valid = np.array([weights[gt == c].sum() > 0 for c in range(num_classes)])
    return {
        'oacc': oacc,
        'miou': float(np.mean(np.array(ious)[valid])) if valid.any() else 0.0,
        'macc': float(np.mean(np.array(accs)[valid])) if valid.any() else 0.0,
    }


# ----------------------------------------------------------------------------
# SE(3) / pose utilities (from scannet_pose_refine_joint_384_v10.py)
# ----------------------------------------------------------------------------
def np_axis_angle(axis, angle):
    a = axis / (np.linalg.norm(axis) + 1e-8)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], dtype=np.float64)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def make_perturb(axis, tdir):
    def f(pose, rot_deg, trans_m):
        R = np_axis_angle(axis, np.deg2rad(rot_deg))
        T = np.eye(4, dtype=np.float64); T[:3, :3] = R; T[:3, 3] = tdir * trans_m
        return (T @ pose).astype(np.float32)
    return f


def pose_error(T, T_clean):
    dR = T_clean[:3, :3] @ T[:3, :3].T
    tr = np.clip((np.trace(dR) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(tr)), float(np.linalg.norm(T_clean[:3, 3] - T[:3, 3]))


def make_se3(R, t):
    bottom = torch.tensor([[0., 0., 0., 1.]], device=R.device, dtype=R.dtype)
    return torch.cat([torch.cat([R, t.reshape(3, 1)], dim=1), bottom], dim=0)


def skew(r):
    x, y, z = r[0], r[1], r[2]
    o = torch.zeros_like(x)
    return torch.stack([torch.stack([o, -z, y]), torch.stack([z, o, -x]), torch.stack([-y, x, o])])


def exp_map_se3(p):
    r = p[:3]; t = p[3:]
    K = skew(r)
    theta_sq = (r * r).sum().clamp_min(1e-12)
    theta = theta_sq.sqrt()
    A = torch.sin(theta) / theta
    B = (1.0 - torch.cos(theta)) / theta_sq
    R = torch.eye(3, device=p.device, dtype=p.dtype) + A * K + B * (K @ K)
    return make_se3(R, t)


def inv_se3(T):
    R = T[:3, :3]; t = T[:3, 3]; Rt = R.transpose(0, 1)
    return make_se3(Rt, -(Rt @ t))


def sample_inline(fp, u, v, scale, patch=14):
    sx, sy = scale
    Hp, Wp = fp.shape[0], fp.shape[1]
    uu = ((u * sx / patch) / max(Wp - 1, 1)) * 2 - 1
    vv = ((v * sy / patch) / max(Hp - 1, 1)) * 2 - 1
    grid = torch.stack([uu, vv], dim=1).unsqueeze(0).unsqueeze(0)
    f = fp.permute(2, 0, 1).unsqueeze(0).float()
    out = F.grid_sample(f, grid, mode='bilinear', padding_mode='border', align_corners=True)
    return out[0, :, 0].T


def load_depth_cache(frames, feats_map):
    cache = {}
    for name, _, _ in feats_map:
        dp = os.path.join(frames, 'depth', name[:-4] + '.png')
        if os.path.exists(dp):
            cache[name] = cv2.imread(dp, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    return cache


# ----------------------------------------------------------------------------
# 2D -> 3D projection (geometric reliability weighting)
# ----------------------------------------------------------------------------
def project_teacher(coords, feats_map, Kc, Kd, pose_dict, depth_cache, device='cuda'):
    xyz = torch.from_numpy(coords).to(device)
    xyz1 = torch.cat([xyz, torch.ones((xyz.shape[0], 1), device=device)], dim=1)
    fs = torch.zeros((xyz.shape[0], 384), device=device)
    ws = torch.zeros(xyz.shape[0], device=device)
    n = len(feats_map)
    for i, (name, fp, scale) in enumerate(feats_map):
        if name not in pose_dict or name not in depth_cache:
            continue
        Tinv = np.linalg.inv(pose_dict[name])
        cam = (torch.from_numpy(Tinv).float().to(device) @ xyz1.t()).t()[:, :3]
        z = cam[:, 2]
        ud = Kd[0, 0] * cam[:, 0] / z.clamp_min(1e-6) + Kd[0, 2]
        vd = Kd[1, 1] * cam[:, 1] / z.clamp_min(1e-6) + Kd[1, 2]
        depth = depth_cache[name]
        dt = torch.from_numpy(depth).to(device); H, W = depth.shape
        ds = F.grid_sample(dt.unsqueeze(0).unsqueeze(0),
                           torch.stack([2 * ud / (W - 1) - 1, 2 * vd / (H - 1) - 1], 1).unsqueeze(0).unsqueeze(2),
                           mode='bilinear', align_corners=True, padding_mode='zeros').squeeze()
        vis = ((ds > 0.1) & (z > 0.2) & ((z - ds).abs() < 0.15)).float()
        rd = (z - ds).abs().clamp(max=1.0)
        w = vis * torch.exp(-rd / 0.10)
        uc = Kc[0, 0] * cam[:, 0] / z.clamp_min(1e-6) + Kc[0, 2]
        vc = Kc[1, 1] * cam[:, 1] / z.clamp_min(1e-6) + Kc[1, 2]
        zt = sample_inline(fp.float().to(device), uc, vc, scale)
        fs += w[:, None] * zt
        ws += w
        if (i + 1) % 8 == 0 or i + 1 == n:
            print(f'    project frame {i+1}/{n}', flush=True)
    f = fs / ws.clamp_min(1e-8)[:, None]
    f = F.normalize(f, dim=1)
    f[ws <= 0] = 0
    return f, ws


def refine_pose_joint(pose_init, Xs, depth_t, Kd, Kc, fp, scale, f3d, f3d_valid,
                      n_iter=200, lr=0.003, lam=10.0,
                      sem_reliability=False, sem_sigma=0.10, sem_conf=False):
    device = pose_init.device
    H, W = depth_t.shape
    Xh = torch.cat([Xs, torch.ones((Xs.shape[0], 1), device=device)], dim=1)
    fp_t = fp.float().to(device)
    p = torch.zeros(6, device=device, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([p], lr=lr)
    for _ in range(n_iter):
        T = exp_map_se3(p)
        pose = T @ pose_init
        Tinv = inv_se3(pose)
        cam = (Tinv @ Xh.t()).t()[:, :3]
        z = cam[:, 2]
        ud = Kd[0, 0] * cam[:, 0] / z.clamp_min(1e-6) + Kd[0, 2]
        vd = Kd[1, 1] * cam[:, 1] / z.clamp_min(1e-6) + Kd[1, 2]
        d = F.grid_sample(depth_t.unsqueeze(0).unsqueeze(0),
                          torch.stack([2 * ud / (W - 1) - 1, 2 * vd / (H - 1) - 1], 1).unsqueeze(0).unsqueeze(2),
                          mode='bilinear', align_corners=True, padding_mode='zeros').squeeze()
        valid = (z > 0.2) & (d > 0.1)
        if valid.sum() < 10:
            break
        loss_d = F.huber_loss(z[valid] - d[valid], torch.zeros_like(z[valid]), delta=0.05)
        uc = Kc[0, 0] * cam[:, 0] / z.clamp_min(1e-6) + Kc[0, 2]
        vc = Kc[1, 1] * cam[:, 1] / z.clamp_min(1e-6) + Kc[1, 2]
        zt = F.normalize(sample_inline(fp_t, uc, vc, scale), dim=1)
        cos = (zt * f3d).sum(1)
        vs = valid & f3d_valid
        if vs.any():
            sem_err = 1.0 - cos[vs]
            if sem_reliability:
                # reliability weighting: geometric (depth-residual correspondence confidence)
                w = torch.exp(-((z[vs] - d[vs]).abs()) / sem_sigma)
                if sem_conf:
                    # self-consistency confidence: only trust high-agreement correspondences
                    w = w * torch.sigmoid((cos[vs] - 0.2) / 0.1)
                loss_s = (sem_err * w).sum() / w.sum().clamp_min(1e-8)
            else:
                loss_s = sem_err.mean()
        else:
            loss_s = torch.zeros((), device=device)
        loss = loss_d + lam * loss_s
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_([p], 0.5); opt.step()
    with torch.no_grad():
        return exp_map_se3(p) @ pose_init


# ----------------------------------------------------------------------------
# Learnable backbone
# ----------------------------------------------------------------------------
class PoSeDCNet(nn.Module):
    def __init__(self, emb_dim=128, teacher_dim=384, hid=256):
        super().__init__()
        self.backbone = sbb.MinkUNet(in_channels=3, out_channels=emb_dim)
        self.head = nn.Sequential(nn.Linear(emb_dim, hid), nn.ReLU(), nn.Linear(hid, teacher_dim))

    def forward_sparse(self, sparse, inv):
        z = sbb.per_point_feature(self.backbone, sparse, inv)     # (N, emb_dim)
        z = F.normalize(z, dim=1)
        z_proj = self.head(z)                                     # (N, 384)
        return z, z_proj


def make_sparse(coords_np, colors_np, voxel_size, device):
    """Voxelize once (coords/colors fixed across training)."""
    coords_np = coords_np - coords_np.mean(0, keepdims=True)
    colors_np = colors_np.astype(np.float32) / 255.0 - 0.5
    return sbb.voxelize(coords_np, colors_np, voxel_size, device=device)


# ----------------------------------------------------------------------------
# Clustering (fast MiniBatchKMeans)
# ----------------------------------------------------------------------------
def kmeans_assign(x, k, seed):
    return MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=1, max_iter=100,
                           max_no_improvement=10, batch_size=8192, reassignment_ratio=0.0).fit_predict(x).astype(np.int64)


def init_prototypes(z_obs, labels_obs, k):
    counts = np.bincount(labels_obs, minlength=k)
    sums = np.zeros((k, z_obs.shape[1]), dtype=np.float32)
    np.add.at(sums, labels_obs, z_obs)
    mu = sums / np.maximum(counts, 1).astype(np.float32)[:, None]
    empty = counts == 0
    mu[empty] = z_obs.mean(axis=0)
    return F.normalize(torch.from_numpy(mu), dim=1)


def ema_update_prototypes(mu, znp, Pt_np, observed, consistent, ema):
    """Vectorized EMA update of prototypes from consistent + confident samples."""
    sel = consistent & observed
    zc = znp[sel]; pc = Pt_np[sel]
    if zc.shape[0] == 0:
        return mu
    k = mu.shape[0]
    counts = np.bincount(pc, minlength=k)
    sums = np.zeros((k, zc.shape[1]), dtype=np.float32)
    np.add.at(sums, pc, zc)
    valid = counts > 0
    new_mu = sums[valid] / counts[valid].astype(np.float32)[:, None]
    dev = mu.device
    mu_new = torch.from_numpy(new_mu).to(dev)
    mu[valid] = F.normalize(ema * mu[valid] + (1.0 - ema) * mu_new, dim=1)
    return mu


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def evaluate(z_np, labels20, weights, observed, K, seed):
    mask = observed & (labels20 >= 0)
    X = normalize(z_np[mask].astype(np.float32))
    pred = MiniBatchKMeans(n_clusters=K, random_state=seed, n_init=1, max_iter=100,
                           max_no_improvement=10, batch_size=8192, reassignment_ratio=0.0).fit_predict(X).astype(np.int64)
    return many_to_one_metrics(pred, labels20[mask].astype(np.int64), weights[mask], 20)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--feats', required=True)
    ap.add_argument('--frames', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--emb-dim', type=int, default=128)
    ap.add_argument('--n-protos', type=int, default=300)
    ap.add_argument('--temp', type=float, default=0.1)
    ap.add_argument('--ema', type=float, default=0.99)
    ap.add_argument('--tau', type=float, default=0.7)
    ap.add_argument('--lambda-distill', type=float, default=1.0)
    ap.add_argument('--lambda-proto', type=float, default=1.0)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lr-decay', action='store_true')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--recluster-every', type=int, default=10)
    ap.add_argument('--eval-every', type=int, default=10)
    ap.add_argument('--voxel-size', type=float, default=0.05)
    ap.add_argument('--max-points', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--pose-noise', action='store_true')
    ap.add_argument('--rot-deg', type=float, default=5.0)
    ap.add_argument('--trans-cm', type=float, default=11.0)
    ap.add_argument('--pose-refine', action='store_true')
    ap.add_argument('--pose-refine-every', type=int, default=5)
    ap.add_argument('--icp-points', type=int, default=3000)
    ap.add_argument('--lam', type=float, default=10.0)
    ap.add_argument('--sem-reliability', action='store_true')
    ap.add_argument('--sem-sigma', type=float, default=0.10)
    ap.add_argument('--sem-conf', action='store_true')
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device={device}', flush=True)

    # ---- load data ----
    z = np.load(args.npz)
    coords = z['coords'].astype(np.float32)
    colors = z['colors'].astype(np.float32)
    labels20 = z['labels20'].astype(np.int64) if 'labels20' in z else z['labels'].astype(np.int64)
    weights = z['weight_sum'].astype(np.float32)
    observed = (weights > 0) & (labels20 >= 0)

    if args.max_points > 0 and args.max_points < len(coords):
        sel = np.random.RandomState(args.seed).choice(len(coords), args.max_points, replace=False)
        coords, colors, labels20, weights, observed = coords[sel], colors[sel], labels20[sel], weights[sel], observed[sel]

    feats_map = torch.load(args.feats, map_location='cpu')
    feats_map_dict = {name: (fp, scale) for name, fp, scale in feats_map}
    Kc = np.loadtxt(os.path.join(args.frames, 'intrinsic_color.txt'))[:3, :3]
    Kd = np.loadtxt(os.path.join(args.frames, 'intrinsic_depth.txt'))[:3, :3]
    Kc_t = torch.from_numpy(Kc).float().to(device)
    Kd_t = torch.from_numpy(Kd).float().to(device)
    depth_cache = load_depth_cache(args.frames, feats_map)

    poses_clean = {}
    for name, _, _ in feats_map:
        pp = os.path.join(args.frames, 'pose', name[:-4] + '.txt')
        if os.path.exists(pp):
            poses_clean[name] = np.loadtxt(pp)

    # ---- pose noise ----
    poses = dict(poses_clean)
    if args.pose_noise:
        rng = np.random.RandomState(args.seed)
        axis = rng.randn(3); tdir = rng.randn(3); tdir /= (np.linalg.norm(tdir) + 1e-8)
        pf = make_perturb(axis, tdir)
        poses = {n: pf(p, args.rot_deg, args.trans_cm / 100.0) for n, p in poses_clean.items()}

    all_idx = np.where(observed)[0]
    rng = np.random.RandomState(args.seed)
    icp_idx = all_idx[rng.choice(len(all_idx), min(args.icp_points, len(all_idx)), replace=False)]
    Xs = torch.from_numpy(coords[icp_idx]).float().to(device)

    # ---- precompute sparse tensor ONCE ----
    t0 = time.time()
    sparse, inv = make_sparse(coords, colors, args.voxel_size, device)
    print(f'sparse voxelized: {sparse.F.shape[0]} voxels in {time.time()-t0:.1f}s', flush=True)

    net = PoSeDCNet(emb_dim=args.emb_dim).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    # ---- initial teacher ----
    t0 = time.time()
    teacher, teach_w = project_teacher(coords, feats_map, Kc, Kd, poses, depth_cache, device)
    print(f'teacher projected in {time.time()-t0:.1f}s', flush=True)
    teach_w_t = teach_w.to(device)

    # ---- init prototypes ----
    net.eval()
    with torch.no_grad():
        zz, _ = net.forward_sparse(sparse, inv)
        znp = zz.cpu().numpy()
    P = kmeans_assign(normalize(znp[observed]), args.n_protos, args.seed)
    P_full = -np.ones(len(coords), dtype=np.int64); P_full[observed] = P
    mu = init_prototypes(znp[observed], P, args.n_protos).to(device)
    Pt = torch.from_numpy(P_full).to(device)

    log = []
    best = {'miou': 0.0, 'epoch': 0}
    for ep in range(1, args.epochs + 1):
        te = time.time()
        net.train()

        # ---- Phase A: feature learning ----
        zz, z_proj = net.forward_sparse(sparse, inv)
        cos = (F.normalize(z_proj, dim=1) * teacher).sum(1)
        wmask = teach_w_t > 0
        loss_distill = (teach_w_t[wmask] * (1.0 - cos[wmask])).mean() if wmask.any() else torch.zeros((), device=device)

        logits = zz @ mu.t() / args.temp
        omask = torch.from_numpy(observed).to(device)
        loss_proto = F.cross_entropy(logits[omask], Pt[omask]) if omask.any() else torch.zeros((), device=device)

        loss = args.lambda_distill * loss_distill + args.lambda_proto * loss_proto
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
        if args.lr_decay:
            lr = args.lr * 0.5 * (1.0 + math.cos(math.pi * ep / args.epochs))
            for g in opt.param_groups:
                g['lr'] = lr

        # EMA prototypes (vectorized) -- skipped when distill-only
        if args.lambda_proto > 0:
            with torch.no_grad():
                zz, _ = net.forward_sparse(sparse, inv)
                logits = zz @ mu.t() / args.temp
                conf, pred = F.softmax(logits, dim=1).max(dim=1)
                consistent = ((pred == Pt) & (conf >= args.tau)).cpu().numpy()
                znp = zz.cpu().numpy(); pnp = Pt.cpu().numpy()
                mu = ema_update_prototypes(mu, znp, pnp, observed, consistent, args.ema)

            # periodic re-clustering
            if ep % args.recluster_every == 0:
                with torch.no_grad():
                    zz, _ = net.forward_sparse(sparse, inv)
                    znp = zz.cpu().numpy()
                P = kmeans_assign(normalize(znp[observed]), args.n_protos, args.seed + ep)
                P_full = -np.ones(len(coords), dtype=np.int64); P_full[observed] = P
                Pt = torch.from_numpy(P_full).to(device)

        # ---- Phase B: pose refinement ----
        if args.pose_refine and ep % args.pose_refine_every == 0:
            net.eval()
            with torch.no_grad():
                zz, z_proj = net.forward_sparse(sparse, inv)
                f3d = F.normalize(z_proj, dim=1)[icp_idx]
                f3d_valid = f3d.norm(dim=1) > 0.5
            for name, p0 in poses.items():
                if name not in depth_cache:
                    continue
                depth_t = torch.from_numpy(depth_cache[name]).float().to(device)
                p0t = torch.from_numpy(p0).float().to(device)
                fp, scale = feats_map_dict[name]
                poses[name] = refine_pose_joint(p0t, Xs, depth_t, Kd_t, Kc_t, fp, scale, f3d, f3d_valid,
                                                lam=args.lam, sem_reliability=args.sem_reliability,
                                                sem_sigma=args.sem_sigma, sem_conf=args.sem_conf).cpu().numpy()
            teacher, teach_w = project_teacher(coords, feats_map, Kc, Kd, poses, depth_cache, device)
            teach_w_t = teach_w.to(device)

        # ---- evaluate ----
        if ep % args.eval_every == 0 or ep == args.epochs:
            net.eval()
            with torch.no_grad():
                zz, _ = net.forward_sparse(sparse, inv)
                znp = zz.cpu().numpy()
            m = evaluate(znp, labels20, weights, observed, args.n_protos, args.seed)
            line = {'epoch': ep, 'loss': float(loss), 'loss_distill': float(loss_distill),
                    'loss_proto': float(loss_proto), 'miou': float(m['miou']),
                    'macc': float(m['macc']), 'oacc': float(m['oacc'])}
            if args.pose_noise:
                pe = np.mean([pose_error(poses[n], poses_clean[n]) for n in poses if n in poses_clean], axis=0)
                line['pose_err_deg'] = float(pe[0]); line['pose_err_cm'] = float(pe[1] * 100)
            log.append(line)
            if m['miou'] > best['miou']:
                best = {'miou': m['miou'], 'epoch': ep}
                torch.save({'net': net.state_dict(), 'prototypes': mu.cpu(), 'poses': poses},
                           os.path.join(args.out_dir, 'best.pt'))
            extra = '' if not args.pose_noise else f'  pose {line["pose_err_deg"]:.2f}deg/{line["pose_err_cm"]:.2f}cm'
            print(f'ep {ep:3d} ({time.time()-te:.1f}s) loss {loss.item():.4f} (d {loss_distill.item():.4f} p {loss_proto.item():.4f}) '
                  f'mIoU {m["miou"]:.4f} mAcc {m["macc"]:.4f} oAcc {m["oacc"]:.4f}{extra}', flush=True)

    with open(os.path.join(args.out_dir, 'train_log.json'), 'w') as f:
        json.dump({'args': vars(args), 'best': best, 'log': log}, f, indent=2)
    print(f'BEST: mIoU {best["miou"]:.4f} @ epoch {best["epoch"]}')
    print('saved:', os.path.join(args.out_dir, 'train_log.json'))


if __name__ == '__main__':
    main()
