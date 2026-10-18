#!/usr/bin/env python3
"""标准 ScanNet20 协议重算: 20类全算, 缺席类 IoU=0 (区别于 present-only 虚高).

注意: 这是"单场景"标准协议; 论文级数字需多场景(全 val/test)平均, 本脚本只算 scene0001_00.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/root')
import scannet_posec_train_384 as P


def standard_metrics(pred, gt, weights, num_classes=20):
    """标准协议: 20类全算 mIoU/mAcc, 缺席类=0."""
    nproto = int(pred.max()) + 1
    hist = np.zeros((nproto, num_classes), dtype=np.float64)
    np.add.at(hist, (pred, gt), weights)
    mapping = hist.argmax(axis=1).astype(np.int64)
    mapped = mapping[pred]
    ious, accs = [], []
    for c in range(num_classes):
        tp = weights[(mapped == c) & (gt == c)].sum()
        fp = weights[(mapped == c) & (gt != c)].sum()
        fn = weights[(mapped != c) & (gt == c)].sum()
        gt_w = weights[gt == c].sum()
        denom = tp + fp + fn
        ious.append(tp / denom if denom > 0 else 0.0)
        accs.append(tp / gt_w if gt_w > 0 else 0.0)
    oacc = float(weights[mapped == gt].sum() / max(weights.sum(), 1))
    return {'miou': float(np.mean(ious)), 'macc': float(np.mean(accs)), 'oacc': oacc}


def eval_standard(features, labels20, weights, observed, seed=0, K=300):
    mask = observed & (labels20 >= 0)
    X = P.normalize(features[mask].astype(np.float32))
    pred = P.kmeans_assign(X, K, seed)
    return standard_metrics(pred, labels20[mask].astype(np.int64), weights[mask], 20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--feats', required=True)
    ap.add_argument('--frames', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--emb-dim', type=int, default=128)
    ap.add_argument('--rot-deg', type=float, default=5.0)
    ap.add_argument('--trans-cm', type=float, default=11.0)
    ap.add_argument('--lam', type=float, default=10.0)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rng = np.random.RandomState(args.seed)

    z = np.load(args.npz)
    coords = z['coords'].astype(np.float32)
    colors = z['colors'].astype(np.float32)
    labels20 = z['labels20'].astype(np.int64) if 'labels20' in z else z['labels'].astype(np.int64)
    weights = z['weight_sum'].astype(np.float32)
    observed = (weights > 0) & (labels20 >= 0)

    feats_map = torch.load(args.feats, map_location='cpu')
    Kc = np.loadtxt(os.path.join(args.frames, 'intrinsic_color.txt'))[:3, :3]
    Kd = np.loadtxt(os.path.join(args.frames, 'intrinsic_depth.txt'))[:3, :3]
    Kc_t = torch.from_numpy(Kc).float().to(device)
    Kd_t = torch.from_numpy(Kd).float().to(device)
    depth_cache = P.load_depth_cache(args.frames, feats_map)
    feats_map_dict = {name: (fp, scale) for name, fp, scale in feats_map}
    poses_clean = {}
    for name, _, _ in feats_map:
        pp = os.path.join(args.frames, 'pose', name[:-4] + '.txt')
        if os.path.exists(pp):
            poses_clean[name] = np.loadtxt(pp)

    net = P.PoSeDCNet(emb_dim=args.emb_dim).to(device)
    ck = torch.load(args.ckpt, map_location='cpu')
    net.load_state_dict(ck['net'])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    sparse, inv = P.make_sparse(coords, colors, 0.05, device)
    with torch.no_grad():
        zz, z_proj = net.forward_sparse(sparse, inv)
    z_proj_np = z_proj.cpu().numpy()
    f3d = F.normalize(z_proj, dim=1)

    all_idx = np.where(observed)[0]
    icp_idx = all_idx[rng.choice(len(all_idx), min(3000, len(all_idx)), replace=False)]
    Xs = torch.from_numpy(coords[icp_idx]).float().to(device)
    f3d_icp = f3d[icp_idx]
    f3d_valid = f3d_icp.norm(dim=1) > 0.5

    axis = rng.randn(3); tdir = rng.randn(3); tdir /= (np.linalg.norm(tdir) + 1e-8)
    pf = P.make_perturb(axis, tdir)
    poses_noisy = {n: pf(p, args.rot_deg, args.trans_cm / 100.0) for n, p in poses_clean.items()}

    poses_refined = {}
    errs_n, errs_r = [], []
    for name, p0 in poses_noisy.items():
        if name not in depth_cache:
            poses_refined[name] = p0; continue
        depth_t = torch.from_numpy(depth_cache[name]).float().to(device)
        p0t = torch.from_numpy(p0).float().to(device)
        fp, scale = feats_map_dict[name]
        pr = P.refine_pose_joint(p0t, Xs, depth_t, Kd_t, Kc_t, fp, scale, f3d_icp, f3d_valid,
                                 n_iter=300, lr=0.003, lam=args.lam,
                                 sem_reliability=True, sem_sigma=0.10, sem_conf=True).cpu().numpy()
        poses_refined[name] = pr
        a0, d0 = P.pose_error(p0, poses_clean[name]); a1, d1 = P.pose_error(pr, poses_clean[name])
        errs_n.append((a0, d0 * 100)); errs_r.append((a1, d1 * 100))

    # 评估: 标准协议
    print('evaluating standard protocol...', flush=True)
    m_feat = eval_standard(z_proj_np, labels20, weights, observed, args.seed)
    teacher_clean, _ = P.project_teacher(coords, feats_map, Kc, Kd, poses_clean, depth_cache, device)
    teacher_n, _ = P.project_teacher(coords, feats_map, Kc, Kd, poses_noisy, depth_cache, device)
    teacher_r, _ = P.project_teacher(coords, feats_map, Kc, Kd, poses_refined, depth_cache, device)
    m_clean = eval_standard(teacher_clean.cpu().numpy(), labels20, weights, observed, args.seed)
    m_n = eval_standard(teacher_n.cpu().numpy(), labels20, weights, observed, args.seed)
    m_r = eval_standard(teacher_r.cpu().numpy(), labels20, weights, observed, args.seed)
    eb = np.array(errs_n).mean(0); er = np.array(errs_r).mean(0)

    res = {
        'protocol': 'ScanNet20 standard (20 classes, absent=0), single scene',
        'frozen_feat': {'miou': m_feat['miou'], 'macc': m_feat['macc'], 'oacc': m_feat['oacc']},
        'teacher_clean': {'miou': m_clean['miou'], 'macc': m_clean['macc'], 'oacc': m_clean['oacc']},
        'teacher_noisy': {'miou': m_n['miou'], 'macc': m_n['macc'], 'oacc': m_n['oacc']},
        'teacher_refined': {'miou': m_r['miou'], 'macc': m_r['macc'], 'oacc': m_r['oacc']},
        'pose_noisy': {'deg': float(eb[0]), 'cm': float(eb[1])},
        'pose_refined': {'deg': float(er[0]), 'cm': float(er[1])},
    }
    print(json.dumps(res, indent=2))
    with open(os.path.join(args.out_dir, 'standard_protocol.json'), 'w') as f:
        json.dump(res, f, indent=2)
    print('saved:', os.path.join(args.out_dir, 'standard_protocol.json'))


if __name__ == '__main__':
    main()
