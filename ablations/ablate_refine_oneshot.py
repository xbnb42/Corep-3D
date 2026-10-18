#!/usr/bin/env python3
"""
One-shot pose-refinement ablation (matches the MAIN table design, not the
in-loop training design).

The headline mechanism: train student on CLEAN teacher (distill-only), then at
eval time inject pose noise and run a SINGLE refinement pass.  This script
ablates that single-pass refine_pose_joint by toggling its components:
  geom (depth residual)  vs  semantic (DINO feature consistency)
  reliability weighting (sem_reliability / sem_conf)

Uses a trained student checkpoint (from scannet_posec_train_384.py, clean).
"""
import argparse, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/root')
import scannet_posec_train_384 as P

DATA = '/root/autodl-tmp/pointdc_data/ScanNet'
CACHE = '/root/autodl-tmp/posec_cache'


def standard_miou(pred, gt, weights, num_classes=20):
    nproto = int(pred.max()) + 1
    hist = np.zeros((nproto, num_classes), dtype=np.float64)
    np.add.at(hist, (pred, gt), weights)
    mapping = hist.argmax(axis=1).astype(np.int64)
    mapped = mapping[pred]
    ious = []
    for c in range(num_classes):
        tp = weights[(mapped == c) & (gt == c)].sum()
        fp = weights[(mapped == c) & (gt != c)].sum()
        fn = weights[(mapped != c) & (gt == c)].sum()
        denom = tp + fp + fn
        ious.append(tp / denom if denom > 0 else 0.0)
    return float(np.mean(ious))


def eval_standard(features, labels20, weights, observed, seed=0, K=300):
    mask = observed & (labels20 >= 0)
    X = P.normalize(features[mask].astype(np.float32))
    pred = P.kmeans_assign(X, K, seed)
    return standard_miou(pred, labels20[mask].astype(np.int64), weights[mask], 20)


def refine_pose_joint(pose_init, Xs, depth_t, Kd, Kc, fp, scale, f3d, f3d_valid,
                      n_iter=300, lr=0.003, lam=10.0,
                      sem_reliability=False, sem_sigma=0.10, sem_conf=False,
                      geom_only=False, sem_only=False, sem_conf_only=False):
    device = pose_init.device
    H, W = depth_t.shape
    Xh = torch.cat([Xs, torch.ones((Xs.shape[0], 1), device=device)], dim=1)
    fp_t = fp.float().to(device)
    p = torch.zeros(6, device=device, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([p], lr=lr)
    for _ in range(n_iter):
        T = P.exp_map_se3(p)
        pose = T @ pose_init
        Tinv = P.inv_se3(pose)
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
        loss_d = torch.zeros((), device=device) if sem_only else \
            F.huber_loss(z[valid] - d[valid], torch.zeros_like(z[valid]), delta=0.05)
        uc = Kc[0, 0] * cam[:, 0] / z.clamp_min(1e-6) + Kc[0, 2]
        vc = Kc[1, 1] * cam[:, 1] / z.clamp_min(1e-6) + Kc[1, 2]
        zt = F.normalize(P.sample_inline(fp_t, uc, vc, scale), dim=1)
        cos = (zt * f3d).sum(1)
        vs = valid & f3d_valid
        if geom_only:
            loss_s = torch.zeros((), device=device)
        elif vs.any():
            sem_err = 1.0 - cos[vs]
            if sem_reliability:
                w = torch.exp(-((z[vs] - d[vs]).abs()) / sem_sigma)
                if sem_conf:
                    w = w * torch.sigmoid((cos[vs] - 0.2) / 0.1)
                loss_s = (sem_err * w).sum() / w.sum().clamp_min(1e-8)
            elif sem_conf_only:
                w = torch.sigmoid((cos[vs] - 0.2) / 0.1)
                loss_s = (sem_err * w).sum() / w.sum().clamp_min(1e-8)
            else:
                loss_s = sem_err.mean()
        else:
            loss_s = torch.zeros((), device=device)
        loss = loss_d + lam * loss_s
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_([p], 0.5); opt.step()
    with torch.no_grad():
        return P.exp_map_se3(p) @ pose_init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', required=True)
    ap.add_argument('--feats', required=True)
    ap.add_argument('--frames', required=True)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--emb-dim', type=int, default=128)
    ap.add_argument('--rot-deg', type=float, default=5.0)
    ap.add_argument('--trans-cm', type=float, default=11.0)
    ap.add_argument('--lam', type=float, default=10.0)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

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

    # student features (semantic reference)
    net = P.PoSeDCNet(emb_dim=args.emb_dim).to(device)
    ck = torch.load(args.ckpt, map_location='cpu')
    net.load_state_dict(ck['net'])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    sparse, inv = P.make_sparse(coords, colors, 0.05, device)
    with torch.no_grad():
        _, z_proj = net.forward_sparse(sparse, inv)
    f3d_full = F.normalize(z_proj, dim=1)

    all_idx = np.where(observed)[0]
    icp_idx = all_idx[rng.choice(len(all_idx), min(3000, len(all_idx)), replace=False)]
    Xs = torch.from_numpy(coords[icp_idx]).float().to(device)
    f3d = f3d_full[icp_idx]
    f3d_valid = (f3d.norm(dim=1) > 0.5)

    # inject noise
    axis = rng.randn(3); tdir = rng.randn(3); tdir /= (np.linalg.norm(tdir) + 1e-8)
    pf = P.make_perturb(axis, tdir)
    poses_noisy = {n: pf(p, args.rot_deg, args.trans_cm / 100.0) for n, p in poses_clean.items()}

    # teacher clean (reference)
    teacher_clean, _ = P.project_teacher(coords, feats_map, Kc, Kd, poses_clean, depth_cache, device)
    teacher_n, _ = P.project_teacher(coords, feats_map, Kc, Kd, poses_noisy, depth_cache, device)
    m_frozen = eval_standard(z_proj.cpu().numpy(), labels20, weights, observed, args.seed)
    m_clean = eval_standard(teacher_clean.cpu().numpy(), labels20, weights, observed, args.seed)
    m_noisy = eval_standard(teacher_n.cpu().numpy(), labels20, weights, observed, args.seed)

    variants = {
        'full':      dict(sem_reliability=True, sem_conf=True),
        'geom_only': dict(geom_only=True),
        'sem_only':  dict(sem_only=True, sem_reliability=True, sem_conf=True),
        'no_weight': dict(),
        'geom_rel':  dict(sem_reliability=True, sem_conf=False),
        'conf_only': dict(sem_conf_only=True),
    }

    res = {'frozen_miou': m_frozen, 'teacher_clean_miou': m_clean, 'teacher_noisy_miou': m_noisy,
           'variants': {}}
    for vname, kw in variants.items():
        poses_ref = {}
        errs_n, errs_r = [], []
        for name, p0 in poses_noisy.items():
            if name not in depth_cache:
                poses_ref[name] = p0; continue
            depth_t = torch.from_numpy(depth_cache[name]).float().to(device)
            p0t = torch.from_numpy(p0).float().to(device)
            fp, scale = feats_map_dict[name]
            pr = refine_pose_joint(p0t, Xs, depth_t, Kd_t, Kc_t, fp, scale, f3d, f3d_valid,
                                   n_iter=300, lr=0.003, lam=args.lam, **kw).cpu().numpy()
            poses_ref[name] = pr
            a0, d0 = P.pose_error(p0, poses_clean[name]); a1, d1 = P.pose_error(pr, poses_clean[name])
            errs_n.append((a0, d0 * 100)); errs_r.append((a1, d1 * 100))
        teacher_r, _ = P.project_teacher(coords, feats_map, Kc, Kd, poses_ref, depth_cache, device)
        m_r = eval_standard(teacher_r.cpu().numpy(), labels20, weights, observed, args.seed)
        eb = np.array(errs_n).mean(0); er = np.array(errs_r).mean(0)
        res['variants'][vname] = {
            'pose_before_deg': float(eb[0]), 'pose_before_cm': float(eb[1]),
            'pose_after_deg': float(er[0]), 'pose_after_cm': float(er[1]),
            'teacher_refined_miou': m_r}
        print(f'{vname:10s} pose {eb[0]:.2f}/{eb[1]:.1f} -> {er[0]:.2f}deg/{er[1]:.1f}cm  refined_miou {m_r:.4f}', flush=True)

    with open(args.out, 'w') as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    print('saved:', args.out)


if __name__ == '__main__':
    main()
