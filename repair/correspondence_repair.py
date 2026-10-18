#!/usr/bin/env python3
"""Correspondence-repair smoke test (scene0001_00).

Compares frozen-teacher projection under:
  clean / noisy / refined(no_weight) / repaired(candidate search + null + sharpness)

Verdict: repaired > refined => relocating sampling (main.pdf) recovers evidence
that pose refinement alone misses.
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


def repair_teacher(coords, feats_map, Kc, Kd, pose_dict, depth_cache, f3d_ref,
                   device='cuda', grid=5, n_cell=2, tau=0.10, e_null=2.5,
                   lam_d=20.0, lam_f=1.0, use_feat=True, sharp_th=0.30,
                   top2_margin=0.05, chunk=20000, mode='soft'):
    """Candidate-search projection with null state + sharpness test.

    mode='soft' : softmax posterior over candidates + null (main.pdf Eq.5).
    mode='argmin': pick min-depth candidate (geometry-only hard repair).
    Returns (features Nx384, support N).
    """
    N = coords.shape[0]
    D = 384
    acc = torch.zeros((N, D), device=device)
    ws = torch.zeros(N, device=device)
    off = np.linspace(-n_cell * 14, n_cell * 14, grid).astype(np.float32)
    du, dv = np.meshgrid(off, off)
    du = torch.from_numpy(du.ravel()).to(device)
    dv = torch.from_numpy(dv.ravel()).to(device)
    G = du.shape[0]
    f3d_n = F.normalize(f3d_ref, dim=1) if (f3d_ref is not None and use_feat) else None
    xyz = torch.from_numpy(coords).to(device)
    xyz1 = torch.cat([xyz, torch.ones((N, 1), device=device)], dim=1)

    for name, fp, scale in feats_map:
        if name not in pose_dict or name not in depth_cache:
            continue
        Tinv = torch.from_numpy(np.linalg.inv(pose_dict[name])).float().to(device)
        cam = (Tinv @ xyz1.t()).t()[:, :3]
        z = cam[:, 2]
        uc = Kc[0, 0] * cam[:, 0] / z.clamp_min(1e-6) + Kc[0, 2]
        vc = Kc[1, 1] * cam[:, 1] / z.clamp_min(1e-6) + Kc[1, 2]
        depth = torch.from_numpy(depth_cache[name]).to(device)
        H, W = depth.shape
        fp_t = fp.float().to(device)

        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            C = e - s
            zc = z[s:e]
            uc_c = uc[s:e, None] + du[None, :]
            vc_c = vc[s:e, None] + dv[None, :]
            # map candidate color pixels -> depth pixels via ray direction
            # (Kc and Kd may differ in resolution/focal)
            ud_c = Kd[0, 0] * (uc_c - Kc[0, 2]) / Kc[0, 0] + Kd[0, 2]
            vd_c = Kd[1, 1] * (vc_c - Kc[1, 2]) / Kc[1, 1] + Kd[1, 2]
            gd = torch.stack([2 * ud_c / (W - 1) - 1, 2 * vd_c / (H - 1) - 1],
                             dim=-1).view(1, C * G, 1, 2)
            ds = F.grid_sample(depth.unsqueeze(0).unsqueeze(0), gd, mode='bilinear',
                               align_corners=True, padding_mode='zeros').view(C * G)
            ds = ds.view(C, G)
            edep = (zc[:, None] - ds).abs()
            zt = P.sample_inline(fp_t, uc_c.ravel(), vc_c.ravel(), scale).view(C, G, -1)

            if mode == 'argmin':
                best = edep.argmin(dim=1)
                r = zt[torch.arange(C, device=device), best]
                keep = (edep.min(dim=1).values < 0.15) & (zc > 0.2) & (ds.gather(1, best[:, None]).squeeze(1) > 0.1)
            else:
                E = lam_d * F.huber_loss(edep, torch.zeros_like(edep), delta=0.05,
                                         reduction='none')
                if f3d_n is not None:
                    ztn = F.normalize(zt, dim=2)
                    cos = (ztn * f3d_n[s:e, None, :]).sum(2)
                    E = E + lam_f * (1.0 - cos)
                logits = torch.cat([torch.full((C, 1), -e_null, device=device),
                                    -E / tau], dim=1)
                q = torch.softmax(logits, dim=1)
                q_null = q[:, 0]
                qc = q[:, 1:]
                qsum = qc.sum(1).clamp_min(1e-8)
                r = (qc[:, :, None] * zt).sum(1) / qsum[:, None]
                keep = q_null < 0.5

            r = F.normalize(r, dim=1)
            acc[s:e] += keep[:, None].float() * r
            ws[s:e] += keep.float()

    feat = acc / ws.clamp_min(1e-8)[:, None]
    feat[ws <= 0] = 0
    return feat, ws


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
    ap.add_argument('--grid', type=int, default=5)
    ap.add_argument('--n-cell', type=float, default=2)
    ap.add_argument('--tau', type=float, default=0.10)
    ap.add_argument('--e-null', type=float, default=2.5)
    ap.add_argument('--lam-d', type=float, default=20.0)
    ap.add_argument('--lam-f', type=float, default=1.0)
    ap.add_argument('--sharp-th', type=float, default=0.30)
    ap.add_argument('--top2-margin', type=float, default=0.05)
    ap.add_argument('--no-feature', action='store_true')
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

    icp_idx = np.where(observed)[0]
    icp_idx = icp_idx[rng.choice(len(icp_idx), min(3000, len(icp_idx)), replace=False)]
    Xs = torch.from_numpy(coords[icp_idx]).float().to(device)
    f3d_icp = f3d_full[icp_idx]
    f3d_valid = (f3d_icp.norm(dim=1) > 0.5)
    axis = rng.randn(3); tdir = rng.randn(3); tdir /= (np.linalg.norm(tdir) + 1e-8)
    pf = P.make_perturb(axis, tdir)
    poses_noisy = {n: pf(p, args.rot_deg, args.trans_cm / 100.0) for n, p in poses_clean.items()}

    teacher_clean, _ = P.project_teacher(coords, feats_map, Kc, Kd, poses_clean, depth_cache, device)
    teacher_n, _ = P.project_teacher(coords, feats_map, Kc, Kd, poses_noisy, depth_cache, device)
    m_clean = eval_standard(teacher_clean.cpu().numpy(), labels20, weights, observed, args.seed)
    m_noisy = eval_standard(teacher_n.cpu().numpy(), labels20, weights, observed, args.seed)
    print('clean %.4f  noisy %.4f' % (m_clean, m_noisy), flush=True)

    icp_idx = np.where(observed)[0]
    icp_idx = icp_idx[rng.choice(len(icp_idx), min(3000, len(icp_idx)), replace=False)]
    Xs = torch.from_numpy(coords[icp_idx]).float().to(device)
    f3d_icp = f3d_full[icp_idx]
    f3d_valid = (f3d_icp.norm(dim=1) > 0.5)
    poses_ref = {}
    for name, p0 in poses_noisy.items():
        if name not in depth_cache:
            poses_ref[name] = p0; continue
        depth_t = torch.from_numpy(depth_cache[name]).float().to(device)
        p0t = torch.from_numpy(p0).float().to(device)
        fp, scale = feats_map_dict[name]
        pr = P.refine_pose_joint(p0t, Xs, depth_t, Kd_t, Kc_t, fp, scale, f3d_icp, f3d_valid,
                                 n_iter=300, lr=0.003, lam=10.0).cpu().numpy()
        poses_ref[name] = pr
    teacher_r, _ = P.project_teacher(coords, feats_map, Kc, Kd, poses_ref, depth_cache, device)
    m_refined = eval_standard(teacher_r.cpu().numpy(), labels20, weights, observed, args.seed)
    print('refined(no_weight) %.4f' % m_refined, flush=True)

    teacher_rep, ws_rep = repair_teacher(
        coords, feats_map, Kc_t, Kd_t, poses_ref, depth_cache, f3d_full,
        device=device, grid=args.grid, n_cell=args.n_cell, tau=args.tau,
        e_null=args.e_null, lam_d=args.lam_d, lam_f=args.lam_f,
        use_feat=not args.no_feature, sharp_th=args.sharp_th,
        top2_margin=args.top2_margin)
    m_repaired = eval_standard(teacher_rep.cpu().numpy(), labels20, weights, observed, args.seed)
    print('repaired(candidate+null+sharp) %.4f' % m_repaired, flush=True)

    teacher_am, _ = repair_teacher(
        coords, feats_map, Kc_t, Kd_t, poses_ref, depth_cache, f3d_full,
        device=device, grid=args.grid, n_cell=args.n_cell, mode='argmin')
    m_argmin = eval_standard(teacher_am.cpu().numpy(), labels20, weights, observed, args.seed)
    print('repaired(argmin-depth) %.4f' % m_argmin, flush=True)

    # main.pdf setting: candidate repair on NOISY poses (no pose refinement)
    teacher_rep_n, _ = repair_teacher(
        coords, feats_map, Kc_t, Kd_t, poses_noisy, depth_cache, f3d_full,
        device=device, grid=args.grid, n_cell=args.n_cell, tau=args.tau,
        e_null=args.e_null, lam_d=args.lam_d, lam_f=args.lam_f,
        use_feat=not args.no_feature, sharp_th=args.sharp_th,
        top2_margin=args.top2_margin)
    m_repaired_n = eval_standard(teacher_rep_n.cpu().numpy(), labels20, weights, observed, args.seed)
    print('norefine(soft) %.4f' % m_repaired_n, flush=True)

    teacher_am_n, _ = repair_teacher(
        coords, feats_map, Kc_t, Kd_t, poses_noisy, depth_cache, f3d_full,
        device=device, grid=args.grid, n_cell=args.n_cell, mode='argmin')
    m_argmin_n = eval_standard(teacher_am_n.cpu().numpy(), labels20, weights, observed, args.seed)
    print('norefine(argmin) %.4f' % m_argmin_n, flush=True)

    res = {
        'scene': args.npz.split('/')[-1].replace('_dino384_20cls.npz', ''),
        'rot_deg': args.rot_deg, 'trans_cm': args.trans_cm,
        'clean': m_clean, 'noisy': m_noisy,
        'refined_no_weight': m_refined,
        'repaired_soft': m_repaired,
        'repaired_argmin': m_argmin,
        'repaired_gain_over_refined': round(m_repaired - m_refined, 4),
        'norefine_soft': m_repaired_n,
        'norefine_argmin': m_argmin_n,
        'norefine_gain_over_noisy': round(m_argmin_n - m_noisy, 4),
    }
    with open(args.out, 'w') as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    print('saved:', args.out)


if __name__ == '__main__':
    main()
