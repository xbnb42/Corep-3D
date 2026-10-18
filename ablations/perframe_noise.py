#!/usr/bin/env python3
"""Per-frame pose noise (per-scan drift): each frame gets its own random 5deg/11cm
perturbation instead of one shared direction. Tests whether the diagnosis and
Mechanism A recovery hold under realistic per-frame drift."""
import sys, os, json, time
import numpy as np
import torch

sys.path.insert(0, '/root')

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

import scannet_posec_train_384 as P

CACHE = "/root/autodl-tmp/posec_cache"
FRAMES = "/root/autodl-tmp/pointdc_data/ScanNet/frames"
LIST = "/root/scannetv2_val.txt"
OUT_SUMMARY = f"{CACHE}/perframe_noise_summary.json"

N_SCENES = 400  # all scenes with a checkpoint (full run, overnight)
ROT_DEG = 5.0
TRANS_M = 0.11
device = 'cuda' if torch.cuda.is_available() else 'cpu'

scenes = [l.strip() for l in open(LIST) if l.strip()]
scenes = [s for s in scenes if os.path.exists(f"{CACHE}/ms_{s}/best.pt")][:N_SCENES]
print(f"scenes ({len(scenes)}): {scenes}", flush=True)

done = {}
if os.path.exists(OUT_SUMMARY):
    try:
        done = {r["scene"]: r for r in json.load(open(OUT_SUMMARY))}
    except Exception:
        pass
rows = list(done.values())

for sc in scenes:
    if sc in done:
        continue
    npz = f"{CACHE}/{sc}_dino384_20cls.npz"
    feats = f"{CACHE}/{sc}_feats_map.pt"
    frames = f"{FRAMES}/{sc}"
    ckpt = f"{CACHE}/ms_{sc}/best.pt"
    if not (os.path.exists(npz) and os.path.exists(feats) and os.path.exists(ckpt)):
        print(f"  skip {sc}", flush=True); continue
    t0 = time.time()
    z = np.load(npz)
    coords = z['coords'].astype(np.float32)
    colors = z['colors'].astype(np.float32)
    labels20 = z['labels20'].astype(np.int64) if 'labels20' in z else z['labels'].astype(np.int64)
    weights = z['weight_sum'].astype(np.float32)
    observed = (weights > 0) & (labels20 >= 0)

    feats_map = torch.load(feats, map_location='cpu')
    Kc = np.loadtxt(os.path.join(frames, 'intrinsic_color.txt'))[:3,:3]
    Kd = np.loadtxt(os.path.join(frames, 'intrinsic_depth.txt'))[:3,:3]
    Kc_t = torch.from_numpy(Kc).float().to(device)
    Kd_t = torch.from_numpy(Kd).float().to(device)
    depth_cache = P.load_depth_cache(frames, feats_map)
    feats_map_dict = {name: (fp, scale) for name, fp, scale in feats_map}
    poses_clean = {}
    for name, _, _ in feats_map:
        pp = os.path.join(frames, 'pose', name[:-4] + '.txt')
        if os.path.exists(pp):
            poses_clean[name] = np.loadtxt(pp)

    # student + f3d
    net = P.PoSeDCNet(emb_dim=128).to(device)
    ck = torch.load(ckpt, map_location='cpu')
    net.load_state_dict(ck['net']); net.eval()
    for p in net.parameters(): p.requires_grad_(False)
    sparse, inv = P.make_sparse(coords, colors, 0.05, device)
    with torch.no_grad():
        _, z_proj = net.forward_sparse(sparse, inv)
    f3d_full = torch.nn.functional.normalize(z_proj, dim=1)

    # rng order matches P4: icp_idx choice first
    rng = np.random.RandomState(0)
    all_idx = np.where(observed)[0]
    icp_idx = all_idx[rng.choice(len(all_idx), min(3000, len(all_idx)), replace=False)]
    Xs = torch.from_numpy(coords[icp_idx]).float().to(device)
    f3d_icp = f3d_full[icp_idx]
    f3d_valid = f3d_icp.norm(dim=1) > 0.5

    # per-frame noise: each frame its own random direction (same magnitude)
    poses_noisy = {}
    for name, p in poses_clean.items():
        axis = rng.randn(3)
        tdir = rng.randn(3); tdir /= (np.linalg.norm(tdir) + 1e-8)
        pf = P.make_perturb(axis, tdir)
        poses_noisy[name] = pf(p, ROT_DEG, TRANS_M)

    # refine (Mechanism A)
    poses_ref = {}
    for name, p0 in poses_noisy.items():
        if name not in depth_cache:
            poses_ref[name] = p0; continue
        depth_t = torch.from_numpy(depth_cache[name]).float().to(device)
        p0t = torch.from_numpy(p0).float().to(device)
        fp, scale = feats_map_dict[name]
        poses_ref[name] = P.refine_pose_joint(p0t, Xs, depth_t, Kd_t, Kc_t, fp, scale, f3d_icp, f3d_valid,
                                               n_iter=300, lr=0.003, lam=10.0).cpu().numpy()

    teacher_clean, _ = P.project_teacher(coords, feats_map, Kc, Kd, poses_clean, depth_cache, device)
    teacher_n, _ = P.project_teacher(coords, feats_map, Kc, Kd, poses_noisy, depth_cache, device)
    teacher_r, _ = P.project_teacher(coords, feats_map, Kc, Kd, poses_ref, depth_cache, device)
    m_clean = eval_standard(teacher_clean.cpu().numpy(), labels20, weights, observed, 0, 300)
    m_noisy = eval_standard(teacher_n.cpu().numpy(), labels20, weights, observed, 0, 300)
    m_ref = eval_standard(teacher_r.cpu().numpy(), labels20, weights, observed, 0, 300)

    row = {"scene": sc, "clean": float(m_clean), "noisy_perframe": float(m_noisy), "refined": float(m_ref)}
    rows.append(row)
    json.dump(rows, open(OUT_SUMMARY, "w"), indent=2)
    print(f"  [{len(rows)}/{len(scenes)}] {sc} ({time.time()-t0:.0f}s): clean={m_clean:.4f} noisy_pf={m_noisy:.4f} refined={m_ref:.4f}", flush=True)

print("PERFRAME_DONE", flush=True)
