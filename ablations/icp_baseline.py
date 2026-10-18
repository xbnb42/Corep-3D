#!/usr/bin/env python3
"""ICP baseline: standard Open3D point-to-plane ICP on noisy poses.

For each scene, perturb poses with 5deg/11cm (seed=0, same as the paper),
correct each frame with Open3D point-to-plane ICP against the scene point
cloud, re-project the frozen teacher, and report mIoU.
"""
import sys, os, json, time
import numpy as np
import torch
import open3d as o3d

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
OUT_SUMMARY = f"{CACHE}/icp_baseline_summary.json"

N_SCENES = 20
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

def run_icp(coords, poses_noisy, depth_cache, Kd, n_src=20000, max_dist=0.2):
    # target = scene point cloud + normals
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(coords.astype(np.float64))
    rng = np.random.RandomState(0)
    poses_icp = {}
    for name, p0 in poses_noisy.items():
        if name not in depth_cache:
            poses_icp[name] = p0; continue
        depth = depth_cache[name].astype(np.float32)
        H, W = depth.shape
        v, u = np.mgrid[0:H, 0:W]
        z = depth
        valid = (z > 0.1) & (z < 10.0)
        u, v, z = u[valid].ravel(), v[valid].ravel(), z[valid].ravel()
        if len(u) < 100:
            poses_icp[name] = p0; continue
        if len(u) > n_src:
            idx = rng.choice(len(u), n_src, replace=False)
            u, v, z = u[idx], v[idx], z[idx]
        Xc = (u - Kd[0,2]) / Kd[0,0] * z
        Yc = (v - Kd[1,2]) / Kd[1,1] * z
        Zc = z
        src = o3d.geometry.PointCloud()
        src.points = o3d.utility.Vector3dVector(np.stack([Xc, Yc, Zc], axis=1).astype(np.float64))
        reg = o3d.registration.registration_icp(
            src, target, max_correspondence_distance=max_dist, init=p0.astype(np.float64),
            criteria=o3d.registration.ICPConvergenceCriteria(max_iteration=50))
        poses_icp[name] = np.asarray(reg.transformation).astype(np.float32)
    return poses_icp

for sc in scenes:
    if sc in done:
        continue
    npz = f"{CACHE}/{sc}_dino384_20cls.npz"
    feats = f"{CACHE}/{sc}_feats_map.pt"
    frames = f"{FRAMES}/{sc}"
    if not (os.path.exists(npz) and os.path.exists(feats) and os.path.exists(frames)):
        print(f"  skip {sc}", flush=True); continue
    t0 = time.time()
    z = np.load(npz)
    coords = z['coords'].astype(np.float32)
    labels20 = z['labels20'].astype(np.int64) if 'labels20' in z else z['labels'].astype(np.int64)
    weights = z['weight_sum'].astype(np.float32)
    observed = (weights > 0) & (labels20 >= 0)
    feats_map = torch.load(feats, map_location='cpu')
    Kc = np.loadtxt(os.path.join(frames, 'intrinsic_color.txt'))[:3,:3]
    Kd = np.loadtxt(os.path.join(frames, 'intrinsic_depth.txt'))[:3,:3]
    depth_cache = P.load_depth_cache(frames, feats_map)
    poses_clean = {}
    for name, _, _ in feats_map:
        pp = os.path.join(frames, 'pose', name[:-4] + '.txt')
        if os.path.exists(pp):
            poses_clean[name] = np.loadtxt(pp)
    # noise (same as paper: seed=0, 5deg/11cm)
    rng = np.random.RandomState(0)
    all_idx = np.where(observed)[0]
    _ = all_idx[rng.choice(len(all_idx), min(3000, len(all_idx)), replace=False)]  # match P4 rng order
    axis = rng.randn(3); tdir = rng.randn(3); tdir /= (np.linalg.norm(tdir) + 1e-8)
    pf = P.make_perturb(axis, tdir)
    poses_noisy = {n: pf(p, 5.0, 0.11) for n, p in poses_clean.items()}
    # teacher clean / noisy
    Kc_t = torch.from_numpy(Kc).float().to(device); Kd_t = torch.from_numpy(Kd).float().to(device)
    teacher_clean, _ = P.project_teacher(coords, feats_map, Kc, Kd, poses_clean, depth_cache, device)
    teacher_n, _ = P.project_teacher(coords, feats_map, Kc, Kd, poses_noisy, depth_cache, device)
    m_clean = eval_standard(teacher_clean.cpu().numpy(), labels20, weights, observed, 0, 300)
    m_noisy = eval_standard(teacher_n.cpu().numpy(), labels20, weights, observed, 0, 300)
    # ICP correction
    poses_icp = run_icp(coords, poses_noisy, depth_cache, Kd)
    teacher_icp, _ = P.project_teacher(coords, feats_map, Kc, Kd, poses_icp, depth_cache, device)
    m_icp = eval_standard(teacher_icp.cpu().numpy(), labels20, weights, observed, 0, 300)
    row = {"scene": sc, "clean": float(m_clean), "noisy": float(m_noisy), "icp_refined": float(m_icp)}
    rows.append(row)
    json.dump(rows, open(OUT_SUMMARY, "w"), indent=2)
    print(f"  [{len(rows)}/{len(scenes)}] {sc} ({time.time()-t0:.0f}s): clean={m_clean:.4f} noisy={m_noisy:.4f} icp={m_icp:.4f}", flush=True)

print("ICP_DONE", flush=True)
