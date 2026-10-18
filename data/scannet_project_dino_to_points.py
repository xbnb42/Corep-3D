#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np
import torch
from plyfile import PlyData


def read_ply_xyzrgb(path):
    ply = PlyData.read(path)['vertex']
    xyz = np.stack([ply['x'], ply['y'], ply['z']], axis=1).astype(np.float32)
    rgb = np.stack([ply['red'], ply['green'], ply['blue']], axis=1).astype(np.float32)
    return xyz, rgb


def read_labels(path):
    ply = PlyData.read(path)['vertex']
    key = 'label' if 'label' in ply.data.dtype.names else 'class'
    return np.asarray(ply[key]).astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('scene', default='scene0001_00', nargs='?')
    ap.add_argument('--pointcloud', default='')
    ap.add_argument('--labels', default='')
    ap.add_argument('--feats', default='')
    ap.add_argument('--frames', default='')
    ap.add_argument('--out', default='')
    ap.add_argument('--posec-dir', default='/root/posec_impl')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    data = '/root/autodl-tmp/pointdc_data/ScanNet'
    pointcloud = args.pointcloud or f'{data}/raw/scans/{args.scene}/{args.scene}_vh_clean_2.ply'
    labels_path = args.labels or f'{data}/raw/scans/{args.scene}/{args.scene}_vh_clean_2.labels.ply'
    feats_path = args.feats or f'/root/autodl-tmp/posec_cache/{args.scene}_feats_map.pt'
    frames = args.frames or f'{data}/frames/{args.scene}'
    out_path = args.out or f'/root/autodl-tmp/posec_cache/{args.scene}_dino384.npz'

    sys.path.insert(0, args.posec_dir)
    from posec.geom import compute_visibility, depth_residual
    from posec.teacher import sample_patch

    coords, colors = read_ply_xyzrgb(pointcloud)
    labels = read_labels(labels_path)
    if len(coords) != len(labels):
        raise RuntimeError(f'point/label mismatch: {len(coords)} vs {len(labels)}')

    feats_map = torch.load(feats_path, map_location='cpu')
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    xyz = torch.from_numpy(coords).to(device)
    xyz1 = torch.cat([xyz, torch.ones((xyz.shape[0], 1), device=device)], dim=1)

    Kc = np.loadtxt(os.path.join(frames, 'intrinsic_color.txt'))[:3, :3]
    Kd = np.loadtxt(os.path.join(frames, 'intrinsic_depth.txt'))[:3, :3]
    feat_sum = torch.zeros((xyz.shape[0], 384), device=device)
    weight_sum = torch.zeros(xyz.shape[0], device=device)
    visible_frames = torch.zeros(xyz.shape[0], dtype=torch.int32, device=device)

    for i, (name, fp, scale) in enumerate(feats_map):
        pose_path = os.path.join(frames, 'pose', name[:-4] + '.txt')
        depth_path = os.path.join(frames, 'depth', name[:-4] + '.png')
        if not os.path.exists(pose_path) or not os.path.exists(depth_path):
            print(f'skip frame {i}: missing {name}')
            continue
        import cv2
        pose = np.loadtxt(pose_path)
        Tinv = np.linalg.inv(pose)
        cam = (torch.from_numpy(Tinv).float().to(device) @ xyz1.t()).t()[:, :3]
        z = cam[:, 2]
        ud = Kd[0, 0] * cam[:, 0] / z.clamp_min(1e-6) + Kd[0, 2]
        vd = Kd[1, 1] * cam[:, 1] / z.clamp_min(1e-6) + Kd[1, 2]
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
        depth_t = torch.from_numpy(depth).to(device)
        H, W = depth.shape
        uv_d = torch.stack([ud, vd], dim=1)
        vis = compute_visibility(uv_d, z, depth_t, H, W)
        rd = depth_residual(z, depth_t, uv_d, H, W)
        uc = Kc[0, 0] * cam[:, 0] / z.clamp_min(1e-6) + Kc[0, 2]
        vc = Kc[1, 1] * cam[:, 1] / z.clamp_min(1e-6) + Kc[1, 2]
        zt = sample_patch(fp.float().to(device), uc, vc, scale).to(device)
        rd = torch.where(torch.isfinite(rd), rd, torch.full_like(rd, 1e3))
        w = vis * torch.exp(-rd / 0.10)
        w = torch.where(vis > 0, w, torch.zeros_like(w))
        feat_sum += w[:, None] * zt
        weight_sum += w
        visible_frames += (vis > 0).to(torch.int32)
        print(f'frame {i}: visible={int((vis > 0).sum())}/{len(coords)}', flush=True)

    feature = feat_sum / weight_sum.clamp_min(1e-8)[:, None]
    feature = torch.nn.functional.normalize(feature, dim=1)
    feature[weight_sum <= 0] = 0
    feature = feature.cpu().numpy().astype(np.float32)

    np.savez_compressed(
        out_path,
        coords=coords,
        colors=colors,
        labels=labels,
        features=feature,
        weight_sum=weight_sum.cpu().numpy(),
        visible_frames=visible_frames.cpu().numpy(),
    )
    print('saved:', out_path)
    print('points:', len(coords), 'feature_dim:', feature.shape[1])
    print('observed ratio:', float((weight_sum > 0).float().mean().cpu()))


if __name__ == '__main__':
    main()
