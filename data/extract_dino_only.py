#!/usr/bin/env python3
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('scene')
    p.add_argument('--frames-root', default='/root/autodl-tmp/pointdc_data/ScanNet/frames')
    p.add_argument('--cache', default='/root/autodl-tmp/posec_cache')
    p.add_argument('--n-frames', type=int, default=12)
    p.add_argument('--posec-dir', default='/root/posec_impl')
    args = p.parse_args()

    sys.path.insert(0, args.posec_dir)
    from posec.teacher import load_teacher, patch_features

    frames = os.path.join(args.frames_root, args.scene)
    cols = sorted(
        glob.glob(os.path.join(frames, 'color', '*.jpg')),
        key=lambda x: int(os.path.basename(x).split('.')[0]),
    )
    if not cols:
        raise FileNotFoundError(os.path.join(frames, 'color', '*.jpg'))

    valid = []
    for c in cols:
        pose = os.path.join(frames, 'pose', os.path.basename(c)[:-4] + '.txt')
        try:
            mat = np.loadtxt(pose)
        except Exception:
            continue
        if np.isfinite(mat).all():
            valid.append(c)
    cols = valid
    if not cols:
        raise RuntimeError('no valid color/pose frames')

    step = max(1, len(cols) // args.n_frames)
    sel = cols[::step][:args.n_frames]
    teacher = load_teacher()
    os.makedirs(args.cache, exist_ok=True)
    out_list = []
    for i, cp in enumerate(sel):
        rgb = cv2.imread(cp)[:, :, ::-1]
        feat_patch, scale = patch_features(teacher, rgb)
        out_list.append((os.path.basename(cp), feat_patch.float(), scale))
        print(f'frame {i}: {os.path.basename(cp)} patch={tuple(feat_patch.shape)}', flush=True)

    out_path = os.path.join(args.cache, args.scene + '_feats_map.pt')
    torch.save(out_list, out_path)
    print('saved:', out_path)


if __name__ == '__main__':
    main()
