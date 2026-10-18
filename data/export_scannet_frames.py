#!/usr/bin/env python3
"""Export RGB, depth, pose and intrinsics from a ScanNet .sens file.

Run after cloning ScanNet/SensReader:
  python export_scannet_frames.py scene0001_00 --sens-root /root/autodl-tmp/pointdc_data/ScanNet/scans --out-root /root/autodl-tmp/pointdc_data/ScanNet/frames --frame-skip 30
"""
import argparse
import os
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('scene')
    p.add_argument('--sens-root', default='/root/autodl-tmp/pointdc_data/ScanNet/scans')
    p.add_argument('--out-root', default='/root/autodl-tmp/pointdc_data/ScanNet/frames')
    p.add_argument('--sens-reader', default='/root/ScanNet/SensReader/python')
    p.add_argument('--sensor-module', default='SensorData_py3')
    p.add_argument('--frame-skip', type=int, default=30)
    args = p.parse_args()

    sys.path.insert(0, args.sens_reader)
    module = __import__(args.sensor_module)
    SensorData = module.SensorData

    sens = os.path.join(args.sens_root, args.scene, args.scene + '.sens')
    out = os.path.join(args.out_root, args.scene)
    if not os.path.exists(sens):
        raise FileNotFoundError(sens)
    os.makedirs(out, exist_ok=True)

    sd = SensorData(sens)
    sd.export_color_images(os.path.join(out, 'color'), frame_skip=args.frame_skip)
    sd.export_depth_images(os.path.join(out, 'depth'), frame_skip=args.frame_skip)
    sd.export_poses(os.path.join(out, 'pose'), frame_skip=args.frame_skip)
    sd.export_intrinsics(out)

    n = len(os.listdir(os.path.join(out, 'color')))
    print(f'exported {n} frames -> {out}')


if __name__ == '__main__':
    main()
