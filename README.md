# CoRep: Correspondence Repair for Pose-Robust Unsupervised 3D Segmentation

Official code for *"Pose Noise in Unsupervised 2D-to-3D Distillation: A Diagnosis and a Minimal Repair"*.

Camera pose noise corrupts the pixel-to-point correspondence that supplies the training targets in unsupervised 2D-to-3D distillation, and the projected features collapse (frozen-teacher mIoU drops by **56.7%** at 5 deg / 11 cm on ScanNet-v2). We show that (i) reweighting cannot repair this failure, and (ii) pose refinement alone, even a standard ICP baseline, leaves a residual error from occluded and boundary correspondences. CoRep repairs this residual at the **sampling end**: it relocates the image sampling within a bounded window and discards bad correspondences through an explicit unmatched (null) state, recovering to (and slightly beyond) the clean baseline.

## Repository layout

```
train/        student training (sparse-conv backbone) and the backbone module
eval/         frozen-teacher evaluation (clean / noisy / refined)
repair/       Mechanism A (pose refinement) + Mechanism B (candidate search + null)
drivers/      multi-scene drivers for the main table, dual mechanism, noise curve, ablations
ablations/    component ablation, K sensitivity, ICP baseline, per-frame noise, multi-seed
data/         ScanNet frame export, DINO feature extraction, projection, label remapping
```

## Environment

- Python 3.8 / 3.12, PyTorch >= 1.10, CUDA
- `pip install -r requirements.txt`
- spconv: install the wheel matching your CUDA version (`spconv-cu113` / `spconv-cu118`, or build from source)
- DINOv2 teacher is loaded through `torch.hub` (`facebookresearch/dinov2`, ViT-S/14)

## Data preparation

Each scene goes through four steps (scripts in `data/`):

```bash
# 1. Export RGB / depth / pose / intrinsics from the .sens file (needs ScanNet/SensReader)
python data/export_scannet_frames.py <scene> --sens-root <scans> --out-root <frames> --frame-skip 30

# 2. Extract frozen DINOv2 (ViT-S/14) features from the color frames
python data/extract_dino_only.py <scene> --frames-root <frames> --cache <cache>

# 3. Project per-pixel features onto points (reads the .ply + labels, writes an npz)
python data/scannet_project_dino_to_points.py <scene> --frames-root <frames> --cache <cache>

# 4. Remap raw 40-class labels to the 20-class protocol
python data/remap_scannet_labels_v2.py --npz <cache>/<scene>_dino384.npz --out <cache>/<scene>_dino384_20cls.npz
```

The final `<scene>_dino384_20cls.npz` contains `coords`, `colors`, `labels20`, `weight_sum`.

## Reproduction

### 1. Train the student (clean poses)

```bash
python train/scannet_posec_train_384.py \
  --npz <cache>/<scene>_dino384_20cls.npz \
  --feats <cache>/<scene>_feats_map.pt \
  --frames <frames>/<scene> \
  --out-dir <cache>/ms_<scene> \
  --emb-dim 128 --lr 3e-3 --lr-decay --epochs 300 --eval-every 10 --lambda-proto 0
```

This is a distill-only run on clean poses; the student consumes only point coordinates and colors.

### 2. Evaluate under pose noise (frozen teacher)

```bash
python eval/scannet_eval_noweight.py \
  --npz <cache>/<scene>_dino384_20cls.npz \
  --feats <cache>/<scene>_feats_map.pt \
  --frames <frames>/<scene> \
  --ckpt <cache>/ms_<scene>/best.pt \
  --out-dir <cache>/ms_<scene>_nw --rot-deg 5 --trans-cm 11
```

Reports `teacher_clean`, `teacher_noisy`, `teacher_refined` (Mechanism A, no reliability weighting) and the refined pose error.

### 3. Repair the correspondence (Mechanism A + B)

```bash
python repair/correspondence_repair.py \
  --npz <cache>/<scene>_dino384_20cls.npz \
  --feats <cache>/<scene>_feats_map.pt \
  --frames <frames>/<scene> \
  --ckpt <cache>/ms_<scene>/best.pt \
  --out <cache>/ab_<scene>.json
```

Outputs `clean`, `noisy`, `refined_no_weight` (A), `norefine_soft` (B), `repaired_soft` (A+B).

### 4. Multi-scene drivers

- Main table (311 scenes, A only): `python drivers/eval_311_noweight.py`
- Dual mechanism (311 scenes, A/B/A+B): `python drivers/ab_repair_driver.py`
- Noise curve (20 scenes): `python drivers/p3_noise_curve_driver.py`
- Component ablation (20 scenes): `python drivers/ablate_driver.py`

### 5. Ablations

- Component ablation: `python ablations/ablate_refine_oneshot.py`
- K sensitivity (K in {100,200,300,500}): `python ablations/k_sensitivity.py`
- Standard ICP baseline: `python ablations/icp_baseline.py`
- Per-frame drift: `python ablations/perframe_noise.py`
- Multi-seed robustness: `python ablations/multiseed.py`

## Main results (ScanNet-v2, 311 scenes, 20-class protocol)

| Setting | frozen-teacher mIoU | recovery |
| --- | --- | --- |
| clean | 0.2494 | -- |
| noisy (5 deg / 11 cm) | 0.1081 | -- |
| A (pose refinement) | 0.2274 | 84.4% |
| B (candidate search) | 0.1556 | 33.7% |
| A+B | 0.2522 | 102% |

A standard point-to-point ICP baseline recovers 90% of the gap; A+B reaches (and slightly exceeds) the clean baseline, with the gain coming from the null state dropping correspondences that no camera pose can correct.

## Citation

```bibtex
@article{corep2026,
  title={Pose Noise in Unsupervised 2D-to-3D Distillation: A Diagnosis and a Minimal Repair},
  author={...},
  year={2026}
}
```
