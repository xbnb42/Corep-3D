#!/usr/bin/env python3
"""311-scene teacher_refined redo with no_weight refinement (mechanism A, correct).

Reuses existing ms_<scene>/best.pt checkpoints; only re-runs the eval with
no_weight pose refinement (sem_reliability=False, sem_conf=False).
Same seed=0 noise as the original run => clean/noisy identical, only refined changes.
"""
import subprocess, json, os, time

CACHE = "/root/autodl-tmp/posec_cache"
FRAMES = "/root/autodl-tmp/pointdc_data/ScanNet/frames"
LIST = "/root/scannetv2_val.txt"
SUMMARY_PATH = f"{CACHE}/refine_noweight_summary.json"
PY = "/root/miniconda3/bin/python"

scenes = [l.strip() for l in open(LIST) if l.strip()]
done = {}
if os.path.exists(SUMMARY_PATH):
    try:
        for r in json.load(open(SUMMARY_PATH)):
            done[r.get("scene")] = r
    except Exception:
        pass
summary = list(done.values())
print(f"total {len(scenes)} scenes, done {len(done)}", flush=True)

for idx, sc in enumerate(scenes):
    if sc in done:
        continue
    npz = f"{CACHE}/{sc}_dino384_20cls.npz"
    feats = f"{CACHE}/{sc}_feats_map.pt"
    frames = f"{FRAMES}/{sc}"
    ckpt = f"{CACHE}/ms_{sc}/best.pt"
    out = f"{CACHE}/ms_{sc}_nw"
    if not (os.path.exists(npz) and os.path.exists(feats) and os.path.exists(ckpt)):
        continue
    print(f"===== [{idx+1}/{len(scenes)}] {sc} =====", flush=True)
    t0 = time.time()
    with open(f"{CACHE}/ms_{sc}_nw.log", "w") as f:
        r = subprocess.run([PY, "/root/scannet_eval_noweight.py",
                            "--npz", npz, "--feats", feats, "--frames", frames,
                            "--ckpt", ckpt, "--out-dir", out],
                           stdout=f, stderr=subprocess.STDOUT)
    try:
        j = json.load(open(f"{out}/standard_protocol.json"))
        row = {"scene": sc,
               "teacher_clean_miou": j["teacher_clean"]["miou"],
               "teacher_noisy_miou": j["teacher_noisy"]["miou"],
               "teacher_refined_nw_miou": j["teacher_refined"]["miou"],
               "pose_refined_deg": j["pose_refined"]["deg"]}
    except Exception as e:
        row = {"scene": sc, "error": f"rc={r.returncode} {e}"}
    summary.append(row)
    json.dump(summary, open(SUMMARY_PATH, "w"), indent=2)
    print(f"  {sc} ({time.time()-t0:.0f}s): refined_nw={row.get('teacher_refined_nw_miou')}", flush=True)

def mean(key):
    vals = [r[key] for r in summary if key in r]
    return sum(vals)/len(vals) if vals else float('nan')

ok = [r for r in summary if "teacher_refined_nw_miou" in r]
print("\n===== REFINE_NOWEIGHT SUMMARY =====", flush=True)
print(f"done: {len(ok)}/{len(scenes)}", flush=True)
print(f"mean clean     = {mean('teacher_clean_miou'):.4f}", flush=True)
print(f"mean noisy     = {mean('teacher_noisy_miou'):.4f}", flush=True)
print(f"mean refined_nw= {mean('teacher_refined_nw_miou'):.4f}", flush=True)
print("NOWEIGHT_DONE", flush=True)
