#!/usr/bin/env python3
"""
Verify 'unweighted refinement > weighted' on val scenes (main-table setting).
Reuses per-scene checkpoints already trained in the 311-scene run.
Runs ablate_refine_oneshot.py per scene, collects a compact summary.
"""
import subprocess, json, os, time

PY = "/root/miniconda3/bin/python"
CACHE = "/root/autodl-tmp/posec_cache"
FRAMES = "/root/autodl-tmp/pointdc_data/ScanNet/frames"
SCENES = ["scene0568_00", "scene0011_00", "scene0019_00", "scene0050_00", "scene0304_00", "scene0015_00"]
SUMMARY = f"{CACHE}/verify_val_ablation.json"

done = {}
if os.path.exists(SUMMARY):
    try:
        done = {r["scene"]: r for r in json.load(open(SUMMARY))}
    except Exception:
        done = {}
summary = list(done.values())

for i, sc in enumerate(SCENES):
    if sc in done:
        print(f"[{i+1}/{len(SCENES)}] skip {sc}", flush=True)
        continue
    npz = f"{CACHE}/{sc}_dino384_20cls.npz"
    feats = f"{CACHE}/{sc}_feats_map.pt"
    ckpt = f"{CACHE}/ms_{sc}/best.pt"
    out = f"{CACHE}/verify_{sc}.json"
    if not (os.path.exists(npz) and os.path.exists(feats) and os.path.exists(ckpt)):
        print(f"[{i+1}/{len(SCENES)}] MISS data {sc}", flush=True)
        continue
    print(f"===== [{i+1}/{len(SCENES)}] {sc} =====", flush=True)
    t0 = time.time()
    with open(f"{CACHE}/verify_{sc}.log", "w") as f:
        r = subprocess.run([PY, "/root/ablate_refine_oneshot.py",
                            "--npz", npz, "--feats", feats, "--frames", f"{FRAMES}/{sc}",
                            "--ckpt", ckpt, "--out", out], stdout=f, stderr=subprocess.STDOUT)
    try:
        j = json.load(open(out))
        row = {"scene": sc, "frozen": j["frozen_miou"], "clean": j["teacher_clean_miou"],
               "noisy": j["teacher_noisy_miou"], "variants": j["variants"]}
    except Exception as e:
        row = {"scene": sc, "error": f"rc={r.returncode} {e}"}
    summary.append(row)
    json.dump(summary, open(SUMMARY, "w"), indent=2)
    print(f"  {sc} done ({time.time()-t0:.0f}s): "
          f"no_weight={row.get('variants',{}).get('no_weight',{}).get('teacher_refined_miou')} "
          f"vs full={row.get('variants',{}).get('full',{}).get('teacher_refined_miou')}", flush=True)

print("\n===== VERIFY SUMMARY (no_weight vs full refined mIoU) =====", flush=True)
for r in summary:
    if "variants" in r:
        nw = r["variants"].get("no_weight", {}).get("teacher_refined_miou")
        fl = r["variants"].get("full", {}).get("teacher_refined_miou")
        cf = r["variants"].get("conf_only", {}).get("teacher_refined_miou")
        print(f'{r["scene"]}: clean={r["clean"]:.3f} noisy={r["noisy"]:.3f} | no_weight={nw:.3f} full={fl:.3f} conf_only={cf:.3f}', flush=True)
print("VERIFY_DONE", flush=True)
