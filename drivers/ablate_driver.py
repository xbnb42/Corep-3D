#!/usr/bin/env python3
"""Component-ablation driver (expand 6 -> 20 scenes).

Runs ablate_refine_oneshot.py (6 variants per scene) on the first 20 scenes
with a clean-trained ms_ checkpoint. Waits for P3 noise-curve to finish first.
"""
import subprocess, json, os, time

CACHE = "/root/autodl-tmp/posec_cache"
FRAMES = "/root/autodl-tmp/pointdc_data/ScanNet/frames"
PY = "/root/miniconda3/bin/python"
ABLATE = "/root/ablate_refine_oneshot.py"
LIST = "/root/scannetv2_val.txt"
P3_LOG = "/root/p3_noise_curve.log"
OUT_SUMMARY = f"{CACHE}/ablate_summary.json"

N_SCENES = 20

# wait for P3
while not os.path.exists(P3_LOG) or "P3_DONE" not in open(P3_LOG, errors="ignore").read():
    print("waiting for P3_DONE ...", flush=True)
    time.sleep(180)

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
    out = f"{CACHE}/ablate_{sc}.json"
    if not (os.path.exists(npz) and os.path.exists(feats) and os.path.exists(ckpt)):
        print(f"  skip {sc} (missing data)", flush=True)
        continue
    t0 = time.time()
    with open(f"{CACHE}/ablate_{sc}.log", "w") as f:
        r = subprocess.run([PY, ABLATE, "--npz", npz, "--feats", feats, "--frames", frames,
                            "--ckpt", ckpt, "--out", out], stdout=f, stderr=subprocess.STDOUT)
    try:
        j = json.load(open(out))
        row = {"scene": sc, "clean": j["teacher_clean_miou"], "noisy": j["teacher_noisy_miou"],
               "variants": j["variants"]}
    except Exception as e:
        row = {"scene": sc, "error": f"rc={r.returncode} {e}"}
    rows.append(row)
    json.dump(rows, open(OUT_SUMMARY, "w"), indent=2)
    v = row.get("variants", {})
    print(f"  [{len(rows)}/{len(scenes)}] {sc} ({time.time()-t0:.0f}s): "
          f"geom_only={v.get('geom_only',{}).get('teacher_refined_miou')} "
          f"sem_only={v.get('sem_only',{}).get('teacher_refined_miou')} "
          f"no_weight={v.get('no_weight',{}).get('teacher_refined_miou')}", flush=True)

print("ABLATE_DONE", flush=True)
