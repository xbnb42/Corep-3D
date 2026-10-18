#!/usr/bin/env python3
"""B / A+B full-scale driver (all scenes with a clean-trained ms_ checkpoint).

Runs correspondence_repair.py at 5deg/11cm on every scene that has a ms_
checkpoint. Extracts: clean / noisy / A (refined_no_weight) /
B (norefine_soft) / A+B (repaired_soft).

Waits for the component-ablation run (ablate.log ABLATE_DONE) before starting,
so the two jobs never contend for the GPU.
"""
import subprocess, json, os, time

CACHE = "/root/autodl-tmp/posec_cache"
FRAMES = "/root/autodl-tmp/pointdc_data/ScanNet/frames"
PY = "/root/miniconda3/bin/python"
REPAIR = "/root/correspondence_repair.py"
LIST = "/root/scannetv2_val.txt"
ABLATE_LOG = "/root/ablate.log"
OUT_SUMMARY = f"{CACHE}/ab_repair_summary.json"

# ---- wait for P3 to finish ----
while not os.path.exists(ABLATE_LOG) or "ABLATE_DONE" not in open(ABLATE_LOG, errors="ignore").read():
    print("waiting for ABLATE_DONE ...", flush=True)
    time.sleep(180)

scenes = [l.strip() for l in open(LIST) if l.strip()]
scenes = [s for s in scenes if os.path.exists(f"{CACHE}/ms_{s}/best.pt")]
print(f"scenes ({len(scenes)})", flush=True)

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
    out = f"{CACHE}/ab_{sc}.json"
    if not (os.path.exists(npz) and os.path.exists(feats) and os.path.exists(ckpt)):
        print(f"  skip {sc} (missing data)", flush=True)
        continue
    t0 = time.time()
    with open(f"{CACHE}/ab_{sc}.log", "w") as f:
        r = subprocess.run([PY, REPAIR, "--npz", npz, "--feats", feats, "--frames", frames,
                            "--ckpt", ckpt, "--out", out], stdout=f, stderr=subprocess.STDOUT)
    try:
        j = json.load(open(out))
        row = {"scene": sc, "clean": j["clean"], "noisy": j["noisy"],
               "A_refined": j["refined_no_weight"], "B_norefine": j["norefine_soft"],
               "AB_repaired": j["repaired_soft"]}
    except Exception as e:
        row = {"scene": sc, "error": f"rc={r.returncode} {e}"}
    rows.append(row)
    json.dump(rows, open(OUT_SUMMARY, "w"), indent=2)
    print(f"  [{len(rows)}/{len(scenes)}] {sc} ({time.time()-t0:.0f}s): "
          f"clean={row.get('clean')} noisy={row.get('noisy')} A={row.get('A_refined')} "
          f"B={row.get('B_norefine')} AB={row.get('AB_repaired')}", flush=True)

print("AB_DONE", flush=True)
