#!/usr/bin/env python3
"""P3 noise-curve driver (CoRep Mechanism A, no-weight refinement).

Sweeps pose noise magnitude and records frozen-teacher mIoU under
clean / noisy / refined for a fixed scene subset. Clean and (5deg,11cm)
are reused from the P4 summary (refine_noweight_summary.json); this
driver only runs the NEW levels: (2,11), (8,11), (5,5), (5,17).
"""
import subprocess, json, os, time

CACHE = "/root/autodl-tmp/posec_cache"
FRAMES = "/root/autodl-tmp/pointdc_data/ScanNet/frames"
PY = "/root/miniconda3/bin/python"
EVAL = "/root/scannet_eval_noweight.py"
LIST = "/root/scannetv2_val.txt"
P4_SUMMARY = f"{CACHE}/refine_noweight_summary.json"
OUT_SUMMARY = f"{CACHE}/p3_noise_curve_summary.json"

N_SCENES = 20
NOISE_LEVELS = [(2.0, 11.0), (8.0, 11.0), (5.0, 5.0), (5.0, 17.0)]

scenes = [l.strip() for l in open(LIST) if l.strip()]
scenes = [s for s in scenes if os.path.exists(f"{CACHE}/ms_{s}/best.pt")][:N_SCENES]
print(f"scenes ({len(scenes)}): {scenes}", flush=True)

p4 = {}
try:
    p4 = {r["scene"]: r for r in json.load(open(P4_SUMMARY)) if "teacher_clean_miou" in r}
except Exception as e:
    print("WARN: cannot load P4 summary", e, flush=True)

done = {}
if os.path.exists(OUT_SUMMARY):
    try:
        done = {(r["scene"], r["rot_deg"], r["trans_cm"]): r for r in json.load(open(OUT_SUMMARY))}
    except Exception:
        pass
rows = list(done.values())

for sc in scenes:
    for rot, trans in NOISE_LEVELS:
        if (sc, rot, trans) in done:
            continue
        npz = f"{CACHE}/{sc}_dino384_20cls.npz"
        feats = f"{CACHE}/{sc}_feats_map.pt"
        frames = f"{FRAMES}/{sc}"
        ckpt = f"{CACHE}/ms_{sc}/best.pt"
        out = f"{CACHE}/p3_{sc}_r{rot:g}_t{trans:g}"
        if not (os.path.exists(npz) and os.path.exists(feats) and os.path.exists(ckpt)):
            print(f"  skip {sc} (missing data)", flush=True)
            continue
        t0 = time.time()
        with open(f"{out}.log", "w") as f:
            r = subprocess.run([PY, EVAL, "--npz", npz, "--feats", feats, "--frames", frames,
                                "--ckpt", ckpt, "--out-dir", out,
                                "--rot-deg", str(rot), "--trans-cm", str(trans)],
                               stdout=f, stderr=subprocess.STDOUT)
        try:
            j = json.load(open(f"{out}/standard_protocol.json"))
            row = {"scene": sc, "rot_deg": rot, "trans_cm": trans,
                   "clean": p4.get(sc, {}).get("teacher_clean_miou"),
                   "noisy": j["teacher_noisy"]["miou"],
                   "refined": j["teacher_refined"]["miou"],
                   "pose_refined_deg": j["pose_refined"]["deg"]}
        except Exception as e:
            row = {"scene": sc, "rot_deg": rot, "trans_cm": trans,
                   "error": f"rc={r.returncode} {e}"}
        rows.append(row)
        json.dump(rows, open(OUT_SUMMARY, "w"), indent=2)
        print(f"  {sc} r{rot:g} t{trans:g} ({time.time()-t0:.0f}s): "
              f"noisy={row.get('noisy')} refined={row.get('refined')}", flush=True)

print("P3_DONE", flush=True)
