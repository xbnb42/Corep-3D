#!/usr/bin/env python3
"""312 场景全量驱动: 每个场景蒸馏300ep + 两段式标准评测, 断点续跑, 逐场景存结果."""
import subprocess, json, os, time

CACHE = "/root/autodl-tmp/posec_cache"
FRAMES = "/root/autodl-tmp/pointdc_data/ScanNet/frames"
LIST = "/root/scannetv2_val.txt"
SUMMARY_PATH = f"{CACHE}/multiscene_all_summary.json"
PY = "/root/miniconda3/bin/python"

# 读 val 列表
scenes = [l.strip() for l in open(LIST) if l.strip()]
print(f"val 场景总数: {len(scenes)}", flush=True)

# 断点续跑: 读已有 summary
done = {}
if os.path.exists(SUMMARY_PATH):
    try:
        for r in json.load(open(SUMMARY_PATH)):
            done[r.get("scene")] = r
    except Exception:
        pass
summary = list(done.values())
print(f"已完成: {len(done)}", flush=True)

for idx, sc in enumerate(scenes):
    if sc in done:
        continue
    npz = f"{CACHE}/{sc}_dino384_20cls.npz"
    feats = f"{CACHE}/{sc}_feats_map.pt"
    frames = f"{FRAMES}/{sc}"
    if not os.path.exists(npz) or not os.path.exists(feats):
        print(f"[{idx+1}/{len(scenes)}] skip {sc} (npz/feats 未就绪)", flush=True)
        continue

    out = f"{CACHE}/ms_{sc}"
    log = f"{CACHE}/ms_{sc}.log"
    print(f"===== [{idx+1}/{len(scenes)}] {sc} 蒸馏训练 =====", flush=True)
    t0 = time.time()
    with open(log, "w") as f:
        r = subprocess.run([PY, "/root/scannet_posec_train_384.py",
            "--npz", npz, "--feats", feats, "--frames", frames,
            "--out-dir", out, "--emb-dim", "128", "--lr", "3e-3", "--lr-decay",
            "--epochs", "300", "--eval-every", "10", "--lambda-proto", "0"],
            stdout=f, stderr=subprocess.STDOUT)
    print(f"  训练返回 {r.returncode} ({time.time()-t0:.0f}s)", flush=True)

    if r.returncode != 0:
        row = {"scene": sc, "error": f"train rc={r.returncode}"}
        summary.append(row)
        json.dump(summary, open(SUMMARY_PATH, "w"), indent=2)
        print(f"  {sc}: {row}", flush=True)
        continue

    print(f"===== {sc} 两段式标准评测 =====", flush=True)
    elog = f"{CACHE}/ms_{sc}_eval.log"
    t0 = time.time()
    with open(elog, "w") as f:
        r = subprocess.run([PY, "/root/scannet_eval_standard_384.py",
            "--npz", npz, "--feats", feats, "--frames", frames,
            "--ckpt", f"{out}/best.pt", "--out-dir", f"{out}_eval"],
            stdout=f, stderr=subprocess.STDOUT)
    print(f"  评测返回 {r.returncode} ({time.time()-t0:.0f}s)", flush=True)

    jpath = f"{out}_eval/standard_protocol.json"
    try:
        j = json.load(open(jpath))
        row = {"scene": sc,
               "frozen_miou": j["frozen_feat"]["miou"],
               "teacher_clean_miou": j["teacher_clean"]["miou"],
               "teacher_noisy_miou": j["teacher_noisy"]["miou"],
               "teacher_refined_miou": j["teacher_refined"]["miou"],
               "pose_refined_deg": j["pose_refined"]["deg"]}
    except Exception as e:
        row = {"scene": sc, "error": f"eval: {e}"}
    summary.append(row)
    json.dump(summary, open(SUMMARY_PATH, "w"), indent=2)
    print(f"  {sc}: {row}", flush=True)

# 最终汇总
def mean(key):
    vals = [r[key] for r in summary if key in r]
    return sum(vals) / len(vals) if vals else float("nan")

ok = [r for r in summary if "frozen_miou" in r]
print("\n===== 全量标准协议汇总 =====", flush=True)
print(f"成功场景数: {len(ok)} / {len(scenes)}", flush=True)
print(f"平均 frozen mIoU  = {mean('frozen_miou'):.4f}", flush=True)
print(f"平均 noisy   mIoU  = {mean('teacher_noisy_miou'):.4f}", flush=True)
print(f"平均 refined mIoU  = {mean('teacher_refined_miou'):.4f}", flush=True)
print("MULTISCENE_ALL_DONE", flush=True)
