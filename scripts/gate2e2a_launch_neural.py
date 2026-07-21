#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, subprocess, time
from pathlib import Path
from gate2e2a_common import ROOT, config, write_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3,4,5")
    args = ap.parse_args()
    gpus = args.gpus.split(",")
    cfg = config()
    all_jobs = [(p, f, a, s) for p in cfg["protocols"] for f in range(5) for a in cfg["arms"] for s in cfg["seeds"]]
    jobs, done, failed, active = [], [], [], {}
    for job in all_jobs:
        p, f, arm, seed = job
        summary = ROOT / cfg["local_root"] / "neural" / p / f"fold{f}" / arm / f"seed{seed}" / "summary.json"
        (done if summary.exists() else jobs).append({"job": job, "status": "completed_before_resume"} if summary.exists() else job)
    while jobs or active:
        for gpu in [g for g in gpus if g not in active]:
            if not jobs:
                break
            p, fold, arm, seed = job = jobs.pop(0)
            log = ROOT / cfg["local_root"] / "launch_logs" / f"{p}_f{fold}_{arm}_s{seed}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            handle = log.open("w")
            env = os.environ.copy()
            env.update({"CUDA_VISIBLE_DEVICES": gpu, "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONPATH": str(ROOT / "scripts")})
            cmd = ["/home/changliu/miniconda3/envs/ML/bin/python", "scripts/gate2e2a_train_fold.py", "--protocol", p, "--fold", str(fold), "--arm", arm, "--seed", str(seed), "--physical-gpu", gpu]
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
            active[gpu] = (proc, job, log, handle)
        time.sleep(2)
        for gpu, (proc, job, log, handle) in list(active.items()):
            rc = proc.poll()
            if rc is None:
                continue
            handle.close()
            item = {"job": job, "gpu": gpu, "pid": proc.pid, "returncode": rc, "log": str(log.relative_to(ROOT))}
            (done if rc == 0 else failed).append(item)
            del active[gpu]
        write_json("logs/gate2e2a_launch_state.json", {"pending": len(jobs), "active": [{"gpu": g, "pid": v[0].pid, "job": v[1]} for g, v in active.items()], "done": done, "failed": failed, "explicit_free_gpu_pool": True})
    write_json("logs/gate2e2a_launch_state.json", {"pending": 0, "active": [], "done": done, "failed": failed, "explicit_free_gpu_pool": True})
    if failed:
        raise RuntimeError(f"{len(failed)} neural jobs failed")

if __name__ == "__main__":
    main()
