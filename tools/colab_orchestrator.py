#!/usr/bin/env python3
"""Unattended orchestration of VOUCH benchmark runs on ephemeral Colab GPUs.

Colab sessions on free/limited accounts are reclaimed after ~1 h; runs are
therefore structured as resumable jobs (method-level partial saves +
--resume) and this orchestrator loops:

  provision session -> stage code + prior results -> launch job --resume
  -> poll, downloading results every cycle -> on session death, reprovision
  and resume -> on job completion, advance to the next job in the queue.

Usage:  python3 tools/colab_orchestrator.py
Queue and paths are configured below.  State survives orchestrator restarts
because all progress lives in the downloaded results JSONs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SESSION = "vouch"
TARBALL = "/tmp/vouch_orch.tar.gz"
VMDIR = "/content/vouchB"

JOBS = [
    {
        "name": "tofu",
        "cmd": (f"cd {VMDIR} && nohup python experiments/run_benchmark.py "
                "--dataset tofu --seeds 0 1 2 --resume --dtype fp32 --pairs 512 "
                "> /content/bench_tofu.log 2>&1 &"),
        "log": "/content/bench_tofu.log",
        "result": f"{VMDIR}/results/lm_e2e_tofu_phi-1_5.json",
        "local": f"{REPO}/results/lm_e2e_tofu_phi-1_5.json",
        "seeds": 3, "methods": 7,
    },
    {
        "name": "muse",
        "cmd": (f"cd {VMDIR} && nohup python experiments/run_benchmark.py "
                "--dataset muse --seeds 0 1 2 --resume --dtype fp32 --pairs 512 "
                "> /content/bench_muse.log 2>&1 &"),
        "log": "/content/bench_muse.log",
        "result": f"{VMDIR}/results/lm_e2e_muse_phi-1_5.json",
        "local": f"{REPO}/results/lm_e2e_muse_phi-1_5.json",
        "seeds": 3, "methods": 7,
    },
    {
        "name": "qwen3_06b",
        "cmd": (f"cd {VMDIR} && nohup python experiments/run_lm_big.py "
                "--model Qwen/Qwen3-0.6B --seeds 0 --pairs 384 "
                "--eps 0.2 --dtype fp32 --train-steps 400 --batch 4 "
                "--queries 2 --tag qwen3_06b "
                "--methods none retrain npo npo_P3_jailbreak "
                "> /content/bench_qwen3_06b.log 2>&1 &"),
        "log": "/content/bench_qwen3_06b.log",
        "result": f"{VMDIR}/results/lm_e2e_qwen3_06b.json",
        "local": f"{REPO}/results/lm_e2e_qwen3_06b.json",
        "seeds": 1, "methods": 4,
    },
    {
        "name": "qwen3_4b",
        "cmd": (f"cd {VMDIR} && nohup python experiments/run_lm_big.py "
                "--model Qwen/Qwen3-4B-Instruct-2507 --seeds 0 --pairs 384 "
                "--eps 0.2 --dtype bf16 --train-steps 400 --batch 4 "
                "--queries 2 --tag qwen3_4b "
                "--methods none retrain npo npo_P3_jailbreak "
                "> /content/bench_qwen3_4b.log 2>&1 &"),
        "log": "/content/bench_qwen3_4b.log",
        "result": f"{VMDIR}/results/lm_e2e_qwen3_4b.json",
        "local": f"{REPO}/results/lm_e2e_qwen3_4b.json",
        "seeds": 1, "methods": 4,
    },
    {
        "name": "nemotron3_4b",
        "cmd": (f"cd {VMDIR} && nohup python experiments/run_lm_big.py "
                "--model nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 --seeds 0 --pairs 384 "
                "--eps 0.2 --dtype bf16 --train-steps 400 --batch 4 "
                "--queries 2 --tag nemotron3_4b "
                "--methods none retrain npo npo_P3_jailbreak "
                "> /content/bench_nemotron3_4b.log 2>&1 &"),
        "log": "/content/bench_nemotron3_4b.log",
        "result": f"{VMDIR}/results/lm_e2e_nemotron3_4b.json",
        "local": f"{REPO}/results/lm_e2e_nemotron3_4b.json",
        "seeds": 1, "methods": 4,
    },
    {
        "name": "gemma4",
        "cmd": (f"cd {VMDIR} && nohup python experiments/run_lm_big.py "
                "--model google/gemma-4-E2B-it --seeds 0 --pairs 384 "
                "--eps 0.2 --dtype bf16 --train-steps 400 --batch 4 "
                "--queries 2 --tag gemma4 "
                "--methods none retrain npo npo_P3_jailbreak "
                "> /content/bench_gemma4.log 2>&1 &"),
        "log": "/content/bench_gemma4.log",
        "result": f"{VMDIR}/results/lm_e2e_gemma4.json",
        "local": f"{REPO}/results/lm_e2e_gemma4.json",
        "seeds": 1, "methods": 4,
    },
]


def sh(cmd, timeout=240):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                           timeout=timeout)
        return r.returncode, (r.stdout + r.stderr)
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


def colab_exec(py, timeout=240):
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(py)
        path = f.name
    try:
        return sh(f"colab exec -s {SESSION} -f {path}", timeout=timeout)
    finally:
        os.unlink(path)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def session_alive() -> bool:
    rc, out = colab_exec("print('PING_OK')", timeout=120)
    return "PING_OK" in out


def provision() -> bool:
    for attempt in range(24):  # up to ~2 h of retries
        rc, out = sh(f"colab new -s {SESSION} --gpu T4", timeout=300)
        if "READY" in out:
            log(f"session ready (attempt {attempt + 1})")
            return True
        log(f"provision attempt {attempt + 1} failed")
        time.sleep(300)
    return False


def stage(job) -> bool:
    sh(f"cd {REPO} && tar czf {TARBALL} --exclude=.git --exclude=results .")
    rc, out = sh(f"colab upload -s {SESSION} {TARBALL} vouch_orch.tar.gz",
                 timeout=600)
    if rc != 0:
        return False
    setup = f"""
import subprocess, os
src = '/vouch_orch.tar.gz' if os.path.exists('/vouch_orch.tar.gz') else '/content/vouch_orch.tar.gz'
os.makedirs('{VMDIR}', exist_ok=True)
subprocess.run(['tar','xzf',src,'-C','{VMDIR}'])
os.makedirs('{VMDIR}/results', exist_ok=True)
subprocess.run(['pip','uninstall','-y','torchao'], capture_output=True)
r = subprocess.run(['pip','install','-q','peft','datasets'], capture_output=True, text=True)
print('SETUP_DONE', r.returncode)
"""
    rc, out = colab_exec(setup, timeout=600)
    if "SETUP_DONE" not in out:
        log(f"setup failed: {out[-200:]}")
        return False
    if os.path.exists(job["local"]):
        rc, out = sh(f"colab upload -s {SESSION} {job['local']} results_prior.json",
                     timeout=600)
        mv = f"""
import shutil, os
src = '/results_prior.json' if os.path.exists('/results_prior.json') else '/content/results_prior.json'
shutil.copy(src, '{job["result"]}')
print('PRIOR_STAGED')
"""
        colab_exec(mv)
    return True


def job_done(job) -> bool:
    if not os.path.exists(job["local"]):
        return False
    try:
        runs = json.load(open(job["local"]))
    except Exception:
        return False
    complete = [r for r in runs if len(r.get("certs", {})) >= job["methods"]]
    return len(complete) >= job["seeds"]


def run_job(job) -> None:
    log(f"=== job {job['name']} ===")
    while not job_done(job):
        if not session_alive():
            log("no live session; provisioning")
            if not provision():
                log("PROVISIONING FAILED for 2h, giving up this cycle")
                time.sleep(600)
                continue
            if not stage(job):
                log("staging failed; retrying")
                time.sleep(120)
                continue
            colab_exec(f"import subprocess; subprocess.Popen({job['cmd']!r}, shell=True); print('LAUNCHED')")
            log("job launched")
            time.sleep(300)
        # poll + download
        rc, out = colab_exec(
            f"import os; print(open({job['log']!r}).read()[-300:] if os.path.exists({job['log']!r}) else 'NOLOG')")
        if "not found" in out or "lost" in out or rc != 0:
            log("session appears dead; will reprovision")
            continue
        if "NOLOG" in out:
            log("log missing; (re)staging and relaunching job")
            if not stage(job):
                log("staging failed; retrying")
                time.sleep(120)
                continue
            colab_exec(f"import subprocess; subprocess.Popen({job['cmd']!r}, shell=True); print('RELAUNCHED')")
            time.sleep(240)
            continue
        if "Traceback" in out:
            log(f"JOB ERROR:\n{out[-300:]}")
            sh(f"colab download -s {SESSION} {job['log']} {REPO}/results/{job['name']}_error.log",
               timeout=300)
            return
        tail = [l for l in out.splitlines() if l.startswith('[seed')][-1:]
        if tail:
            log(f"progress: {tail[0]}")
        sh(f"colab download -s {SESSION} {job['result']} {job['local']}",
           timeout=600)
        # also check for process still running
        rc2, out2 = colab_exec(
            "import subprocess; print('ALIVE' if subprocess.run(['pgrep','-f','run_'], capture_output=True, text=True).stdout.strip() else 'DEADPROC')")
        if "DEADPROC" in out2 and not job_done(job):
            log("process ended without completing; relaunching with --resume")
            colab_exec(f"import subprocess; subprocess.Popen({job['cmd']!r}, shell=True); print('RELAUNCHED')")
        time.sleep(240)
    log(f"=== job {job['name']} COMPLETE ===")


if __name__ == "__main__":
    only = sys.argv[1:] or [j["name"] for j in JOBS]
    for job in JOBS:
        if job["name"] in only:
            run_job(job)
    log("ALL JOBS COMPLETE")
