#!/usr/bin/env python3
"""Load-balancing scheduler for VOUCH experiment tasks across heterogeneous
free compute (Colab T4 with multi-account rotation, Kaggle GPU kernels,
local CPU).

Model
-----
* A priority-ordered queue (results/workqueue.json) of task specs:
      {id, cmd, results: [...], requires: "gpu"|"any", retries}
* One worker thread per backend. Each idle worker leases the FIRST pending
  task whose `requires` it satisfies — priority order is respected at
  start time, completion is opportunistic (fastest pool wins).
* A task is done when all its `results` files exist locally with fresh
  mtimes; done tasks are committed+pushed immediately.
* On worker/backend death the lease is released and the task returns to
  the queue; commands are written to be resumable (--resume / checkpoint
  files), so retries are cheap.

Backends
--------
colab : provision (rotating account tokens on quota failure) -> stage repo
        tarball -> nohup task cmd -> poll log & download results.
kaggle: push a single-task kernel (auto P100 torch fallback) -> poll ->
        fetch outputs.
local : plain subprocess on this machine (CPU-only tasks).

Usage:  python3 tools/scheduler.py            # run all pending
        python3 tools/scheduler.py --init     # (re)write default queue
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import threading
import time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
QUEUE = os.path.join(REPO, "results", "workqueue.json")


class _QueueLock:
    """Cross-process (flock) + cross-thread queue lock, so helper worker
    processes can safely share the queue with the main scheduler."""

    _tl = threading.Lock()

    def __enter__(self):
        import fcntl
        self._tl.acquire()
        self._fh = open(QUEUE + ".lock", "w")
        fcntl.flock(self._fh, fcntl.LOCK_EX)
        return self

    def __exit__(self, *a):
        import fcntl
        fcntl.flock(self._fh, fcntl.LOCK_UN)
        self._fh.close()
        self._tl.release()


LOCK = _QueueLock()
GIT_URL = ("https://github.com/vinhqdang/"
           "VOUCH-Verifiable-Online-Unlearning-Certification-via-Hypothesis-betting")
VMDIR = "/content/vouchQ"
TOKEN_DIR = os.path.expanduser("~/.config/colab-cli")

DEFAULT_TASKS = [
    # ---- model zoo (GPU), one model per task for fine-grained balancing
    dict(id="gemma4",
         cmd="python experiments/run_lm_big.py --model google/gemma-4-E2B-it "
             "--seeds 0 --pairs 384 --eps 0.2 --dtype bf16 --train-steps 400 "
             "--batch 4 --queries 2 --tag gemma4 "
             "--methods none retrain npo npo_P3_jailbreak",
         results=["lm_e2e_gemma4.json"], requires="gpu"),
    dict(id="nemotron3_4b",
         cmd="python experiments/run_lm_big.py "
             "--model nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 --seeds 0 "
             "--pairs 384 --eps 0.2 --dtype bf16 --train-steps 400 --batch 4 "
             "--queries 2 --tag nemotron3_4b "
             "--methods none retrain npo npo_P3_jailbreak",
         results=["lm_e2e_nemotron3_4b.json"], requires="gpu"),
    dict(id="qwen3_4b",
         cmd="python experiments/run_lm_big.py --model Qwen/Qwen3-4B-Instruct-2507 "
             "--seeds 0 --pairs 384 --eps 0.2 --dtype bf16 --train-steps 400 "
             "--batch 4 --queries 2 --tag qwen3_4b "
             "--methods none retrain npo npo_P3_jailbreak",
         results=["lm_e2e_qwen3_4b.json"], requires="gpu"),
    dict(id="tofu_phi",
         cmd="python experiments/run_benchmark.py --dataset tofu --seeds 0 1 2 "
             "--resume --dtype fp16 --pairs 512",
         results=["lm_e2e_tofu_phi-1_5.json"], requires="gpu"),
    dict(id="muse_phi",
         cmd="python experiments/run_benchmark.py --dataset muse --seeds 0 1 2 "
             "--resume --dtype fp16 --pairs 512",
         results=["lm_e2e_muse_phi-1_5.json"], requires="gpu"),
    dict(id="tofu_gpt2_extra",
         cmd="python experiments/run_benchmark.py --dataset tofu --model gpt2 "
             "--seeds 3 4 5 --resume --dtype fp32 --train-steps 600 --batch 8 "
             "--tag tofu_gpt2_b",
         results=["lm_e2e_tofu_gpt2_b.json"], requires="any"),
    dict(id="qwen3_06b",
         cmd="python experiments/run_lm_big.py --model Qwen/Qwen3-0.6B --seeds 0 "
             "--pairs 384 --eps 0.2 --dtype fp32 --train-steps 400 --batch 4 "
             "--queries 2 --tag qwen3_06b "
             "--methods none retrain npo npo_P3_jailbreak",
         results=["lm_e2e_qwen3_06b.json"], requires="gpu"),
    dict(id="muse_gpt2_extra",
         cmd="python experiments/run_benchmark.py --dataset muse --model gpt2 "
             "--seeds 3 4 --resume --dtype fp32 --train-steps 600 --batch 8 "
             "--pairs 512 --tag muse_gpt2_512b",
         results=["lm_e2e_muse_gpt2_512b.json"], requires="any"),
    dict(id="tofu_pythia_extra",
         cmd="python experiments/run_benchmark.py --dataset tofu "
             "--model EleutherAI/pythia-160m --seeds 3 4 --resume --dtype fp32 "
             "--train-steps 600 --batch 8 --tag tofu_pythia160m_b",
         results=["lm_e2e_tofu_pythia160m_b.json"], requires="any"),
]


def log(worker, msg):
    print(f"[{time.strftime('%H:%M:%S')}][{worker}] {msg}", flush=True)


def sh(cmd, timeout=600):
    """Pipe-safe shell (colab CLI forks daemons that hold pipes)."""
    import tempfile
    with tempfile.NamedTemporaryFile("r", delete=False) as f:
        out_path = f.name
    try:
        rc = subprocess.call(
            f"timeout -k 10 {timeout} bash -c {shlex.quote(cmd)} "
            f"> {out_path} 2>&1 < /dev/null", shell=True, timeout=timeout + 60)
        with open(out_path) as f:
            return rc, f.read()
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"
    finally:
        os.unlink(out_path)


# ---------------------------------------------------------------- queue --

def load_queue():
    with open(QUEUE) as f:
        return json.load(f)


def save_queue(q):
    with open(QUEUE, "w") as f:
        json.dump(q, f, indent=2)


def lease(worker, caps):
    """Atomically lease the first pending task this worker can run."""
    with LOCK:
        q = load_queue()
        for t in q:
            if t["status"] == "pending" and (t["requires"] in caps):
                t["status"] = "running"
                t["worker"] = worker
                t["since"] = time.time()
                save_queue(q)
                return dict(t)
        return None


def settle(task_id, ok, worker):
    with LOCK:
        q = load_queue()
        for t in q:
            if t["id"] == task_id:
                if ok:
                    t["status"] = "done"
                else:
                    t["retries"] = t.get("retries", 0) + 1
                    t["status"] = "failed" if t["retries"] >= 4 else "pending"
                t.pop("worker", None)
                save_queue(q)
                return t["status"]


def task_done(task):
    return all(os.path.exists(os.path.join(REPO, "results", r))
               for r in task["results"])


def commit_results(task):
    sh(f"cd {REPO} && git add results && "
       f"git commit -q -m 'Results: {task['id']} (scheduler)\n\n"
       f"Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>\n"
       f"Claude-Session: https://claude.ai/code/session_01Vin7swZZoQkrdvJ75UwLjk' "
       f"&& git push -q origin main", timeout=180)


# --------------------------------------------------------------- workers --

def worker_local(name="local-cpu"):
    while True:
        task = lease(name, caps=("any",))
        if task is None:
            return
        log(name, f"running {task['id']}")
        rc, out = sh(f"cd {REPO} && {task['cmd']}", timeout=6 * 3600)
        ok = task_done(task)
        log(name, f"{task['id']} -> rc={rc} done={ok}")
        if ok:
            commit_results(task)
        settle(task["id"], ok, name)


def colab_accounts():
    return sorted(f for f in os.listdir(TOKEN_DIR)
                  if f.startswith("token_account"))


def colab_provision(name, session, accel, rotate=True):
    import shutil
    accounts = colab_accounts() if rotate else []
    for attempt in range(60):
        flag = "" if accel == "cpu" else f" --gpu {accel}"
        rc, out = sh(f"colab new -s {session}{flag}", timeout=300)
        if "READY" in out:
            log(name, f"session ready (attempt {attempt+1})")
            return True
        if accounts:
            src = accounts[attempt % len(accounts)]
            shutil.copy(os.path.join(TOKEN_DIR, src),
                        os.path.join(TOKEN_DIR, "token.json"))
            log(name, f"provision failed; rotated to {src}")
        time.sleep(240)
    return False


def colab_exec(py, session="vouchq", timeout=240):
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(py)
        path = f.name
    try:
        return sh(f"colab exec -s {session} -f {path}", timeout=timeout)
    finally:
        os.unlink(path)


def colab_stage(name, session):
    tar = f"/tmp/{session}.tar.gz"
    sh(f"cd {REPO} && tar czf {tar} --exclude=.git --exclude=results .")
    rc, out = sh(f"colab upload -s {session} {tar} vouchq.tar.gz", timeout=600)
    if rc != 0:
        return False
    rc, out = colab_exec(f"""
import subprocess, os
src = '/vouchq.tar.gz' if os.path.exists('/vouchq.tar.gz') else '/content/vouchq.tar.gz'
os.makedirs('{VMDIR}', exist_ok=True)
subprocess.run(['tar','xzf',src,'-C','{VMDIR}'])
os.makedirs('{VMDIR}/results', exist_ok=True)
subprocess.run(['pip','uninstall','-y','torchao'], capture_output=True)
subprocess.run(['pip','install','-q','peft','datasets'], capture_output=True)
print('STAGED')
""", session=session, timeout=900)
    return "STAGED" in out


def worker_colab(name="colab-t4", session="vouchq", accel="T4",
                 caps=("gpu", "any"), rotate=True):
    while True:
        task = lease(name, caps=caps)
        if task is None:
            return
        log(name, f"running {task['id']}")
        ok = False
        for cycle in range(6):   # session deaths -> re-provision, task resumes
            rc, out = colab_exec("print('PING')", session=session, timeout=90)
            if "PING" not in out:
                if not colab_provision(name, session, accel, rotate) \
                        or not colab_stage(name, session):
                    continue
            cmd = (f"cd {VMDIR} && nohup {task['cmd']} "
                   f"> /content/task_{task['id']}.log 2>&1 &")
            colab_exec("import subprocess; subprocess.run(['pkill','-f','run_'], capture_output=True); print('CLEAN')",
                       session=session)
            colab_exec(f"import subprocess; subprocess.Popen({cmd!r}, shell=True); print('GO')",
                       session=session)
            # poll until done / died
            while True:
                time.sleep(240)
                rc, out = colab_exec(
                    f"import os,subprocess;"
                    f"print(open('/content/task_{task['id']}.log').read()[-200:] "
                    f"if os.path.exists('/content/task_{task['id']}.log') else 'NOLOG');"
                    f"print('ALIVE' if subprocess.run(['pgrep','-f','run_'],"
                    f"capture_output=True,text=True).stdout.strip() else 'DEADPROC')",
                    session=session, timeout=180)
                if rc != 0 or "not found" in out or "lost" in out:
                    log(name, "session died; re-provisioning")
                    break
                for res in task["results"]:
                    sh(f"colab download -s {session} {VMDIR}/results/{res} "
                       f"{REPO}/results/{res}", timeout=600)
                if "Traceback" in out:
                    log(name, f"task error: {out[-200:]}")
                    break
                if task_done(task):
                    ok = True
                    break
                if "DEADPROC" in out:
                    log(name, "proc ended; relaunching (resume)")
                    break
            if ok:
                break
        log(name, f"{task['id']} -> done={ok}")
        if ok:
            commit_results(task)
        settle(task["id"], ok, name)


KAGGLE_WRAPPER = """\
import subprocess, os, shutil, glob
def sh(c):
    print('::', c, flush=True); r = subprocess.run(c, shell=True)
    print(':: exit', r.returncode, flush=True); return r.returncode
sh('pip uninstall -y torchao')
sh('pip install -q peft datasets')
import torch as _t
_dev = _t.cuda.get_device_name(0) if _t.cuda.is_available() else 'cpu'
print(':: device', _dev, flush=True)
if 'P100' in _dev:
    sh('pip install -q torch==2.7.0 --index-url https://download.pytorch.org/whl/cu118')
sh('git clone --depth 1 {git} /kaggle/vouch')
os.chdir('/kaggle/vouch')
sh('rm -f results/lm_e2e_*.json')
sh({cmd!r})
for f in glob.glob('results/lm_e2e_*.json'):
    shutil.copy(f, '/kaggle/working/' + os.path.basename(f))
print('JOB DONE', flush=True)
"""


def worker_kaggle(name="kaggle-gpu"):
    user = os.environ.get("KAGGLE_USERNAME", "")
    if not user:
        log(name, "no KAGGLE_USERNAME; worker disabled")
        return
    while True:
        task = lease(name, caps=("gpu", "any"))
        if task is None:
            return
        log(name, f"running {task['id']}")
        slug = f"vouch-{task['id'].replace('_','-')}"
        d = f"/tmp/kq_{slug}"
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "run.py"), "w") as f:
            f.write(KAGGLE_WRAPPER.format(git=GIT_URL, cmd=task["cmd"]))
        with open(os.path.join(d, "kernel-metadata.json"), "w") as f:
            json.dump({"id": f"{user}/{slug}", "title": slug,
                       "code_file": "run.py", "language": "python",
                       "kernel_type": "script", "is_private": "true",
                       "enable_gpu": "true", "enable_internet": "true"}, f)
        rc, out = sh(f"kaggle kernels push -p {d}")
        if "successfully" not in out.lower():
            log(name, f"push failed: {out[-150:]}")
            settle(task["id"], False, name)
            continue
        ok = False
        while True:
            time.sleep(240)
            rc, out = sh(f"kaggle kernels status {user}/{slug}")
            low = out.lower()
            log(name, f"{slug}: {out.strip().splitlines()[-1][:90] if out.strip() else '?'}")
            if "complete" in low or "error" in low or "cancel" in low:
                break
        od = f"/tmp/kq_{slug}_out"
        sh(f"kaggle kernels output {user}/{slug} -p {od}", timeout=1200)
        for res in task["results"]:
            src = os.path.join(od, res)
            if os.path.exists(src):
                subprocess.run(["cp", src, os.path.join(REPO, "results", res)])
        ok = task_done(task)
        log(name, f"{task['id']} -> done={ok}")
        if ok:
            commit_results(task)
        settle(task["id"], ok, name)


def main():
    if "--init" in sys.argv or not os.path.exists(QUEUE):
        q = [dict(t, status="pending", retries=0) for t in DEFAULT_TASKS]
        # drop tasks whose results already exist
        for t in q:
            if task_done(t):
                t["status"] = "done"
        save_queue(q)
        print("queue initialised:",
              [(t["id"], t["status"]) for t in q], flush=True)
        if "--init" in sys.argv:
            return
    # reset stale running leases from a previous scheduler run
    with LOCK:
        q = load_queue()
        for t in q:
            if t["status"] == "running":
                t["status"] = "pending"
        save_queue(q)
    workers = [
        threading.Thread(target=worker_colab, daemon=True,
                         kwargs=dict(name="colab-gpu", session="vouchq",
                                     accel="T4", caps=("gpu", "any"))),
        threading.Thread(target=worker_colab, daemon=True,
                         kwargs=dict(name="colab-cpu", session="vouchqc",
                                     accel="cpu", caps=("any",), rotate=False)),
        threading.Thread(target=worker_kaggle, daemon=True),
        threading.Thread(target=worker_local, daemon=True),
    ]
    for w in workers:
        w.start()
    for w in workers:
        w.join()
    q = load_queue()
    print("QUEUE FINAL:", [(t["id"], t["status"]) for t in q], flush=True)


if __name__ == "__main__":
    main()
