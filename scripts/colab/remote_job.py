"""Runs ON the Colab VM, sent by `colab exec -f` from scripts/colab/launch.sh (adapted from LearningExp).
Stdlib only: it runs in Colab's own kernel Python; project code runs in /content/venv built from requirements.lock.

1. unpack the committed code   2. build the venv from the lock file, check imports and bf16
3. run each plan item (train_grpo.py with its args), skipping items whose results/<name>/DONE is committed,
   and keep going past a failed item   4. deadline: stop after max_hours
5. pack this session's results for the laptop (every few minutes and at the end), then release the VM.
The laptop polls /content/wordle_status.json and downloads /content/wordle_results.tar.gz.
"""
import json
import os
import signal
import subprocess
import sys
import tarfile
import threading
import time
import traceback
from pathlib import Path

# Timestamps in IST; set before anything is logged, and inherited by train_grpo.py
os.environ["TZ"] = "Asia/Kolkata"
time.tzset()
P = json.loads(Path(os.environ.get("WORDLE_PARAMS", "/content/wordle_params.json")).read_text())
WORK = Path(P["workdir"])
STATUS = Path(P["status_file"])
LOG = Path(P["log_file"])
VENV = Path(P.get("venv", "/content/venv"))
PY = str(VENV / "bin" / "python")
state = {"state": "starting", "started": time.strftime("%F %T %Z"), "commit": P.get("git_commit"),
         "items_done": [], "items_failed": [], "items_skipped": [], "current": None, "deadline_hit": False,
         "error": None, "last_pack": None}
_pack_lock = threading.Lock()
_ran = []  # names run in this session; only their results are packed (so the pull never replaces older results)


def log(msg):
    line = f"[{time.strftime('%F %T %Z')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def set_state(**kw):
    state.update(kw)
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    tmp.replace(STATUS)  # atomic, so the laptop never reads half a file


def venv_env():
    env = dict(os.environ)
    env.pop("UV_SYSTEM_PYTHON", None)  # Colab sets it, which makes uv install into the system Python
    env.update(VIRTUAL_ENV=str(VENV), PATH=f"{VENV}/bin:{env['PATH']}", PYTHONUNBUFFERED="1",
               TRL_EXPERIMENTAL_SILENCE="1", PYTHONIOENCODING="utf-8")
    return env


def sh(cmd, prefix="  "):
    """Run a command in WORK, relaying its output (the kernel only relays this process's own prints)."""
    p = subprocess.Popen(cmd, cwd=WORK, env=venv_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        log(prefix + line.rstrip())
    return p.wait()


def pack():
    """Tar this session's results atomically for the laptop to download. Of the trainer checkpoints only the
    newest is included (rank-1 LoRA: ~25 MB with optimizer state), so a lost VM loses at most save_steps steps."""
    with _pack_lock:
        tmp = P["results_tar"] + ".tmp"
        with tarfile.open(tmp, "w:gz") as t:
            for name in _ran:
                ckpts = sorted((WORK / "results" / name).glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
                older = {f"results/{name}/{c.name}" for c in ckpts[:-1]}
                for rel in (f"results/{name}", f"results/logs/{name}.log"):
                    if (WORK / rel).exists():
                        t.add(WORK / rel, arcname=rel,
                              filter=lambda ti: None if any(ti.name == o or ti.name.startswith(o + "/") for o in older)
                              else ti)
        os.replace(tmp, P["results_tar"])
        set_state(last_pack=time.strftime("%F %T %Z"))


def periodic_pack(stop):
    while not stop.wait(P.get("pack_minutes", 5) * 60):
        try:
            pack()
        except Exception as e:  # a failed pack must not kill the run
            log(f"pack failed: {e}")


def run_item(item, deadline_at):
    name = item["name"]
    out = WORK / "results" / name
    if (out / "DONE").exists():
        log(f"SKIP {name} (results/{name}/DONE is committed)")
        state["items_skipped"].append(name)
        set_state()
        return
    remaining = deadline_at - time.time()
    if remaining <= 60:
        log(f"DEADLINE: no time left for {name}")
        state["items_failed"].append(name)
        set_state(deadline_hit=True)
        return
    _ran.append(name)
    (WORK / "results" / "logs").mkdir(parents=True, exist_ok=True)
    item_log = WORK / "results" / "logs" / f"{name}.log"
    cmd = [PY, "train_grpo.py", *item["args"], "--output-dir", f"results/{name}"]
    log(f"ITEM {name} started: {' '.join(cmd[1:])}")
    set_state(current=name)
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=WORK, env=venv_env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, start_new_session=True)

    def kill():
        if proc.poll() is None:
            log(f"DEADLINE {P['max_hours']} h reached during {name}: stopping it")
            set_state(deadline_hit=True)
            os.killpg(proc.pid, signal.SIGTERM)
            time.sleep(30)
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)

    timer = threading.Timer(remaining, kill)
    timer.daemon = True
    timer.start()
    with open(item_log, "w") as f:
        for line in proc.stdout:
            f.write(line)
            # tqdm bars and per-request vLLM noise stay in the item log; progress lines go to the main log
            if "it/s]" not in line and "s/it]" not in line:
                log(f"  [{name}] {line.rstrip()}")
    rc = proc.wait()
    timer.cancel()
    mins = (time.time() - t0) / 60
    if rc == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "DONE").write_text(f"commit {P.get('git_commit')} finished {time.strftime('%F %T %Z')}\n")
        state["items_done"].append(name)
        log(f"ITEM {name} DONE in {mins:.1f} min")
    else:
        state["items_failed"].append(name)
        log(f"ITEM {name} FAILED rc={rc} after {mins:.1f} min (see results/logs/{name}.log); continuing")
    set_state(current=None)
    pack()


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    with tarfile.open(P["bundle"]) as t:
        t.extractall(WORK, filter="data")
    log(f"code ready in {WORK} (commit {P.get('git_commit')})")
    deadline_at = time.time() + P["max_hours"] * 3600

    set_state(state="setup")
    if P.get("setup_venv", True):  # False only for local tests of this script with an existing venv
        sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
        if sh(["uv", "venv", "-q", "--python", "3.12", str(VENV)]) or \
           sh(["uv", "pip", "install", "-q", "--python", PY, "-r", "requirements.lock"]):
            raise RuntimeError("venv setup failed")
        check = ("import torch, transformers, trl, peft, vllm, textarena; "
                 "print('GPU', torch.cuda.get_device_name(0), 'bf16', torch.cuda.is_bf16_supported(), "
                 "'torch', torch.__version__, 'trl', trl.__version__, 'vllm', vllm.__version__); "
                 "raise SystemExit(0 if torch.cuda.is_bf16_supported() else 3)")
        if sh([PY, "-c", check]):
            raise RuntimeError("imports failed or GPU has no bf16 (T4?)")
    log(f"setup done; {len(P['plan'])} plan items, deadline {P['max_hours']} h")

    stop = threading.Event()
    threading.Thread(target=periodic_pack, args=(stop,), daemon=True).start()
    set_state(state="running", pipeline_started=True)
    for item in P["plan"]:
        run_item(item, deadline_at)
        if state["deadline_hit"]:
            break
    stop.set()


try:
    main()
except Exception as e:
    log("ERROR: " + "".join(traceback.format_exception(e)))
    set_state(error=str(e))
finally:
    set_state(state="saving")
    try:
        pack()
    except Exception as e:
        log(f"final pack failed: {e}")
    set_state(state="done", finished=time.strftime("%F %T %Z"))
    log(f"FINISHED done={state['items_done']} failed={state['items_failed']} skipped={state['items_skipped']}")
    if P.get("self_release"):
        grace = P.get("release_grace_sec", 300)  # time for the laptop to download the final results archive
        log(f"waiting up to {grace}s for the laptop to download results, then releasing the VM")
        ack = Path(P.get("ack_file", "/content/wordle_pulled"))
        for _ in range(grace // 5):
            if ack.exists():
                break
            time.sleep(5)
        try:
            from google.colab import runtime
            log("releasing VM (runtime.unassign)")
            runtime.unassign()
        except Exception as e:
            log(f"self-release failed ({e}); the laptop launcher will run colab stop")
