#!/bin/bash
# Laptop side (WSL), adapted from LearningExp's launcher. Ships the committed code to a fresh Colab VM, runs
# scripts/colab/remote_job.py there with a plan file, downloads results every poll, and always stops the VM.
#   bash scripts/colab/launch.sh --plan plans/NAME.json [--gpu L4] [--max-hours 1.5] [--session NAME]
# Start detached from PowerShell (survives the Claude session):
#   Start-Process wsl.exe -ArgumentList '-d','Ubuntu-24.04','--','bash','/mnt/c/Users/praty/Projects/Code/Wordle/scripts/colab/launch.sh','--plan','plans/NAME.json' -WindowStyle Hidden
# Needs: Colab CLI login, a clean git tree (the VM gets exactly the committed code: git archive HEAD).
set -uo pipefail
# Run from a private copy: bash reads scripts as it goes, so editing launch.sh mid-run would corrupt it.
if [ -z "${WORDLE_LAUNCH_COPY:-}" ]; then
  export WORDLE_REPO=$(cd "$(dirname "$0")/../.." && pwd)
  copy=$(mktemp /tmp/wordle_launch.XXXXXX.sh); cp "$0" "$copy"
  WORDLE_LAUNCH_COPY=$copy exec bash "$copy" "$@"
fi
trap 'rm -f "$WORDLE_LAUNCH_COPY"' EXIT
cd "$WORDLE_REPO"
export PATH="$HOME/.local/bin:$PATH"
source ~/.venvs/wordle/bin/activate
export TZ=Asia/Kolkata   # timestamps in IST
# oauth2 is the login that works for this project (signed in from WSL on 2026-10-29); ADC is not set up.
C() { "${COLAB_BIN:-colab}" --auth="${COLAB_AUTH:-oauth2}" "$@" 2>&1 | grep -av "new version of Colab CLI\|colab update\|enable_update_check"; return "${PIPESTATUS[0]}"; }

GPU=L4; MAX_HOURS=1.5; PLAN=""; S=""; EXTRA=()
# Must stay well under the VM's release_grace_sec (300 s), so the final pull happens before the VM releases itself.
POLL_SEC=${POLL_SEC:-120}
while [ $# -gt 0 ]; do
  case $1 in
    --plan) PLAN=$2; shift 2 ;;  --gpu) GPU=$2; shift 2 ;;  --max-hours) MAX_HOURS=$2; shift 2 ;;
    --session) S=$2; shift 2 ;;  --preflight-only) PREFLIGHT_ONLY=1; shift ;;
    --extra) EXTRA+=("$2"); shift 2 ;;  # repo-relative path not in git (e.g. a checkpoint), uploaded alongside the code
    *) echo "unknown arg $1"; exit 2 ;;
  esac
done
fail() { echo "PRE-FLIGHT FAILED: $*"; exit 1; }
STAMP=$(date +%Y%m%d-%H%M%S)
[ -n "$S" ] || S="wordle-$STAMP"   # every run gets its own session name
OUT=results/colab_launch_$STAMP     # git-ignored: launch.log, exec.log, status.json, results.tar.gz
TMP=$(mktemp -d)
say() { echo "[$(date '+%F %T %Z')] $*" | tee -a "$OUT/launch.log"; }

# ---------- pre-flight (nothing billable happens before this passes) ----------
[ -n "$PLAN" ] || fail "--plan is required"
[ -z "$(git status --porcelain)" ] || fail "uncommitted changes: commit first so the VM runs known code"
git cat-file -e "HEAD:$PLAN" 2>/dev/null || fail "plan $PLAN is not committed"
case $GPU in L4|A100|H100|G4) ;; *) fail "GPU must support bf16 (L4/A100/H100/G4), got $GPU" ;; esac
C sessions > "$TMP/sessions.txt" || fail "Colab CLI not logged in (run: ~/.local/bin/colab sessions, follow the link)"
grep -qw "$S" "$TMP/sessions.txt" && fail "a Colab session named $S already exists"
COMMIT=$(git rev-parse --short HEAD)
git archive --format=tar.gz -o "$TMP/bundle.tar.gz" HEAD
# Test exactly what the VM receives: no CRLF, shell scripts parse, Python compiles, plan is valid JSON.
mkdir -p "$TMP/check" && tar -xzf "$TMP/bundle.tar.gz" -C "$TMP/check"
crlf=$(grep -rlI $'\r' "$TMP/check" | sed "s|$TMP/check/||")
[ -z "$crlf" ] || fail "CRLF line endings in: $crlf"
for f in $(find "$TMP/check" -name '*.sh'); do bash -n "$f" || fail "bash -n failed: ${f#$TMP/check/}"; done
python -m py_compile $(find "$TMP/check" -name '*.py') || fail "python compile failed in the bundle"
[ -f "$TMP/check/requirements.lock" ] || fail "requirements.lock missing from the bundle"
for e in "${EXTRA[@]}"; do [ -e "$e" ] || fail "--extra $e does not exist"; done
if [ ${#EXTRA[@]} -gt 0 ]; then
  tar -czf "$TMP/extra.tar.gz" "${EXTRA[@]}" && tar -xzf "$TMP/extra.tar.gz" -C "$TMP/check"
fi
python - "$TMP/check" "$PLAN" "$TMP/params.json" "$COMMIT" "$MAX_HOURS" "${#EXTRA[@]}" <<'PY' || fail "plan file invalid"
import json, os, sys
check, plan_rel, params_path, commit, max_hours, n_extra = sys.argv[1:]
plan = json.load(open(os.path.join(check, plan_rel)))
assert plan["items"] and all(set(i) >= {"name", "args"} for i in plan["items"]), "items need name and args"
assert len({i["name"] for i in plan["items"]}) == len(plan["items"]), "duplicate item names"
for i in plan["items"]:  # a checkpoint to resume from must reach the VM (in git or via --extra)
    a = i["args"]
    if "--resume-from-checkpoint" in a:
        ck = a[a.index("--resume-from-checkpoint") + 1]
        assert os.path.isfile(os.path.join(check, ck, "trainer_state.json")), f"{i['name']}: {ck} not in the bundle or --extra"
json.dump({"workdir": "/content/wordle", "bundle": "/content/wordle_bundle.tar.gz",
           "extra": "/content/wordle_extra.tar.gz" if int(n_extra) else None,
           "status_file": "/content/wordle_status.json", "log_file": "/content/wordle_job.log",
           "results_tar": "/content/wordle_results.tar.gz", "ack_file": "/content/wordle_pulled",
           "git_commit": commit, "plan": plan["items"], "max_hours": float(max_hours),
           "pack_minutes": 5, "self_release": True, "release_grace_sec": 300}, open(params_path, "w"), indent=1)
print(f"plan: {', '.join(i['name'] for i in plan['items'])}")
PY
if [ "${PREFLIGHT_ONLY:-0}" = 1 ]; then
  echo "pre-flight OK (--preflight-only: nothing launched): commit $COMMIT, plan $PLAN, GPU $GPU, session $S"
  rm -rf "$TMP"; exit 0
fi
mkdir -p "$OUT"
say "pre-flight OK: commit $COMMIT, plan $PLAN, GPU $GPU, max $MAX_HOURS h, session $S"

# ---------- VM lifetime: from here on, the VM is always stopped on exit ----------
cleanup() {
  [ -n "${CAP_PID:-}" ] && kill "$CAP_PID" 2>/dev/null   # its trap resumes the client first
  [ -n "${EXEC_PID:-}" ] && { kill -CONT "$EXEC_PID" 2>/dev/null; kill "$EXEC_PID" 2>/dev/null; }
  if [ "${KEEP_VM:-0}" = 1 ]; then say "KEEP_VM=1: NOT stopping $S (it keeps billing; run: colab stop -s $S)"
  else say "stopping VM $S"; C stop -s "$S" >> "$OUT/launch.log"; fi
  say "server sessions now: $(C sessions | tr '\n' ' ')"
  say "balance now: $(C usage | tr '\n' ' ')"
  rm -rf "$TMP"; rm -f "$WORDLE_LAUNCH_COPY"
}
trap cleanup EXIT
trap 'say "interrupted"; exit 130' INT TERM

say "balance before: $(C usage | tr '\n' ' ')"
C new -s "$S" --gpu "$GPU" >> "$OUT/launch.log" || { say "colab new failed (quota/entitlement for $GPU?)"; exit 1; }
say "VM $S allocated; billing now: $(C usage | grep -i rate | tr '\n' ' ') (L4 should be ~1.54/hr)"
C upload -s "$S" "$TMP/bundle.tar.gz" /content/wordle_bundle.tar.gz >> "$OUT/launch.log" &&
C upload -s "$S" "$TMP/params.json" /content/wordle_params.json >> "$OUT/launch.log" || { say "upload failed"; exit 1; }
if [ ${#EXTRA[@]} -gt 0 ]; then
  C upload -s "$S" "$TMP/extra.tar.gz" /content/wordle_extra.tar.gz >> "$OUT/launch.log" || { say "extra upload failed"; exit 1; }
  say "uploaded extra: ${EXTRA[*]} ($(du -h "$TMP/extra.tar.gz" | cut -f1))"
fi

# Called directly (not via C) so $! is the client itself. The attached client keeps the VM alive (Colab
# reclaims the VM ~20 min after it disconnects). --timeout: the CLI default is 30 s of execution.
timeout_sec=$(python -c "print(int(($MAX_HOURS + 0.5) * 3600))")
start_exec() {
  "${COLAB_BIN:-colab}" --auth="${COLAB_AUTH:-oauth2}" exec -s "$S" --timeout "$timeout_sec" \
    -f scripts/colab/remote_job.py >> "$OUT/exec.log" 2>&1 &
  EXEC_PID=$!
  if [ "${CPU_CAP:-1}" = 1 ]; then  # pause/resume duty cycle: ~20% of one core at most
    ( trap 'kill -CONT $EXEC_PID 2>/dev/null; exit 0' TERM INT
      while kill -0 "$EXEC_PID" 2>/dev/null; do
        kill -STOP "$EXEC_PID" 2>/dev/null; sleep 0.8; kill -CONT "$EXEC_PID" 2>/dev/null; sleep 0.2
      done ) &
    CAP_PID=$!
    say "client CPU cap on (pid $EXEC_PID, ~20% of a core)"
  fi
}
# The client sometimes loses its connection before the job starts (twice on 2026-10-29), leaving a billed VM
# idle. Until the job has written its status file, a dead client means the job never started: send it again.
started=0
for try in 1 2 3; do
  start_exec
  say "remote job sent (try $try); waiting for its status file"
  for _ in $(seq 1 24); do  # up to ~4 min
    sleep 10
    if C download -s "$S" /content/wordle_status.json "$OUT/status.json" > /dev/null; then started=1; break; fi
    kill -0 "$EXEC_PID" 2>/dev/null || break
  done
  [ $started = 1 ] && break
  if kill -0 "$EXEC_PID" 2>/dev/null; then say "client alive but no status file after 4 min"; break; fi
  say "client exited before the job started (see exec.log); resending"
  [ -n "${CAP_PID:-}" ] && kill "$CAP_PID" 2>/dev/null; CAP_PID=""
done
[ $started = 1 ] || { say "remote job did not start; stopping the VM"; exit 1; }
say "remote job started; polling every $POLL_SEC s (logs: $OUT)"

pull() {  # download this session's results archive and unpack it into results/ (it only holds this session's items)
  C download -s "$S" /content/wordle_results.tar.gz "$OUT/results.tar.gz" > /dev/null &&
    tar -xzf "$OUT/results.tar.gz" -C . && return 0
  return 1
}

# ---------- watch ----------
end=$(( $(date +%s) + $(python -c "print(int(($MAX_HOURS + 0.75) * 3600))") ))
st=unknown; misses=0
while [ "$(date +%s)" -lt "$end" ]; do
  sleep "$POLL_SEC"
  if C download -s "$S" /content/wordle_status.json "$OUT/status.json" > /dev/null; then
    misses=0
    st=$(python -c "import json; print(json.load(open('$OUT/status.json'))['state'])")
    pull && say "pulled (results so far)"
    say "remote state=$st | $(python -c "import json; s=json.load(open('$OUT/status.json')); print('current', s['current'], 'done', s['items_done'], 'failed', s['items_failed'])")"
    [ "$st" = done ] && break
  else
    misses=$((misses + 1)); say "status unreachable ($misses)"
    if ! C sessions | grep -qw "$S"; then say "VM $S no longer exists"; break; fi
    [ $misses -ge 6 ] && { say "status unreachable for $((misses * POLL_SEC / 60)) min; giving up"; break; }
  fi
done
[ "$(date +%s)" -ge "$end" ] && say "laptop deadline reached"

# ---------- results ----------
if pull; then
  say "pulled final results"
  echo ok > "$TMP/ack" && C upload -s "$S" "$TMP/ack" /content/wordle_pulled > /dev/null  # lets the VM release now
else
  say "final pull failed (VM gone?); results from the last successful poll are in results/"
fi
# A pull should only add lines. Removed lines mean a committed file was overwritten: restore it from git.
git diff --numstat -- results | awk '$2 > 0 {print $3 " (" $2 " lines removed)"}' > "$TMP/overwritten.txt"
[ -s "$TMP/overwritten.txt" ] && say "WARNING: the pull removed lines from: $(tr '\n' ';' < "$TMP/overwritten.txt")"
say "final remote status: $(tr -d '\n' < "$OUT/status.json" 2>/dev/null | cut -c1-400)"
# Give the VM a moment to self-release before cleanup's colab stop (the backup).
sleep 20
