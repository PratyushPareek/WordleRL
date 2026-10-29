#!/usr/bin/env bash
# 3-step GRPO smoke test with Qwen3-0.6B on the laptop GPU. Writes logs/smoke.log and a final SMOKE_EXIT=<code> line.
# Start from PowerShell: Start-Process wsl.exe -ArgumentList '-d','Ubuntu-24.04','--','bash','/mnt/c/Users/praty/Projects/Code/Wordle/scripts/smoke_test.sh' -WindowStyle Hidden
cd /mnt/c/Users/praty/Projects/Code/Wordle
mkdir -p logs
LOG=logs/smoke.log
source ~/.venvs/wordle/bin/activate
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1 TRL_EXPERIMENTAL_SILENCE=1
python train_grpo.py \
    --model Qwen/Qwen3-0.6B \
    --max-steps 3 \
    --num-generations 4 \
    --per-device-train-batch-size 4 \
    --gradient-accumulation-steps 4 \
    --max-completion-length 1024 \
    --eval-size 0 \
    --save-steps 1000 \
    --vllm-gpu-memory-utilization 0.3 \
    --output-dir outputs/smoke > "$LOG" 2>&1
echo "SMOKE_EXIT=$?" >> "$LOG"
