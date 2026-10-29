#!/usr/bin/env bash
# Run a command inside the project venv in WSL: wsl -d Ubuntu-24.04 -- bash scripts/wsl_run.sh <cmd> [args...]
set -e
cd /mnt/c/Users/praty/Projects/Code/Wordle
source ~/.venvs/wordle/bin/activate
export PYTHONIOENCODING=utf-8
exec "$@"
