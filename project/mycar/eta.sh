#!/bin/bash
# Prints PPO training progress + ETA. Usage: ./eta.sh [logfile]  (default: newest train_ppo*.log)
cd "$(dirname "$0")"
LOG="${1:-$(ls -t train_ppo*.log | head -1)}"
echo "$LOG"
tr '\r' '\n' < "$LOG" | awk -v run=100000 '
/time_elapsed/ {t=$4} /total_timesteps/ {s=$4; if (!first) {first=s; t0=t}}
END { if (s>0 && t>t0) { start=first-2048; done=s-start; rate=done/t; left=(run-done)/rate
  printf "%d / %d steps this run (%.0f%%) | total %d | %.1f steps/s | ~%d min left\n", done, run, 100*done/run, s, rate, left/60 } }'
