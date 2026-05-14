#!/bin/bash
# Sequential compile+bench queue for the study matrix.
# Each entry: <mode> <seq_len> <batch_size>
set -u

REPO=/home/ubuntu/nki-moe
DRIVER=$REPO/megakernels/gpt_oss/study_results/study_driver.py
RESULTS=$REPO/megakernels/gpt_oss/study_results
LOGDIR=/home/ubuntu/models/study/logs
MODELS=/home/ubuntu/models/study
WORKROOT=/tmp/nxd_work

mkdir -p "$LOGDIR" "$MODELS" "$WORKROOT"

# Queue: <mode> <seq_len> <batch>
# Order: alternate xla/mega per seq_len so each pair is fresh, prioritize seq sweep, then batch sweep.
QUEUE=(
  "mega 2048 1"
  "xla  2048 1"
  "mega 4096 1"
  "xla  4096 1"
  "mega 8192 1"
  "xla  8192 1"
  "mega 640  4"
  "xla  640  4"
  "mega 640  8"
  "xla  640  8"
  "mega 16384 1"
  "xla  16384 1"
  "mega 640  16"
  "xla  640  16"
)

run_one() {
  local mode=$1 seq=$2 bs=$3
  local mode_arg=$([ "$mode" = "mega" ] && echo "nki" || echo "xla")
  local cell="b${bs}_s${seq}_${mode}"
  local out="$MODELS/$cell"
  local rep="$RESULTS/$cell.json"
  local log="$LOGDIR/$cell.log"
  local work="$WORKROOT/$cell"
  mkdir -p "$out" "$work"

  if [ -f "$rep" ]; then
    if grep -q '"latency_ms_p50"' "$rep" 2>/dev/null; then
      echo "[SKIP] $cell already benched"
      return 0
    fi
  fi

  echo "[RUN ] $cell -> log=$log"
  local t0=$(date +%s)
  BASE_COMPILE_WORK_DIR="$work/" \
    python "$DRIVER" \
      --mode "$mode_arg" --seq-len "$seq" --batch-size "$bs" \
      --compiled-out "$out" --report-out "$rep" \
      > "$log" 2>&1
  local rc=$?
  local t1=$(date +%s)
  echo "[DONE] $cell rc=$rc duration=$((t1-t0))s"
  if [ $rc -ne 0 ]; then
    echo "  -> last 25 lines of $log:"
    tail -25 "$log" | sed 's/^/    /'
    # Continue queue rather than fail entire matrix
  fi
}

cd "$REPO"
for entry in "${QUEUE[@]}"; do
  read -r mode seq bs <<< "$entry"
  run_one "$mode" "$seq" "$bs"
done

echo "[QUEUE DONE]"
ls -1 "$RESULTS"/*.json 2>/dev/null
