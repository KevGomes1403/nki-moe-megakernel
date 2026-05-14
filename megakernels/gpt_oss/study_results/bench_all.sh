#!/bin/bash
# Sequential bench for all already-compiled cells.
# Looks at /home/ubuntu/models/study/b*_s*_*/  for compiled artifacts,
# runs bench (--skip-compile) for each cell that doesn't yet have a benched report.
set -u

REPO=/home/ubuntu/nki-moe
DRIVER=$REPO/megakernels/gpt_oss/study_results/study_driver.py
RESULTS=$REPO/megakernels/gpt_oss/study_results
LOGDIR=/home/ubuntu/models/study/logs
MODELS=/home/ubuntu/models/study

# Discover cells: b<B>_s<S>_<mode>
cd "$MODELS"
for cell in b*_s*_*/; do
  cell=${cell%/}
  [[ "$cell" =~ ^b([0-9]+)_s([0-9]+)_(mega|xla)$ ]] || continue
  bs=${BASH_REMATCH[1]}; seq=${BASH_REMATCH[2]}; mode=${BASH_REMATCH[3]}
  mode_arg=$([ "$mode" = "mega" ] && echo "nki" || echo "xla")
  out=$MODELS/$cell
  rep=$RESULTS/$cell.json
  log=$LOGDIR/$cell.bench.log
  work=/tmp/nxd_work/$cell

  # Check artifact validity
  if [ ! -f "$out/model.pt" ]; then
    echo "[SKIP] $cell no model.pt"
    continue
  fi

  # Check if already benched
  if [ -f "$rep" ] && grep -q '"latency_ms_p50"' "$rep" 2>/dev/null; then
    echo "[SKIP] $cell already benched"
    continue
  fi

  echo "[RUN ] benching $cell"
  t0=$(date +%s)
  BASE_COMPILE_WORK_DIR="$work/" NEURON_LOGICAL_NC_CONFIG=1 \
    python "$DRIVER" \
      --mode "$mode_arg" --seq-len "$seq" --batch-size "$bs" \
      --compiled-out "$out" --report-out "$rep" \
      --skip-compile \
      > "$log" 2>&1
  rc=$?
  t1=$(date +%s)
  echo "[DONE] $cell rc=$rc duration=$((t1-t0))s"
  if [ $rc -ne 0 ] || [ ! -f "$rep" ]; then
    echo "  -> last 20 lines:"
    tail -20 "$log" | sed 's/^/    /'
  fi
done
echo "[BENCH ALL DONE]"
