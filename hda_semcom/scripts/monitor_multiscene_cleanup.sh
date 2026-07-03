#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_ROOT="${OUT_ROOT:-outputs/multiscene_surrogate}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-300}"
LOG_PATH="${LOG_PATH:-$OUT_ROOT/cleanup_monitor.log}"

METRICS_DIR="$OUT_ROOT/metrics"
DEGRADED_DIR="$OUT_ROOT/gsfusion_degraded"

mkdir -p "$(dirname "$LOG_PATH")"

declare -A EXPECTED_ROWS=(
  [fit_alhambra]=120
  [fit_bannerman]=96
  [fit_barts]=120
  [fit_bridge]=120
  [fit_colosseum]=120
  [fit_dunnottar]=120
  [fit_eiffel]=120
  [fit_fushimi]=240
  [test_bannerman]=24
  [test_fushimi]=24
)

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG_PATH"
}

csv_rows() {
  local path="$1"
  if [[ ! -s "$path" ]]; then
    echo 0
    return
  fi
  local lines
  lines="$(wc -l < "$path")"
  if (( lines <= 0 )); then
    echo 0
  else
    echo $((lines - 1))
  fi
}

pipeline_running() {
  pgrep -f "scripts/run_multiscene_surrogate_pipeline.sh" >/dev/null 2>&1
}

log "cleanup monitor started; interval=${INTERVAL_SECONDS}s; out_root=$OUT_ROOT"

while true; do
  cleaned_any=0

  for name in "${!EXPECTED_ROWS[@]}"; do
    metrics="$METRICS_DIR/$name.csv"
    degraded="$DEGRADED_DIR/$name"
    expected="${EXPECTED_ROWS[$name]}"

    [[ -d "$degraded" ]] || continue

    rows="$(csv_rows "$metrics")"
    if (( rows >= expected )); then
      size="$(du -sh "$degraded" 2>/dev/null | awk '{print $1}')"
      log "removing completed degraded output: $degraded rows=$rows/$expected size=${size:-unknown}"
      rm -rf -- "$degraded"
      cleaned_any=1
    fi
  done

  if [[ "$cleaned_any" == "1" ]]; then
    df -h . | tee -a "$LOG_PATH"
  fi

  if ! pipeline_running; then
    log "pipeline process is not running; final cleanup pass complete"
    break
  fi

  sleep "$INTERVAL_SECONDS"
done

log "cleanup monitor stopped"
