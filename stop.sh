#!/usr/bin/env bash
# ============================================================================
#  stop.sh -- stop the Qwen3.8-Flash-Next server and free its memory.
#
#  The container holds ~91 GB of weights, so nothing else large will fit until
#  this has run. Removing it is the intended way to stop: the launcher creates
#  it fresh on every start.
# ============================================================================
set -uo pipefail

CONTAINER="flashnext"
PORT="${PORT:-30000}"

G=$'\033[32m'; N=$'\033[0m'

exists(){ docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qx "$CONTAINER"; }

if ! exists; then
  echo "$CONTAINER is not running"
  exit 0
fi

printf 'stopping %s ... ' "$CONTAINER"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || { echo "failed"; exit 1; }

# Weights are released as the process dies; give the port a moment to clear so
# a ./stop.sh && ./start.sh cycle does not race itself.
for _ in $(seq 1 20); do
  ss -ltn 2>/dev/null | grep -q ":${PORT} " || break
  sleep 0.5
done

printf '%sstopped%s\n' "$G" "$N"
awk '/MemAvailable/{printf "  %d GiB RAM now available\n", $2/1048576}' /proc/meminfo
