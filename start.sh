#!/usr/bin/env bash
# ============================================================================
#  start.sh -- boot Qwen3.8-Flash-Next and wait until it answers.
#
#  Thin wrapper around the ~/launch-3.8flash.sh that setup.sh generated:
#  it checks the preconditions the launcher does not, picks the serving mode
#  non-interactively, and explains the launcher's one bogus warning.
#
#  Env (all optional, passed straight through to the launcher):
#    MODE=1|2        1 = HashK R=4 n-gram table, 200k ctx  (default, best quality)
#                    2 = PLE off, 256k ctx, better for 4+ concurrent sessions
#    PORT=30000      MEM_FRACTION=0.90
#    MAX_RUNNING=8   concurrent requests; also sizes req_to_token
#    MAMBA_CACHE=24  mamba slots (~60 MB each); 5 reserved per running request
#    CTX=            context length; defaults per MODE
#    THINKING=medium off | low | medium | xhigh
#
#  Boot takes about 9 minutes. Follow along with: docker logs -f flashnext
#
#  Measurements and findings from running this: LOG.md
# ============================================================================
set -uo pipefail

LAUNCHER="$HOME/launch-3.8flash.sh"
CONTAINER="flashnext"

# Load optional ./.env overrides. Shell env wins; .env fills the gaps; the
# defaults below apply last. Mirrored in setup.sh -- keep the two in sync.
ENV_FILE=".env"
if [ -f "$ENV_FILE" ]; then
  while IFS='=' read -r key value || [ -n "$key" ]; do
    key="${key%$'\r'}"; value="${value%$'\r'}"
    key="${key#"${key%%[![:space:]]*}"}"; key="${key%"${key##*[![:space:]]}"}"
    case "$key" in ''|\#*) continue ;; esac
    [ -z "${!key:-}" ] && export "$key=$value"
  done < "$ENV_FILE"
fi

# The launcher reads these from the environment, so export rather than assign.
export MODE="${MODE:-1}"
export PORT="${PORT:-30000}"
export THINKING="${THINKING:-medium}"
export MEM_FRACTION="${MEM_FRACTION:-0.90}"
# Concurrency. MAX_RUNNING also sizes req_to_token ((N+1) x CTX x 4 bytes), so
# lowering it is the biggest single lever on KV pool size.
export MAX_RUNNING="${MAX_RUNNING:-8}"
export MAMBA_CACHE="${MAMBA_CACHE:-24}"
[ -n "${CTX:-}" ] && export CTX

G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; N=$'\033[0m'
warn(){ printf '%s    %s%s\n' "$Y" "$1" "$N"; }
die(){ printf '\n%sFAIL%s %s\n' "$R" "$N" "$1" >&2; exit 1; }

healthy(){ curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; }
running(){ docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$CONTAINER"; }

[ -x "$LAUNCHER" ] || die "$LAUNCHER not found -- run ./setup.sh first"

# The launcher has its cache and patch paths baked in by the installer (:615),
# so a launcher generated before the paths changed would boot against an empty
# cache and fail deep inside the container. Check them here instead.
BAKED_CACHE="$(sed -n 's/^HF_CACHE="\(.*\)"$/\1/p' "$LAUNCHER" | head -1)"
BAKED_REPO="$(sed -n 's/^REPO="\(.*\)"$/\1/p' "$LAUNCHER" | head -1)"
for d in "$BAKED_CACHE" "$BAKED_REPO"; do
  [ -n "$d" ] && [ -d "$d" ] && continue
  die "$LAUNCHER points at a path that no longer exists:
         $d
       Re-run ./setup.sh to regenerate it."
done
if [ "$MODE" = "1" ] && [ ! -f "$BAKED_REPO/ple_hashk_R4.pt" ]; then
  die "mode 1 needs the HashK table, missing at
         $BAKED_REPO/ple_hashk_R4.pt
       Re-run ./setup.sh, or start with  MODE=2 ./start.sh  (PLE off, 256k ctx)."
fi

# ---- already up? ----------------------------------------------------------
if running; then
  if healthy; then
    printf '%s%s is already running and healthy on port %s%s\n' "$G" "$CONTAINER" "$PORT" "$N"
    exit 0
  fi
  die "$CONTAINER is up but not answering on $PORT -- it may still be booting.
       Watch it:   docker logs -f $CONTAINER
       Or reset:   ./stop.sh && ./start.sh"
fi

# ---- port free? -----------------------------------------------------------
# The launcher binds host networking, so a busy port fails deep inside the
# container instead of here.
if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
  die "port $PORT is already in use by something else.
       Free it, or pick another:  PORT=30001 ./start.sh"
fi

# ---- the launcher's swap warning is spurious at exactly 16 GiB ------------
# The launcher wants >= 16 GiB of swap but reads SwapTotal through awk's %d,
# which truncates. A 16 GiB swapfile reports a few KB under 16 GiB and so
# prints as 15, tripping a warning and a 10-second pause. Say so up front.
SWAP_KB="$(awk '/SwapTotal/{print $2}' /proc/meminfo)"
if [ "$(( SWAP_KB / 1048576 ))" -lt 16 ] && [ "$SWAP_KB" -ge 16000000 ]; then
  warn "the launcher is about to warn about swap and pause 10s. Ignore it:"
  warn "you have $(( SWAP_KB / 1024 )) MiB of swap; its check truncates that to $(( SWAP_KB / 1048576 )) GiB."
fi

# A genuinely short machine is a different matter -- the launcher pauses, but
# booting short of this can lock the box rather than fail cleanly.
RAM_GIB="$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)"
if [ "${RAM_GIB:-0}" -lt 105 ]; then
  warn "only ${RAM_GIB} GiB RAM available, this needs ~105. Close other GPU/memory work first."
fi

printf '\n%s==> starting %s (mode %s, port %s, thinking %s, max_running %s, mamba %s)%s\n\n' \
  "$G" "$CONTAINER" "$MODE" "$PORT" "$THINKING" "$MAX_RUNNING" "$MAMBA_CACHE" "$N"

"$LAUNCHER" || die "launch failed -- docker logs $CONTAINER"

cat <<DONE

  Test:   curl localhost:$PORT/v1/chat/completions -H 'Content-Type: application/json' \\
            -d '{"model":"m","messages":[{"role":"user","content":"Hello"}]}'
  Bench:  bash sgbench.sh
  Logs:   docker logs -f $CONTAINER
  Stop:   ./stop.sh

  The endpoint is /v1/chat/completions -- bare /chat/completions returns 404.

DONE
