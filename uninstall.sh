#!/usr/bin/env bash
# ============================================================================
#  uninstall.sh -- undo setup.sh, in tiers.
#
#  Cheap things go by default. The two expensive ones are opt-in, because the
#  cost of removing them is measured in how long it takes to get them back:
#
#    default     container, ~/launch-3.8flash.sh, .work/   ~13 GB, ~7 min back
#    --image     the SGLang container image                ~14 GB, ~10 min back
#    --model     the 135 GB checkpoint                     ~135 GB, hours back
#    --all       all of the above
#
#    -n, --dry-run   list what would be removed, remove nothing
#    -y, --yes       skip the confirmation prompt
#
#  Paths come from the generated launcher when it exists, so this removes what
#  setup.sh actually created even if it ran with HF_HOME/WORK overrides.
#
#  Leaves alone: setup.sh, start.sh, stop.sh, .gitignore and everything the
#  upstream repo ships.
# ============================================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

CONTAINER="flashnext"
LAUNCHER="$HOME/launch-3.8flash.sh"
INSTALLER="install-3.8flash.sh"
PROJECT="$PWD"

MODEL_ID="$(sed -n 's/^MODEL_ID="\(.*\)"$/\1/p' "$INSTALLER" 2>/dev/null | head -1)"
IMAGE="$(sed -n 's/^IMAGE="\(.*\)"$/\1/p' "$INSTALLER" 2>/dev/null | head -1)"
: "${MODEL_ID:=RadixArk/Qwen3.8-Flash-Next-NVFP4}"
: "${IMAGE:=lmsysorg/sglang:qwen38flashnext}"

G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; N=$'\033[0m'
warn(){ printf '%s    %s%s\n' "$Y" "$1" "$N"; }
die(){ printf '\n%sFAIL%s %s\n' "$R" "$N" "$1" >&2; exit 1; }

DO_IMAGE=0; DO_MODEL=0; DRY=0; YES=0
while [ $# -gt 0 ]; do
  case "$1" in
    --image)        DO_IMAGE=1 ;;
    --model)        DO_MODEL=1 ;;
    --all)          DO_IMAGE=1; DO_MODEL=1 ;;
    -n|--dry-run)   DRY=1 ;;
    -y|--yes)       YES=1 ;;
    -h|--help)      sed -n '2,20p' "$0" | sed 's/^# \{0,2\}//'; exit 0 ;;
    *)              die "unknown option: $1  (try --help)" ;;
  esac
  shift
done

# ---- resolve the paths setup.sh actually used --------------------------------
# The launcher is authoritative: install-3.8flash.sh:615 seds the real REPO and
# HF_CACHE into it. Fall back to this project's defaults if it is already gone.
REPO=""; HF_CACHE=""
if [ -f "$LAUNCHER" ]; then
  REPO="$(sed -n 's/^REPO="\(.*\)"$/\1/p' "$LAUNCHER" | head -1)"
  HF_CACHE="$(sed -n 's/^HF_CACHE="\(.*\)"$/\1/p' "$LAUNCHER" | head -1)"
fi
: "${REPO:=$PROJECT/.work/flashnext-one-spark}"
: "${HF_CACHE:=${HF_HOME:-$PROJECT/.cache/huggingface}}"
WORK="$(dirname "$REPO")"
SNAP_DIR="$HF_CACHE/hub/models--${MODEL_ID//\//--}"

hsize(){ [ -e "$1" ] && du -shL "$1" 2>/dev/null | cut -f1 || echo "-"; }

# ---- what is actually here ---------------------------------------------------
printf '\n%s=== uninstall: what will be removed ===%s\n\n' "$G" "$N"

TARGETS=()
docker ps -a --format '{{.Names}}' 2>/dev/null | grep -qx "$CONTAINER" \
  && { echo "  container   $CONTAINER"; TARGETS+=("container"); }
[ -f "$LAUNCHER" ] && { echo "  launcher    $LAUNCHER"; TARGETS+=("launcher"); }
[ -d "$WORK" ]     && { printf '  work        %-6s %s\n' "$(hsize "$WORK")" "$WORK"; TARGETS+=("work"); }

if [ "$DO_IMAGE" = 1 ]; then
  if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    SZ="$(docker image inspect "$IMAGE" --format '{{.Size}}' 2>/dev/null)"
    printf '  image       %-6s %s\n' "$(( ${SZ:-0} / 1000000000 )) GB" "$IMAGE"
    TARGETS+=("image")
  fi
fi

if [ "$DO_MODEL" = 1 ] && [ -d "$SNAP_DIR" ]; then
  printf '  checkpoint  %-6s %s\n' "$(hsize "$SNAP_DIR")" "$SNAP_DIR"
  TARGETS+=("model")
fi

[ "${#TARGETS[@]}" -eq 0 ] && { echo "  nothing to remove"; exit 0; }

# What is deliberately being kept, so the tiers are never a surprise.
echo
[ "$DO_MODEL" = 0 ] && [ -d "$SNAP_DIR" ] \
  && echo "  keeping     $(hsize "$SNAP_DIR")  checkpoint   (--model to remove)"
[ "$DO_IMAGE" = 0 ] && docker image inspect "$IMAGE" >/dev/null 2>&1 \
  && echo "  keeping             image        (--image to remove)"

# Root-owned files are normal here: the HashK build and the server run as root
# inside the container and write into these paths.
NEEDS_SUDO=0
for p in "$WORK" "$SNAP_DIR"; do
  [ -d "$p" ] || continue
  [ -n "$(find "$p" ! -user "$(id -un)" -print -quit 2>/dev/null)" ] && NEEDS_SUDO=1
done
[ "$NEEDS_SUDO" = 1 ] && { echo; warn "some files are root-owned (written by the container) -- sudo will be used"; }

if [ "$DRY" = 1 ]; then
  printf '\n%sdry run -- nothing removed%s\n\n' "$Y" "$N"
  exit 0
fi

if [ "$YES" = 0 ]; then
  echo
  printf '  Proceed? [y/N] '
  read -r ans
  case "$ans" in [yY]|[yY][eE][sS]) ;; *) echo "  aborted"; exit 0 ;; esac
fi

# ---- remove ------------------------------------------------------------------
echo

# rm as us; fall back to sudo only for the paths that actually need it.
rm_path(){
  local p="$1"
  [ -e "$p" ] || return 0
  if rm -rf "$p" 2>/dev/null && [ ! -e "$p" ]; then return 0; fi
  echo "    (root-owned, using sudo)"
  sudo rm -rf "$p" || { warn "could not remove $p -- try: sudo rm -rf $p"; return 1; }
}

for t in "${TARGETS[@]}"; do
  case "$t" in
    container)
      printf '  removing container ... '
      docker rm -f "$CONTAINER" >/dev/null 2>&1 && echo "done" || echo "failed" ;;
    launcher)
      printf '  removing launcher ... '
      rm -f "$LAUNCHER" && echo "done" || echo "failed" ;;
    work)
      printf '  removing %s ...\n' "$WORK"
      rm_path "$WORK" && echo "    done" ;;
    image)
      printf '  removing image ... '
      docker rmi "$IMAGE" >/dev/null 2>&1 && echo "done" \
        || echo "in use by another container, skipped" ;;
    model)
      printf '  removing checkpoint ...\n'
      rm_path "$SNAP_DIR" && echo "    done"
      # Only drop the cache root when it is this project's own and now holds no
      # other models -- a shared HF_HOME must survive intact.
      if [ "$HF_CACHE" = "$PROJECT/.cache/huggingface" ] \
         && [ -z "$(find "$HF_CACHE/hub" -maxdepth 1 -name 'models--*' -print -quit 2>/dev/null)" ]; then
        rm_path "$PROJECT/.cache" && echo "    removed empty $PROJECT/.cache"
      fi ;;
  esac
done

printf '\n%s=== done ===%s\n' "$G" "$N"
df -h --output=avail "$PROJECT" | tail -1 | xargs echo "  free on this filesystem:"
echo
