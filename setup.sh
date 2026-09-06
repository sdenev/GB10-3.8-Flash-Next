#!/usr/bin/env bash
# ============================================================================
#  setup.sh -- one-shot setup for Qwen3.8-Flash-Next on this DGX Spark.
#
#  Wraps install-3.8flash.sh with the steps it assumes you already did by hand:
#
#    1  preflight: GPU, docker, hf CLI, free disk
#    2  download the checkpoint (135 GB)     -- resumable, skipped if complete
#    3  pre-pull the container image (14 GB) -- see NOTE below
#    4  run install-3.8flash.sh              -- writes ~/launch-3.8flash.sh
#
#  Everything heavy lands INSIDE this project, like the 27B repo next door:
#
#    ./.cache/huggingface   135 GB  checkpoint      (HF_HOME)
#    ./.work                 13 GB  patches + HashK (WORK)
#    ~/launch-3.8flash.sh     4 KB  generated launcher, path hardcoded upstream
#
#  Override either with  HF_HOME=/elsewhere ./setup.sh  or  WORK=/elsewhere.
#  To reclaim the disk later: ./stop.sh, then sudo rm -rf .cache .work
#  (sudo because the HashK build runs as root inside the container).
#
#  NOTE on step 3: install-3.8flash.sh probes the GPU by running the image
#  before its own pull phase, with stderr sent to /dev/null. If the image is
#  absent that docker run pulls 14 GB with no output at all and looks hung for
#  ten minutes. Pulling first turns its phase 3 into "already present".
#
#  Idempotent -- safe to re-run after an interrupted download.
#  Then:  ./start.sh   ./stop.sh
# ============================================================================
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1

INSTALLER="install-3.8flash.sh"

# Track the installer's own values so these never drift apart.
MODEL_ID="$(sed -n 's/^MODEL_ID="\(.*\)"$/\1/p' "$INSTALLER" 2>/dev/null | head -1)"
IMAGE="$(sed -n 's/^IMAGE="\(.*\)"$/\1/p' "$INSTALLER" 2>/dev/null | head -1)"
: "${MODEL_ID:=RadixArk/Qwen3.8-Flash-Next-NVFP4}"
: "${IMAGE:=lmsysorg/sglang:qwen38flashnext}"

# Project-local by default. Exported so install-3.8flash.sh picks them up:
# it reads HF_HOME at :23 and WORK at :25, and seds the resolved cache path
# permanently into the launcher at :615.
PROJECT="$PWD"
export HF_HOME="${HF_HOME:-$PROJECT/.cache/huggingface}"
export WORK="${WORK:-$PROJECT/.work}"
HF_CACHE="$HF_HOME"
SNAP_DIR="$HF_CACHE/hub/models--${MODEL_ID//\//--}"
NEED_SAFETENSORS=206
NEED_GB=155   # 135 checkpoint + 14 image + ~13 HashK artifact, with headroom

G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; N=$'\033[0m'
say(){ printf '\n%s==> %s%s\n' "$G" "$1" "$N"; }
warn(){ printf '%s    %s%s\n' "$Y" "$1" "$N"; }
die(){ printf '\n%sFAIL%s %s\n' "$R" "$N" "$1" >&2; exit 1; }

# How many of the checkpoint's safetensors are present and finished.
have_shards(){ find "$SNAP_DIR" -name '*.safetensors' 2>/dev/null | wc -l; }
downloading(){ [ "$(find "$SNAP_DIR" -name '*.incomplete' 2>/dev/null | wc -l)" -gt 0 ]; }

cat <<HEAD

=== Qwen3.8-Flash-Next -- setup ===

  cache   $HF_CACHE
  work    $WORK
  model   $MODEL_ID
  image   $IMAGE

HEAD

# ---- 1. preflight ---------------------------------------------------------
say "1/4  Preflight"

command -v docker >/dev/null || die "docker not found"
docker info >/dev/null 2>&1 || die "cannot talk to the docker daemon (are you in the docker group?)"
echo "    docker            ok"

command -v nvidia-smi >/dev/null || die "nvidia-smi not found"
CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)"
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
[ "$CAP" = "12.1" ] || warn "GPU is $GPU (compute_cap $CAP), expected 12.1 / GB10. Untested here."
echo "    gpu               ${GPU:-unknown} (sm_${CAP//./})"

# hf lives in ~/.local/bin, which is not always on a non-login shell's PATH.
if ! command -v hf >/dev/null; then
  [ -x "$HOME/.local/bin/hf" ] && export PATH="$HOME/.local/bin:$PATH"
fi
command -v hf >/dev/null || die "the 'hf' CLI is not on PATH.
       Install it with:  pip install --upgrade huggingface_hub"
echo "    hf cli            $(command -v hf)"

mkdir -p "$HF_CACHE" "$WORK" || die "cannot create $HF_CACHE / $WORK"
for d in "$HF_CACHE" "$HF_CACHE/hub" "$WORK"; do
  [ -e "$d" ] || continue
  [ -w "$d" ] && continue
  die "$d is not writable by $(id -un) -- it is owned by $(stat -c '%U' "$d").
       The HashK build and the server run as root inside the container and
       write here. Fix with:
         sudo chown -R $(id -un):$(id -gn) $d"
done
FREE="$(df -BG --output=avail "$HF_CACHE" | tail -1 | tr -dc 0-9)"
[ "${FREE:-0}" -ge "$NEED_GB" ] || die "need ~${NEED_GB} GB free on $HF_CACHE, have ${FREE} GB"
echo "    free disk         ${FREE} GB"

# ---- 2. checkpoint --------------------------------------------------------
say "2/4  Checkpoint (135 GB)"

if [ "$(have_shards)" -eq "$NEED_SAFETENSORS" ] && ! downloading; then
  echo "    already complete: ${NEED_SAFETENSORS}/${NEED_SAFETENSORS} safetensors, skipping"
else
  HAVE="$(have_shards)"
  if [ "$HAVE" -gt 0 ]; then
    echo "    resuming: ${HAVE}/${NEED_SAFETENSORS} safetensors present"
  else
    echo "    downloading -- this is the long one, and it is resumable"
  fi
  hf download "$MODEL_ID" || die "download failed. Re-run ./setup.sh to resume."
  [ "$(have_shards)" -eq "$NEED_SAFETENSORS" ] \
    || die "got $(have_shards)/${NEED_SAFETENSORS} safetensors. Re-run ./setup.sh to resume."
fi

# ---- 3. container image ---------------------------------------------------
say "3/4  Container image (14 GB)"

if docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "    already present, skipping"
else
  docker pull "$IMAGE" || die "image pull failed"
fi

# ---- known-wrong bind target in the upstream launcher -----------------------
# install-3.8flash.sh mounts the flash_attn patch over a dist-packages path
# that this image does not use: sglang is installed into its own prefix at
# /opt/sglang, and /usr/local/lib/python3.12/dist-packages is not on sys.path
# at all. The directory exists but is empty, so Docker would happily create the
# file there and nothing would ever import it -- the sm_121 fix would silently
# not apply. Evidence for the replacement path, checked against this image:
#   * `import flash_attn.cute.flash_fwd` resolves to the /opt/sglang copy
#   * qwen_sparse_attn_backend.py:83 imports top-level `flash_attn.cute.*`
#   * the patch differs from the /opt/sglang copy by ONE line (the sm_121 TMA
#     guard `and mCuSeqlensQ is None`), and from sglang's vendored copy by 508
# Remove an entry here if upstream fixes its path; the check below will tell you.
MOUNT_FIXUPS="/usr/local/lib/python3.12/dist-packages/flash_attn/cute/flash_fwd.py|/opt/sglang/lib/python3.12/site-packages/flash_attn/cute/flash_fwd.py"

remap(){
  local p="$1" pair
  for pair in $MOUNT_FIXUPS; do
    if [ "$p" = "${pair%%|*}" ]; then printf '%s' "${pair#*|}"; return; fi
  done
  printf '%s' "$p"
}

# Pull every  -v "src":/dst:ro  target out of a file. grep -o rather than sed so
# it does not matter whether the -v flags sit on separate lines (quoted
# heredoc, as upstream writes it) or collapse onto one (unquoted heredoc).
extract_mounts(){ grep -o -- '-v "[^"]*":/[^:[:space:]]*:ro' "$1" | sed 's|.*":||; s|:ro$||'; }

# The launcher bind-mounts the patched files OVER four paths inside the image.
# Docker silently CREATES a missing bind target instead of erroring, so a path
# that moved upstream would leave sglang running stock kernels -- which the
# README says corrupt silently on sm_121. That is wrong answers, not a crash,
# so check the targets exist now rather than trusting them at boot.
# Read from the launcher template so this tracks the mounts it actually writes.
MOUNT_TARGETS="$(extract_mounts "$INSTALLER" | tr '\n' ' ')"
EFFECTIVE=""
for t in $MOUNT_TARGETS; do EFFECTIVE="$EFFECTIVE $(remap "$t")"; done
MOUNT_TARGETS="$EFFECTIVE"
if [ -z "${MOUNT_TARGETS// /}" ]; then
  warn "could not read the bind-mount targets from $INSTALLER -- skipping the check"
else
  MISSING="$(docker run --rm --entrypoint sh "$IMAGE" -c \
    "for p in $MOUNT_TARGETS; do [ -f \"\$p\" ] || echo \"\$p\"; done" 2>/dev/null)"
  if [ -n "$MISSING" ]; then
    die "the image is missing files the launcher mounts over:
$(printf '         %s\n' $MISSING)
       Docker would create these as empty paths and sglang would run UNPATCHED,
       which corrupts silently on sm_121. The image or the patch repo has moved
       upstream -- do not launch until this is resolved."
  fi
  echo "    bind targets      $(printf '%s\n' $MOUNT_TARGETS | wc -l)/4 present in the image"
fi

# ---- 4. install -----------------------------------------------------------
say "4/4  Running $INSTALLER"

# Called through bash on purpose: the file ships mode 644, so ./ would fail.
bash "$INSTALLER" || die "$INSTALLER failed"

LAUNCHER="$HOME/launch-3.8flash.sh"
[ -x "$LAUNCHER" ] || die "installer did not produce $LAUNCHER"

# The installer just wrote the upstream (wrong) path into the launcher. Rewrite
# it, then re-verify every mount target against the image -- so the launcher is
# checked as it will actually run, not as we hoped it was generated.
for pair in $MOUNT_FIXUPS; do
  w="${pair%%|*}"; r="${pair#*|}"
  if grep -qF "$w" "$LAUNCHER"; then
    sed -i "s|$w|$r|g" "$LAUNCHER" || die "could not rewrite the mount path in $LAUNCHER"
    echo "    remapped bind target -> $r"
  fi
done

FINAL="$(extract_mounts "$LAUNCHER" | tr '\n' ' ')"
BAD="$(docker run --rm --entrypoint sh "$IMAGE" -c \
  "for p in $FINAL; do [ -f \"\$p\" ] || echo \"\$p\"; done" 2>/dev/null)"
[ -z "$BAD" ] || die "after remapping, $LAUNCHER still mounts over missing paths:
$(printf '         %s\n' $BAD)"
# An empty or short list must not read as success: no mounts found means the
# launcher is not what we think it is, which is exactly the case this whole
# check exists to catch.
NFINAL="$(printf '%s\n' $FINAL | grep -c '^/' || true)"
[ "${NFINAL:-0}" -eq 4 ] || die "expected 4 bind mounts in $LAUNCHER, found ${NFINAL:-0}.
       The launcher is not shaped as expected -- inspect it before launching."
echo "    launcher mounts   ${NFINAL}/4 verified against the image"

cat <<DONE

${G}=== setup complete ===${N}

  Start:  ./start.sh
  Stop:   ./stop.sh
  Bench:  bash sgbench.sh

DONE
