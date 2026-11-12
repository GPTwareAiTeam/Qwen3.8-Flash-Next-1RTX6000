#!/usr/bin/env bash
# Launch TabbyAPI with the patched exllamav3 for Qwen3.8-Flash-Next on one GPU.
#
# Required:
#   TABBY=<dir>       TabbyAPI checkout (564f8fb + deploy/tabbyapi-prefill-chunk-size.patch)
#   ENGINE=<dir>      exllamav3 checkout (v1.6.4 + engine/patches), absolute path; put first on PYTHONPATH
# Optional:
#   PYTHON=<bin>      interpreter with torch + TabbyAPI requirements (default: python3)
#   CONFIG=<yml>      TabbyAPI config (default: deploy/config.yml next to this script)
#   GPU=<idx>         CUDA device (default 0)
#   (the listen port is network.port in CONFIG; TabbyAPI ignores --port when a config file is given)
#   WARMUP=0          skip the warm-up request (default 1)
#   EXL3_NGRAM_LOCK=0 stream the engram table from disk instead of locking it in RAM (default 1)
#   API_KEY=<key>     TabbyAPI API key for the readiness check and warm-up; default: api_key from
#                     $TABBY/api_tokens.yml, which TabbyAPI creates on first start (unused with disable_auth: true)
#
# Warm-up: once /v1/models lists the model, one 128-token greedy completion is sent before the server is handed
# over. With the DFlash2 drafter attached, a fresh server's numerics settle only after ~74 decoded tokens, so
# without this the very first user request can differ slightly (README, "Server history").
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
: "${TABBY:?set TABBY to the TabbyAPI checkout}"
: "${ENGINE:?set ENGINE to the patched exllamav3 checkout}"
PYTHON=${PYTHON:-python3}
CONFIG=${CONFIG:-$HERE/config.yml}
MODEL_NAME=${MODEL_NAME:-Qwen3.8-Flash-Next-exl3-3.18bpw_h6_ng8}
GPU=${GPU:-0}
# The port the warm-up talks to is read from CONFIG (network.port), the only place TabbyAPI takes it from with --config
PORT=$(sed -n '/^network:/,/^[^ ]/s/^  port: *\([0-9]*\).*/\1/p' "$CONFIG" | head -n1)
[ -n "$PORT" ] || { echo "launch: no network.port in $CONFIG" >&2; exit 2; }
WARMUP=${WARMUP:-1}
WARMUP_TIMEOUT=${WARMUP_TIMEOUT:-900}

case "$ENGINE" in /*) ;; *) echo "launch: ENGINE must be an absolute path (got $ENGINE)" >&2; exit 2 ;; esac
[ -d "$ENGINE/exllamav3" ] || { echo "launch: no exllamav3 package under $ENGINE" >&2; exit 2; }

export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH="$ENGINE${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-12.0}

# exllamav3 imports any prebuilt exllamav3_ext it finds before JIT-building the patched sources. A pip-installed
# exllamav3 in the environment would therefore run the patched Python against the unpatched extension.
ext_path=$(cd / && PYTHONPATH="$ENGINE" "$PYTHON" -c 'import importlib.util as u; s=u.find_spec("exllamav3_ext"); print(s.origin if s and s.origin else "")' 2>/dev/null || true)
case "$ext_path" in
  ""|"$ENGINE"/*) ;;
  *) echo "launch: a prebuilt exllamav3_ext outside ENGINE would shadow the patched build: $ext_path" >&2
     echo "launch: pip uninstall exllamav3 (this recipe runs exllamav3 from ENGINE)" >&2; exit 2 ;;
esac

# Engram (n-gram embedding) table: 48.3 GiB. Locked in RAM by default, which needs memlock >= 50 GiB
# (ulimit -l 52428800, or LimitMEMLOCK=52428800K in a systemd unit). With EXL3_NGRAM_LOCK=0 it is read from disk
# per forward; a cold first request can then stall under host memory pressure.
export EXL3_NGRAM_STREAM=1
export EXL3_NGRAM_PREFETCH=1
export EXL3_NGRAM_LOCK=${EXL3_NGRAM_LOCK:-1}
if [ "$EXL3_NGRAM_LOCK" != "0" ]; then
  memlock_kib=$(ulimit -l)
  if [ "$memlock_kib" != "unlimited" ] && [ "$memlock_kib" -lt 52428800 ]; then
    echo "launch: EXL3_NGRAM_LOCK=1 needs memlock >= 52428800 KiB (50 GiB); ulimit -l is $memlock_kib" >&2
    exit 2
  fi
fi

# MoE decode path
export EXL3_MOE_MTILE=0           # one fused launch instead of 32/64-row tiles (decode puts 1-2 rows per expert)
export EXL3_MOE_SHARED_COOP=0     # shared expert as its own graph (runs on the side stream below)
export EXL3_MOE_COOP_WIDE=1       # wide coop tile for the routed experts
export EXL3_MOE_COOP_ROWS=64      # verify windows up to 64 rows through the coop kernels
export EXL3_MOE_SHARED_SIDE=1     # shared expert overlaps the routed experts on a side stream
export EXL3_DFLASH_GRAPH=1        # DFlash2 drafter layer stack as a CUDA graph

cd "$TABBY"
echo "launch: gpu=$GPU port=$PORT engine=$ENGINE config=$CONFIG warmup=$WARMUP ngram_lock=$EXL3_NGRAM_LOCK"
"$PYTHON" -u main.py --config "$CONFIG" &
SERVER=$!
trap 'kill -TERM "$SERVER" 2>/dev/null || true' TERM INT

wait_for() {
  local what=$1; shift
  local t0; t0=$(date +%s)
  until "$@" >/dev/null 2>&1; do
    kill -0 "$SERVER" 2>/dev/null || { echo "launch: server exited before $what" >&2; wait "$SERVER"; exit 1; }
    if [ $(( $(date +%s) - t0 )) -ge "$WARMUP_TIMEOUT" ]; then
      echo "launch: timeout waiting for $what" >&2; kill -TERM "$SERVER"; wait "$SERVER"; exit 1
    fi
    sleep 2
  done
}
health_ok() { curl -fsS -m 5 "http://127.0.0.1:$PORT/health"; }
models_ready() { curl -fsS -m 5 "http://127.0.0.1:$PORT/v1/models" ${API_KEY:+-H "Authorization: Bearer $API_KEY"} | grep -q "\"$MODEL_NAME\""; }

if [ "$WARMUP" != "0" ]; then
  t0=$(date +%s)
  wait_for /health health_ok
  # With auth on and no API_KEY given, use the key TabbyAPI keeps in api_tokens.yml (it creates the file on first start)
  if [ -z "${API_KEY:-}" ] && [ -f "$TABBY/api_tokens.yml" ]; then
    API_KEY=$(sed -n 's/^api_key: *//p' "$TABBY/api_tokens.yml" | tr -d "\"'" | head -n1)
  fi
  wait_for "/v1/models listing $MODEL_NAME" models_ready
  body=$(printf '{"model":"%s","prompt":"The capital of France is","max_tokens":128,"min_tokens":128,"temperature":0}' "$MODEL_NAME")
  if curl -fsS -m 300 -H 'Content-Type: application/json' ${API_KEY:+-H "Authorization: Bearer $API_KEY"} \
      -d "$body" "http://127.0.0.1:$PORT/v1/completions" >/dev/null; then
    echo "launch: warm-up done after $(( $(date +%s) - t0 )) s"
  else
    echo "launch: warm-up request failed (server left running)" >&2
  fi
fi

wait "$SERVER"
