#!/usr/bin/env bash

set -euo pipefail

ERROR_COLOR="\033[0;31m"
LOG_COLOR="\033[0;32m"
WARN_COLOR="\033[0;34m"
DEBUG_COLOR="\033[0;35m"
RESET_COLOR="\033[0m"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

log() {
  local level=$1
  local caller=$2
  local message=$3
  caller=$(echo "$caller" | tr '[:lower:]' '[:upper:]')
  local color=""
  case $level in
  "ERROR") color=$ERROR_COLOR ;;
  "INFO") color=$LOG_COLOR ;;
  "WARN" | "WARNING") color=$WARN_COLOR ;;
  "DEBUG") color=$DEBUG_COLOR ;;
  *) color=$RESET_COLOR ;;
  esac
  echo -e "${color}[${caller}]${RESET_COLOR} ${message}"
}

log_info() { log "INFO" "$1" "$2"; }
log_warn() { log "WARN" "$1" "$2"; }
log_error() { log "ERROR" "$1" "$2"; }
log_debug() { log "DEBUG" "$1" "$2"; }

usage() {
  cat <<EOF
Usage: tp.sh -tp <value> [options]

Required:
  -tp <num>              tensor parallel size

Options:
  -g, --gpus <num>       total GPU count (auto-detect if omitted)
  --ignore-error         skip GPU divisibility check and omit CUDA_VISIBLE_DEVICES
  -f, --force            force rebuild of bentos
  --flush                copy cache into built bentos instead of building
  -h, --help             display this help and exit
EOF
}

main() {
  TP_VALUE=""
  NUM_GPUS=0
  FORCE=false
  FLUSH=false
  IGNORE_ERROR=false
  while [[ $# -gt 0 ]]; do
    case $1 in
    -tp)
      TP_VALUE="$2"
      shift 2
      ;;
    -g | --gpus)
      NUM_GPUS="$2"
      shift 2
      ;;
    -h | --help)
      usage
      return 0
      ;;
    --ignore-error)
      IGNORE_ERROR=true
      shift
      ;;
    -f | --force)
      FORCE=true
      shift
      ;;
    --flush)
      FLUSH=true
      shift
      ;;
    *)
      log_error "tp" "Unknown argument $1"
      return 1
      ;;
    esac
  done
  if [[ -z "$TP_VALUE" ]]; then
    log_error "tp" "Usage: tp.sh -tp <value>"
    return 1
  fi

  if [[ "$NUM_GPUS" -le 0 ]]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
      NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
      log_info "tp" "Detected ${NUM_GPUS} GPU(s) via nvidia-smi; using as default for --gpus"
    else
      NUM_GPUS=1
      log_warn "tp" "nvidia-smi not found; defaulting --gpus to 1"
    fi
  fi

  if [[ "$IGNORE_ERROR" == false ]]; then
    if ((NUM_GPUS % TP_VALUE != 0)); then
      log_error "tp" "--gpus ($NUM_GPUS) must be divisible by tp ($TP_VALUE) (use --ignore-error to override)"
      return 1
    fi
  fi

  GIT_ROOT=$(git rev-parse --show-toplevel) || {
    log_error "tp" "Failed to get git root"
    return 1
  }
  pushd "$GIT_ROOT" >/dev/null || {
    log_error "tp" "Failed to change directory"
    return 1
  }

  uv pip install --compile-bytecode --no-progress rich pathspec pyyaml bentoml

  log_info "tp" "Running generate.py"
  uv run .github/tools/generate.py || {
    log_error "tp" "generate.py failed"
    popd >/dev/null
    return 1
  }

  MODEL_NAMES=()
  for f in *.yaml; do
    [[ -e "$f" ]] || continue
    tp_entry=$(yq -r '.args.tp // ""' "$f" 2>/dev/null)
    if [[ "$tp_entry" == "$TP_VALUE" ]]; then
      MODEL_NAMES+=("${f%.yaml}")
    fi
  done

  if [[ ${#MODEL_NAMES[@]} -eq 0 ]]; then
    log_error "tp" "No models found matching tp=$TP_VALUE"
    popd >/dev/null
    return 1
  fi

  OUTPUT_FILE="successful_tp_${TP_VALUE}.txt"

  if [[ $FLUSH == false ]]; then
    if [[ -f "$OUTPUT_FILE" && $FORCE == false ]]; then
      log_warn "tp" "Existing $OUTPUT_FILE found. Skip build step."
    else
      if [[ $FORCE == true ]]; then
        log_warn "tp" "Force rebuild requested. Removing existing bentos and rebuilding."
        rm -rf "${BENTOML_HOME:-$HOME/bentoml}/bentos" || true
      fi
      log_info "tp" "Building bentos: $OUTPUT_FILE"
      YATAI_T_VERSION=0.35.2 uv run .github/tools/build.py --output-name "$OUTPUT_FILE" "${MODEL_NAMES[@]}" || {
        log_error "tp" "build.py failed"
        popd >/dev/null
        return 1
      }
    fi
  else
    if [[ ! -f "$OUTPUT_FILE" ]]; then
      log_error "tp" "--flush specified but $OUTPUT_FILE not found"
      popd >/dev/null
      return 1
    fi
  fi

  if [[ $FLUSH == false ]]; then
    log_info "tp" "Warmup command for $OUTPUT_FILE (you should run twice)"
    PORT_BASE=8000
    COUNTER=0
    if [[ "$IGNORE_ERROR" == false ]]; then
      GROUP_COUNT=$((NUM_GPUS / TP_VALUE))
    fi
    while IFS= read -r TAG || [[ -n "$TAG" ]]; do
      [[ -z "$TAG" ]] && continue
      BENTO_PATH=$(bentoml get "$TAG" -o path | tr -d '\n')
      mkdir -p "$BENTO_PATH/.cache"

      if [[ "$IGNORE_ERROR" == false ]]; then
        GROUP_IDX=$((COUNTER % GROUP_COUNT))
        START_GPU=$((GROUP_IDX * TP_VALUE))
        END_GPU=$((START_GPU + TP_VALUE - 1))
        GPU_SET=$(seq -s, $START_GPU $END_GPU)
        CUR_PORT=$((PORT_BASE + GROUP_IDX))
        echo "CUDA_VISIBLE_DEVICES=$GPU_SET VLLM_LOGGING_LEVEL=DEBUG bentoml serve $TAG --port $CUR_PORT" >>"cmd.txt"
      else
        CUR_PORT=$((PORT_BASE + COUNTER))
        echo "VLLM_LOGGING_LEVEL=DEBUG bentoml serve $TAG --port $CUR_PORT" >>"cmd.txt"
      fi
      ((++COUNTER))
    done <"$OUTPUT_FILE"
    printf '\n' >>"cmd.txt"
  else
    CACHE_DIR="$HOME/.cache/vllm"
    if [[ -d "$CACHE_DIR" ]]; then
      while IFS= read -r TAG || [[ -n "$TAG" ]]; do
        [[ -z "$TAG" ]] && continue
        BENTO_PATH=$(bentoml get "$TAG" -o path | tr -d '\n')
        if [[ -n "$BENTO_PATH" && -d "$BENTO_PATH" ]]; then
          mkdir -p "$BENTO_PATH/.cache"
          rsync -a "$CACHE_DIR/" "$BENTO_PATH/.cache/vllm/"
          log_info "tp" "Copied cache into $TAG"
        else
          log_warn "tp" "Unable to resolve path for $TAG"
        fi
      done <"$OUTPUT_FILE"
    else
      log_warn "tp" "Cache directory $CACHE_DIR not found; skipping cache copy"
    fi
  fi

  popd >/dev/null
  log_info "tp" "All operations completed successfully"
  return 0
}

main "$@"
exit $?
