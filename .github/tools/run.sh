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

main() {
  # Get the git root directory
  GIT_ROOT=$(git rev-parse --show-toplevel)
  if [ $? -ne 0 ]; then
    log_error "run" "Failed to get git root directory"
    return 1
  fi

  # Change to the git root directory
  pushd "$GIT_ROOT" >/dev/null || {
    log_error "run" "Failed to change to git root directory"
    return 1
  }

  uv pip install --compile-bytecode --no-progress rich pathspec pyyaml bentoml

  # Run the required commands
  log_info "run" "Running generate.py..."
  uv run .github/tools/generate.py
  if [ $? -ne 0 ]; then
    log_error "run" "Failed to run generate.py"
    popd >/dev/null
    return 1
  fi

  log_info "run" "Running build.py..."
  rm -rf $BENTOML_HOME/bentos && uv run .github/tools/build.py
  if [ $? -ne 0 ]; then
    log_error "run" "Failed to run build.py"
    popd >/dev/null
    return 1
  fi

  log_info "run" "Running push.py..."
  uv run .github/tools/push.py --context bentoml
  if [ $? -ne 0 ]; then
    log_error "run" "Failed to run push.py"
    popd >/dev/null
    return 1
  fi

  popd >/dev/null

  log_info "run" "All operations completed successfully."
  return 0
}

main
exit $?
