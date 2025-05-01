#!/usr/bin/env bash

ERROR_COLOR="\033[0;31m" # Red
LOG_COLOR="\033[0;32m"   # Green
WARN_COLOR="\033[0;34m"  # Blue
DEBUG_COLOR="\033[0;35m" # Purple
RESET_COLOR="\033[0m"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

log() {
  local level=$1
  local caller=$2
  local message=$3

  # Convert caller to uppercase
  caller=$(echo "$caller" | tr '[:lower:]' '[:upper:]')

  # Set color based on log level
  local color=""
  case $level in
  "ERROR")
    color=$ERROR_COLOR
    ;;
  "INFO")
    color=$LOG_COLOR
    ;;
  "WARN" | "WARNING")
    color=$WARN_COLOR
    ;;
  "DEBUG")
    color=$DEBUG_COLOR
    ;;
  *)
    color=$RESET_COLOR
    ;;
  esac

  # Print formatted log message
  echo -e "${color}[${caller}]${RESET_COLOR} ${message}"
}

# Log shortcut functions
log_info() {
  local caller=$1
  local message=$2
  log "INFO" "$caller" "$message"
}

log_warn() {
  local caller=$1
  local message=$2
  log "WARN" "$caller" "$message"
}

log_error() {
  local caller=$1
  local message=$2
  log "ERROR" "$caller" "$message"
}

log_debug() {
  local caller=$1
  local message=$2
  log "DEBUG" "$caller" "$message"
}

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
