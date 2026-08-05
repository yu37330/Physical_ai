#!/usr/bin/env bash
# Start a long wrapper so it survives losing the notebook connection.
#
#   bash training/openvla_oft_a100/scripts/colab_run_detached.sh \
#     DATASET_PROFILE=full colab_stage_a.sh s2
#
#   bash training/openvla_oft_a100/scripts/colab_run_detached.sh --status
#   bash training/openvla_oft_a100/scripts/colab_run_detached.sh --tail
#
# A `!command` in a notebook cell is a child of the kernel, so a dropped VS Code
# connection takes half an hour of training with it. setsid detaches the run from
# the kernel's process group; the Discord notification then reports the outcome
# whether or not anything is still watching.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=colab_env.sh
source "$SCRIPT_DIR/colab_env.sh"

LOG_DIR="$WORK_ROOT/logs"
mkdir -p "$LOG_DIR"

case "${1:-}" in
  --status)
    colab::section "Detached runs"
    # The wrapper name is what makes a run recognisable; setsid and env are not.
    if pgrep -af "$SCRIPT_DIR/colab_" | grep -v "colab_run_detached"; then :; else
      echo "Nothing running."
    fi
    echo
    colab::section "Logs"
    ls -lt "$LOG_DIR" 2>/dev/null | head -10 || echo "No logs yet."
    exit 0
    ;;
  --tail)
    latest=$(ls -t "$LOG_DIR"/*.log 2>/dev/null | head -1 || true)
    if [[ -z "$latest" ]]; then
      echo "No logs under $LOG_DIR." >&2
      exit 1
    fi
    echo "$latest"
    tail -n "${TAIL_LINES:-40}" "$latest"
    exit 0
    ;;
  "")
    echo "Usage: $0 [VAR=value ...] <wrapper.sh> [args ...]" >&2
    echo "       $0 --status | --tail" >&2
    exit 2
    ;;
esac

# Leading VAR=value assignments go to env; the first bare word is the wrapper.
env_assignments=()
while [[ "${1:-}" == *=* ]]; do
  env_assignments+=("$1")
  shift
done
if [[ -z "${1:-}" ]]; then
  echo "No wrapper given after the environment assignments." >&2
  exit 2
fi

wrapper="$1"
shift
[[ "$wrapper" != /* ]] && wrapper="$SCRIPT_DIR/$wrapper"
if [[ ! -f "$wrapper" ]]; then
  echo "Wrapper not found: $wrapper" >&2
  exit 1
fi

LOG_FILE="$LOG_DIR/$(basename "$wrapper" .sh)-$(date +%Y%m%d-%H%M%S).log"

# < /dev/null so a prompt cannot leave it blocked forever with no terminal to
# answer from.
setsid nohup env "${env_assignments[@]}" bash "$wrapper" "$@" \
  > "$LOG_FILE" 2>&1 < /dev/null &
pid=$!
disown || true

colab::section "Started detached"
echo "PID:  $pid"
echo "Log:  $LOG_FILE"
echo
echo "Watch it with:"
echo "  bash $0 --tail"
echo "Discord will report the outcome even if the notebook disconnects."
