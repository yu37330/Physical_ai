#!/usr/bin/env bash
# Run a fresh runtime all the way to a validated submission, unattended.
#
#   bash training/openvla_oft_a100/scripts/colab_run_detached.sh colab_pipeline.sh
#   STAGES="prepare stage_a submit" bash .../colab_pipeline.sh
#
# A recycled VM costs the whole of /content, and the recovery is four wrappers in
# a fixed order with nothing to decide between them. Each stage reports to
# Discord on its own, so a failure names itself without anyone watching the log.
#
# Everything expensive -- the RLDS conversion and the 2,400-file download behind
# it -- is restored from Drive rather than redone. Stage A S1 comes back from
# Drive too, so only S2 actually trains.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=colab_env.sh
source "$SCRIPT_DIR/colab_env.sh"
colab::notify_on_exit "Full pipeline"

cd "$PROJECT_ROOT"
colab::require_drive

STAGES="${STAGES:-setup prepare stage_a submit}"
STAGE_A_STAGE="${STAGE_A_STAGE:-s2}"
export DATASET_PROFILE="${DATASET_PROFILE:-full}"

# Up front: the run is unattended for close to two hours, and a name typed wrong
# would otherwise surface after setup has already spent twenty minutes.
for stage in $STAGES; do
  case "$stage" in
    setup|prepare|stage_a|submit) ;;
    *) echo "Unknown stage: $stage" >&2
       echo "Valid stages: setup prepare stage_a submit" >&2
       exit 2 ;;
  esac
done

colab::section "Pipeline"
echo "Stages:  $STAGES"
echo "Profile: $DATASET_PROFILE"
echo "Stage A: $STAGE_A_STAGE"
echo "Started: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

for stage in $STAGES; do
  case "$stage" in
    setup)    bash "$SCRIPT_DIR/colab_setup.sh" ;;
    prepare)  bash "$SCRIPT_DIR/colab_dataset_prepare.sh" ;;
    stage_a)  bash "$SCRIPT_DIR/colab_stage_a.sh" "$STAGE_A_STAGE" ;;
    submit)   RUN_DYNAMIC_SMOKE="${RUN_DYNAMIC_SMOKE:-1}" \
                bash "$SCRIPT_DIR/colab_submission_validate.sh" ;;
  esac
done

colab::report_disk
colab::section "Pipeline complete"
echo "Archive: $DRIVE_SUBMISSIONS/parc2026_track1_openvla_oft_plus.zip"
echo "SHA256:  $DRIVE_SUBMISSIONS/submission_sha256.txt"
echo "Download it from Drive and check the hash before submitting."
