#!/usr/bin/env bash
# 公式ForkとPyPI transformers＋自前Patchが同じActionを出すかを照合する。
#
#   bash training/openvla_oft_a100/scripts/colab_action_parity.sh
#
# OPENVLA_OFFLINE_RUNTIME_ASSESSMENT.md 8章の最初の項目にあたる。ここがずれていると、
# 公開Benchmark値が提出物で再現しない。Latency・VRAMがどれだけ余裕でも意味が無くなる
# 種類の不一致なので、学習と並行してでも潰しておく。
#
# transformersを2回入れ替えるため、実行後の環境はPyPI版になる。学習へ戻る前に
# colab_setup.sh を回し直すこと。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=colab_env.sh
source "$SCRIPT_DIR/colab_env.sh"

cd "$PROJECT_ROOT"

CHECKPOINT="${PARITY_CHECKPOINT:-$BASE_CHECKPOINT}"
OUT_ROOT="${PARITY_ROOT:-$WORK_ROOT/action_parity}"
NUM_FRAMES="${PARITY_FRAMES:-8}"
# Action値はbfloat16推論なので、bit単位一致は期待できない。実装の食い違いは
# これよりはるかに大きな差になって出る。
ATOL="${PARITY_ATOL:-1e-3}"

if [[ ! -d "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  echo "Run colab_setup.sh first." >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"

colab::section "Reference: OpenVLA-OFT fork, native bidirectional attention"
python -m pip install -q --force-reinstall --no-deps \
  "transformers @ git+https://github.com/moojink/transformers-openvla-oft.git@${TRANSFORMERS_FORK_COMMIT:-bc339d9ad707454c0c115970db43c260067c61ab}"
python scripts/capture_action_chunks.py \
  --checkpoint-dir "$CHECKPOINT" \
  --num-frames "$NUM_FRAMES" \
  --no-attention-patch \
  --output "$OUT_ROOT/fork.npy"

colab::section "Candidate: PyPI transformers + bidirectional_attention.py"
python -m pip install -q --force-reinstall --no-deps "transformers==4.40.1"
python scripts/capture_action_chunks.py \
  --checkpoint-dir "$CHECKPOINT" \
  --num-frames "$NUM_FRAMES" \
  --output "$OUT_ROOT/pypi.npy"

colab::section "Comparison"
python submission/openvla_oft_offline/tools/compare_action_chunks.py \
  "$OUT_ROOT/fork.npy" "$OUT_ROOT/pypi.npy" --atol "$ATOL" \
  | tee "$OUT_ROOT/action_parity.json"

colab::section "Persist to Drive"
if colab::require_drive; then
  colab::persist "$OUT_ROOT/action_parity.json" \
    "$DRIVE_EXPERIMENTS/action_parity/action_parity.json"
fi

colab::section "Action parity passed"
echo "Reports: $OUT_ROOT"
echo "transformers is now the PyPI build; re-run colab_setup.sh before training."
