"""Google Colabで起動するPhysical AI Agent Cockpit。"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.agent_cockpit import AgentMode, AgentObservation, AgentOrchestrator
from src.agent_cockpit.evaluator import StateDeltaEvaluator
from src.agent_cockpit.planner import RuleBasedPlanner
from src.agent_cockpit.safety import ActionSafetyValidator
from src.agent_cockpit.storage import TraceStore


def _parse_vector(raw: str, *, name: str) -> list[float]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise gr.Error(f"{name}はJSON配列で入力してください: {exc}") from exc
    if not isinstance(value, list):
        raise gr.Error(f"{name}はJSON配列である必要があります。")
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise gr.Error(f"{name}には数値だけを入力してください。") from exc


def _build_orchestrator() -> AgentOrchestrator:
    return AgentOrchestrator(
        planner=RuleBasedPlanner(action_dim=7),
        validator=ActionSafetyValidator(action_dim=7),
        evaluator=StateDeltaEvaluator(),
        trace_store=TraceStore.from_environment(),
        executor=None,
    )


def _create_run(model_id: str, dataset_id: str) -> tuple[str, dict[str, Any]]:
    run_id = datetime.now().strftime("agent_%Y%m%d_%H%M%S")
    store = TraceStore.from_environment()
    run_dir = store.create_run(
        run_id,
        {
            "model_id": model_id or "not_selected",
            "dataset_id": dataset_id or "not_selected",
            "mode": "agent_cockpit",
        },
    )
    return run_id, {
        "run_id": run_id,
        "trace_root": str(run_dir),
        "drive_root": os.environ.get(
            "PHYSICAL_AI_DRIVE_ROOT",
            "/content/drive/MyDrive/PARC2026",
        ),
    }


def _run_agent_step(
    run_id: str,
    step_id: int,
    goal: str,
    instruction: str,
    state_json: str,
    after_state_json: str,
    mode: str,
    approved: bool,
    image_path: str | None,
) -> tuple[dict[str, Any], list[list[Any]], dict[str, Any], dict[str, Any], str]:
    if not run_id:
        raise gr.Error("先にRunを作成してください。")
    state = _parse_vector(state_json, name="Current State")
    after_state = (
        _parse_vector(after_state_json, name="After State")
        if after_state_json.strip()
        else None
    )
    observation = AgentObservation(
        step_id=int(step_id),
        instruction=instruction.strip(),
        state=state,
        image_path=image_path,
    )
    result = _build_orchestrator().run_step(
        run_id=run_id,
        goal=goal,
        observation=observation,
        mode=AgentMode(mode),
        approved=approved,
        after_state=after_state,
    )
    candidates = [
        [
            item["action_id"],
            item["label"],
            item["score"],
            json.dumps(item["action"]),
            item["expected_result"],
        ]
        for item in result["proposal"]["candidates"]
    ]
    selected = result["proposal"]["selected"]
    del selected
    status = (
        f"保存完了: {result['trace_path']} / "
        f"Safety={result['safety']['passed']} / "
        f"Execution={result['execution']['reason'] if not result['execution']['executed'] else 'executed'}"
    )
    return result["proposal"], candidates, result["safety"], result["evaluation"], status


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Physical AI Agent Cockpit") as app:
        gr.Markdown(
            "# Physical AI Agent Cockpit\n"
            "Colab GPU、Google Drive、OpenVLA Policyを接続するための検証用UIです。"
        )

        with gr.Tab("Dashboard"):
            with gr.Row():
                model_id = gr.Textbox(label="Model / Checkpoint ID", value="openvla-oft-pending")
                dataset_id = gr.Textbox(label="Dataset ID", value="selected_800_rlds")
            create_run_button = gr.Button("新しいRunを作成", variant="primary")
            run_id = gr.Textbox(label="Run ID", interactive=False)
            run_info = gr.JSON(label="Run / Drive情報")
            create_run_button.click(
                _create_run,
                inputs=[model_id, dataset_id],
                outputs=[run_id, run_info],
            )

        with gr.Tab("Dataset Explorer"):
            gr.Markdown(
                "Dataset Manifest、Episode映像、State、Action、Action chunk、"
                "RLDSBatchTransform前後のshapeをここへ接続します。"
            )
            with gr.Row():
                gr.Number(label="Episode ID", value=0, precision=0)
                gr.Number(label="Frame ID", value=0, precision=0)
            gr.Image(label="Front / Wrist Image", type="filepath")
            gr.JSON(label="Episode Manifest / Tensor Shape")

        with gr.Tab("Inference"):
            gr.Markdown(
                "OpenVLA-OFTの単発推論アダプター接続先です。"
                "予測Action、正解Action、軸別誤差、推論時間を表示します。"
            )
            gr.JSON(label="Inference input / output")

        with gr.Tab("Agent Cockpit"):
            with gr.Row():
                goal = gr.Textbox(
                    label="最終目標",
                    value="対象物を把持して指定位置へ置く",
                )
                instruction = gr.Textbox(
                    label="現在のInstruction",
                    value="pick up the target object",
                )
            observation_image = gr.Image(label="Observation Image", type="filepath")
            with gr.Row():
                step_id = gr.Number(label="Step ID", value=0, precision=0)
                mode = gr.Dropdown(
                    label="自律レベル",
                    choices=[item.value for item in AgentMode],
                    value=AgentMode.PROPOSE.value,
                )
                approved = gr.Checkbox(label="人がActionを承認", value=False)
            current_state = gr.Textbox(
                label="Current State JSON",
                value="[0, 0, 0, 0, 0, 0, 0, 0]",
            )
            after_state = gr.Textbox(
                label="After State JSON（未実行なら空欄）",
                value="",
            )
            run_step_button = gr.Button("次のActionを提案・記録", variant="primary")
            proposal = gr.JSON(label="Goal / Subgoal / Selected Action")
            candidates = gr.Dataframe(
                headers=["action_id", "label", "score", "action", "expected_result"],
                datatype=["str", "str", "number", "str", "str"],
                label="候補Action",
                interactive=False,
            )
            with gr.Row():
                safety = gr.JSON(label="Safety Validator")
                evaluation = gr.JSON(label="Evaluator")
            status = gr.Textbox(label="Trace保存結果", interactive=False)
            run_step_button.click(
                _run_agent_step,
                inputs=[
                    run_id,
                    step_id,
                    goal,
                    instruction,
                    current_state,
                    after_state,
                    mode,
                    approved,
                    observation_image,
                ],
                outputs=[proposal, candidates, safety, evaluation, status],
            )

        with gr.Tab("Run Trace"):
            gr.Markdown(
                "各StepはGoogle Driveの "
                "`40_experiments/agent_cockpit/<run_id>/steps/` に保存されます。"
            )
            gr.JSON(label="将来接続: Timeline / Failure analysis / Model comparison")

    return app


if __name__ == "__main__":
    build_app().launch(share=True, debug=True)
