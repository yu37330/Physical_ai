"""Google Colabで起動するPhysical AI Agent Cockpit。"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.agent_cockpit import AgentMode, AgentObservation, AgentOrchestrator
from src.agent_cockpit.autonomous_replay import (
    AutonomousReplayRunner,
    ReplayLoopConfig,
)
from src.agent_cockpit.dataset_explorer import (
    RLDSEpisodeReader,
    build_dataset_overview,
    build_episode_table,
)
from src.agent_cockpit.evaluator import StateDeltaEvaluator
from src.agent_cockpit.planner import RuleBasedPlanner
from src.agent_cockpit.policy_adapter import (
    OpenVLAPolicyAdapter,
    OpenVLAPolicyPlanner,
    compare_action_chunks,
)
from src.agent_cockpit.safety import ActionSafetyValidator
from src.agent_cockpit.storage import TraceStore
from src.agent_cockpit.visualization import (
    action_chunk_figure,
    action_chunk_rows,
    replay_metrics_figure,
)


DRIVE_ROOT = os.environ.get(
    "PHYSICAL_AI_DRIVE_ROOT",
    "/content/drive/MyDrive/PARC2026",
)


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


def _parse_matrix(raw: str, *, name: str) -> np.ndarray:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise gr.Error(f"{name}はJSON二次元配列で入力してください: {exc}") from exc
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 7:
        raise gr.Error(f"{name}は(T, 7)の二次元配列である必要があります。")
    return array


@lru_cache(maxsize=2)
def _cached_policy(checkpoint_dir: str) -> OpenVLAPolicyAdapter:
    return OpenVLAPolicyAdapter(checkpoint_dir)


def _build_orchestrator(
    planner_type: str,
    checkpoint_dir: str,
) -> AgentOrchestrator:
    if planner_type == "openvla":
        if not checkpoint_dir.strip():
            raise gr.Error("OpenVLA PlannerではCheckpoint directoryが必要です。")
        planner = OpenVLAPolicyPlanner(_cached_policy(checkpoint_dir.strip()))
    else:
        planner = RuleBasedPlanner(action_dim=7)
    return AgentOrchestrator(
        planner=planner,
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
        "drive_root": DRIVE_ROOT,
    }


def _load_dataset_overview(
    manifest_path: str,
    conversion_report_path: str,
    compatibility_report_path: str,
    selection_path: str,
) -> tuple[dict[str, Any], list[list[Any]]]:
    try:
        overview = build_dataset_overview(
            manifest_path=manifest_path,
            conversion_report_path=conversion_report_path.strip() or None,
            compatibility_report_path=compatibility_report_path.strip() or None,
        )
        rows = build_episode_table(selection_path) if selection_path.strip() else []
        return overview, rows
    except (FileNotFoundError, ValueError, KeyError) as exc:
        raise gr.Error(str(exc)) from exc


def _load_rlds_sample(
    dataset_dir: str,
    split: str,
    episode_offset: int,
    frame_id: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    str,
    str,
    list[list[Any]],
    dict[str, Any],
    dict[str, Any],
]:
    try:
        sample = RLDSEpisodeReader(dataset_dir).read_sample(
            split=split,
            episode_offset=int(episode_offset),
            frame_id=int(frame_id),
        )
    except (FileNotFoundError, ValueError, KeyError, IndexError, RuntimeError) as exc:
        raise gr.Error(str(exc)) from exc

    rows = action_chunk_rows(sample.action_chunk, prefix="target")
    state_json = json.dumps(sample.state.astype(float).tolist())
    action_json = json.dumps(sample.action.astype(float).tolist())
    state_payload = {
        "front_image": sample.front_image,
        "wrist_image": sample.wrist_image,
        "state": sample.state,
        "action": sample.action,
        "action_chunk": sample.action_chunk,
        "instruction": sample.instruction,
        "summary": sample.summary(),
    }
    return (
        sample.front_image,
        sample.wrist_image,
        state_json,
        action_json,
        rows,
        sample.summary(),
        state_payload,
    )


def _predict_openvla(
    checkpoint_dir: str,
    use_loaded_sample: bool,
    loaded_sample: dict[str, Any] | None,
    front_image: np.ndarray | None,
    wrist_image: np.ndarray | None,
    state_json: str,
    instruction: str,
    target_chunk_json: str,
) -> tuple[dict[str, Any], list[list[Any]], dict[str, Any], Any]:
    if use_loaded_sample:
        if not loaded_sample:
            raise gr.Error("Dataset Explorerで先にRLDS Sampleを読み込んでください。")
        front = loaded_sample["front_image"]
        wrist = loaded_sample["wrist_image"]
        state = loaded_sample["state"]
        resolved_instruction = str(loaded_sample["instruction"])
        target = np.asarray(loaded_sample["action_chunk"], dtype=np.float32)
    else:
        if front_image is None or wrist_image is None:
            raise gr.Error("Front imageとWrist imageを入力してください。")
        front = front_image
        wrist = wrist_image
        state = _parse_vector(state_json, name="State")
        resolved_instruction = instruction.strip()
        target = (
            _parse_matrix(target_chunk_json, name="Target Action chunk")
            if target_chunk_json.strip()
            else None
        )

    try:
        result = _cached_policy(checkpoint_dir.strip()).predict_rlds(
            front_image=front,
            wrist_image=wrist,
            state=state,
            instruction=resolved_instruction,
        )
        metrics = (
            compare_action_chunks(result.action_chunk, target)
            if target is not None
            else {"status": "target_not_provided"}
        )
        rows = action_chunk_rows(result.action_chunk, prefix="predicted")
        if target is not None:
            rows.extend(action_chunk_rows(target, prefix="target"))
        figure = action_chunk_figure(result.action_chunk, target)
        return result.to_dict(), rows, metrics, figure
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as exc:
        raise gr.Error(str(exc)) from exc


def _persist_observation_images(
    *,
    store: TraceStore,
    run_id: str,
    step_id: int,
    use_loaded_sample: bool,
    loaded_sample: dict[str, Any] | None,
    front_path: str | None,
    wrist_path: str | None,
) -> tuple[str | None, str | None]:
    if use_loaded_sample:
        if not loaded_sample:
            raise gr.Error("Dataset Explorerで先にRLDS Sampleを読み込んでください。")
        persisted_front = store.save_image_array(
            run_id,
            step_id,
            "front_image.png",
            loaded_sample["front_image"],
        )
        persisted_wrist = store.save_image_array(
            run_id,
            step_id,
            "wrist_image.png",
            loaded_sample["wrist_image"],
        )
        return str(persisted_front), str(persisted_wrist)

    persisted_front = (
        store.copy_artifact(run_id, step_id, front_path, "front_image.png")
        if front_path
        else None
    )
    persisted_wrist = (
        store.copy_artifact(run_id, step_id, wrist_path, "wrist_image.png")
        if wrist_path
        else None
    )
    return (
        str(persisted_front) if persisted_front else None,
        str(persisted_wrist) if persisted_wrist else None,
    )


def _run_agent_step(
    run_id: str,
    step_id: int,
    goal: str,
    instruction: str,
    state_json: str,
    after_state_json: str,
    mode: str,
    approved: bool,
    planner_type: str,
    checkpoint_dir: str,
    use_loaded_sample: bool,
    loaded_sample: dict[str, Any] | None,
    front_image_path: str | None,
    wrist_image_path: str | None,
) -> tuple[
    dict[str, Any],
    list[list[Any]],
    dict[str, Any],
    dict[str, Any],
    str,
    Any,
]:
    if not run_id:
        raise gr.Error("先にRunを作成してください。")
    resolved_step_id = int(step_id)
    if use_loaded_sample:
        if not loaded_sample:
            raise gr.Error("Dataset Explorerで先にRLDS Sampleを読み込んでください。")
        state = np.asarray(loaded_sample["state"], dtype=np.float32).astype(float).tolist()
        resolved_instruction = str(loaded_sample["instruction"])
    else:
        state = _parse_vector(state_json, name="Current State")
        resolved_instruction = instruction.strip()

    after_state = (
        _parse_vector(after_state_json, name="After State")
        if after_state_json.strip()
        else None
    )
    store = TraceStore.from_environment()
    persisted_front, persisted_wrist = _persist_observation_images(
        store=store,
        run_id=run_id,
        step_id=resolved_step_id,
        use_loaded_sample=use_loaded_sample,
        loaded_sample=loaded_sample,
        front_path=front_image_path,
        wrist_path=wrist_image_path,
    )
    observation = AgentObservation(
        step_id=resolved_step_id,
        instruction=resolved_instruction,
        state=state,
        image_path=persisted_front,
        metadata={
            "wrist_image_path": persisted_wrist,
            "source": "rlds_sample" if use_loaded_sample else "manual",
        },
    )
    try:
        result = _build_orchestrator(planner_type, checkpoint_dir).run_step(
            run_id=run_id,
            goal=goal,
            observation=observation,
            mode=AgentMode(mode),
            approved=approved,
            after_state=after_state,
        )
    except (FileNotFoundError, ValueError, RuntimeError, KeyError) as exc:
        raise gr.Error(str(exc)) from exc

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
    proposal_metadata = result["proposal"].get("metadata", {})
    action_chunk = proposal_metadata.get("action_chunk")
    figure = action_chunk_figure(action_chunk) if action_chunk else None
    status = (
        f"保存完了: {result['trace_path']} / "
        f"Safety={result['safety']['passed']} / "
        f"Execution={result['execution']['reason'] if not result['execution']['executed'] else 'executed'}"
    )
    return (
        result["proposal"],
        candidates,
        result["safety"],
        result["evaluation"],
        status,
        figure,
    )


def _run_autonomous_replay(
    parent_run_id: str,
    dataset_dir: str,
    split: str,
    episode_offset: int,
    start_frame: int,
    goal: str,
    planner_type: str,
    checkpoint_dir: str,
    max_steps: int,
    action_mae_threshold: float,
    max_consecutive_mismatches: int,
    max_repeated_actions: int,
    stop_on_safety_failure: bool,
) -> tuple[dict[str, Any], list[list[Any]], Any, str]:
    if not parent_run_id:
        raise gr.Error("Dashboardで先に親Runを作成してください。")
    replay_run_id = (
        f"{parent_run_id}_replay_"
        f"{datetime.now().strftime('%H%M%S_%f')}"
    )
    store = TraceStore.from_environment()
    try:
        episode = RLDSEpisodeReader(dataset_dir).read_episode(
            split=split,
            episode_offset=int(episode_offset),
        )
        store.create_run(
            replay_run_id,
            {
                "parent_run_id": parent_run_id,
                "mode": "offline_autonomous_replay",
                "dataset_dir": dataset_dir,
                "split": split,
                "episode_offset": int(episode_offset),
                "planner_type": planner_type,
                "checkpoint_dir": checkpoint_dir.strip(),
            },
        )
        config = ReplayLoopConfig(
            max_steps=int(max_steps),
            action_mae_threshold=float(action_mae_threshold),
            max_consecutive_mismatches=int(max_consecutive_mismatches),
            max_repeated_actions=int(max_repeated_actions),
            stop_on_safety_failure=bool(stop_on_safety_failure),
        )
        runner = AutonomousReplayRunner(
            orchestrator=_build_orchestrator(planner_type, checkpoint_dir),
            trace_store=store,
            config=config,
        )
        summary = runner.run(
            run_id=replay_run_id,
            goal=goal,
            episode=episode,
            start_frame=int(start_frame),
        )
    except (FileNotFoundError, ValueError, KeyError, IndexError, RuntimeError) as exc:
        raise gr.Error(str(exc)) from exc

    rows = [
        [
            row["loop_step"],
            row["frame_id"],
            row["action_mae"],
            row["action_rmse"],
            row["action_match"],
            row["safety_passed"],
            row["repeated_action_count"],
            row["consecutive_mismatches"],
            row.get("latency_ms"),
            row["stop_reason"],
        ]
        for row in summary["steps"]
    ]
    figure = replay_metrics_figure(
        summary["steps"],
        float(action_mae_threshold),
    )
    status = (
        f"Replay完了: {replay_run_id} / "
        f"steps={summary['executed_steps']} / stop={summary['stop_reason']} / "
        f"summary={summary['summary_path']}"
    )
    return summary, rows, figure, status


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Physical AI Agent Cockpit") as app:
        loaded_sample = gr.State(value=None)
        gr.Markdown(
            "# Physical AI Agent Cockpit\n"
            "Colab GPU、Google Drive、RLDS、OpenVLA Policyを接続する検証用UIです。"
        )

        with gr.Tab("Dashboard"):
            with gr.Row():
                model_id = gr.Textbox(
                    label="Model / Checkpoint ID",
                    value="openvla-oft-pending",
                )
                dataset_id = gr.Textbox(
                    label="Dataset ID",
                    value="selected_800_rlds",
                )
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
                "Dataset Manifest、選定Episode、変換済みRLDSの画像・State・Actionを確認します。"
            )
            manifest_path = gr.Textbox(
                label="Dataset Manifest path",
                value=f"{DRIVE_ROOT}/20_processed/dataset_manifest.json",
            )
            conversion_report_path = gr.Textbox(
                label="Conversion report path（任意）",
                value=f"{DRIVE_ROOT}/20_processed/rlds_conversion_report.json",
            )
            compatibility_report_path = gr.Textbox(
                label="OpenVLA compatibility report path（任意）",
                value=f"{DRIVE_ROOT}/20_processed/openvla_compatibility_report.json",
            )
            selection_path = gr.Textbox(
                label="Episode selection path（任意）",
                value=f"{DRIVE_ROOT}/20_processed/libero_plus_selection_v001.json",
            )
            load_overview_button = gr.Button("ManifestとEpisode一覧を読み込む")
            dataset_overview = gr.JSON(label="Dataset overview")
            episode_table = gr.Dataframe(
                headers=["episode_index", "split", "suite", "instruction", "frames"],
                datatype=["number", "str", "str", "str", "number"],
                label="Selected Episodes",
                interactive=False,
            )
            load_overview_button.click(
                _load_dataset_overview,
                inputs=[
                    manifest_path,
                    conversion_report_path,
                    compatibility_report_path,
                    selection_path,
                ],
                outputs=[dataset_overview, episode_table],
            )

            rlds_dataset_dir = gr.Textbox(
                label="TFDS Builder directory",
                placeholder="conversion reportのdata_dirを指定",
            )
            with gr.Row():
                rlds_split = gr.Dropdown(
                    label="Split",
                    choices=["train", "val"],
                    value="train",
                )
                episode_offset = gr.Number(
                    label="Episode offset in split",
                    value=0,
                    precision=0,
                )
                frame_id = gr.Number(label="Frame ID", value=0, precision=0)
            load_sample_button = gr.Button("RLDS Sampleを読み込む", variant="primary")
            with gr.Row():
                sample_front = gr.Image(label="Front Image", type="numpy")
                sample_wrist = gr.Image(label="Wrist Image", type="numpy")
            with gr.Row():
                sample_state_json = gr.Textbox(label="State (8)")
                sample_action_json = gr.Textbox(label="Action (7)")
            sample_chunk = gr.Dataframe(
                headers=["source", "step", "a0", "a1", "a2", "a3", "a4", "a5", "a6"],
                datatype=["str", "number"] + ["number"] * 7,
                label="Target Action chunk",
                interactive=False,
            )
            sample_summary = gr.JSON(label="Sample / Tensor summary")
            load_sample_button.click(
                _load_rlds_sample,
                inputs=[rlds_dataset_dir, rlds_split, episode_offset, frame_id],
                outputs=[
                    sample_front,
                    sample_wrist,
                    sample_state_json,
                    sample_action_json,
                    sample_chunk,
                    sample_summary,
                    loaded_sample,
                ],
            )

        with gr.Tab("Inference"):
            gr.Markdown(
                "OpenVLA-OFTのAction chunkを推論し、RLDS教師Actionとの誤差を確認します。"
            )
            inference_checkpoint = gr.Textbox(
                label="Checkpoint directory",
                placeholder=f"{DRIVE_ROOT}/30_models/...",
            )
            use_loaded_inference = gr.Checkbox(
                label="Dataset Explorerで読み込んだSampleを使用",
                value=True,
            )
            with gr.Row():
                inference_front = gr.Image(label="Manual Front Image", type="numpy")
                inference_wrist = gr.Image(label="Manual Wrist Image", type="numpy")
            inference_state = gr.Textbox(
                label="Manual State JSON",
                value="[0, 0, 0, 0, 0, 0, 0, 0]",
            )
            inference_instruction = gr.Textbox(
                label="Manual Instruction",
                value="pick up the target object",
            )
            target_chunk_json = gr.Textbox(
                label="Manual Target Action chunk JSON（任意）",
                value="",
            )
            inference_button = gr.Button("OpenVLA推論を実行", variant="primary")
            inference_result = gr.JSON(label="Inference result")
            inference_rows = gr.Dataframe(
                headers=["source", "step", "a0", "a1", "a2", "a3", "a4", "a5", "a6"],
                datatype=["str", "number"] + ["number"] * 7,
                label="Predicted / Target chunk",
                interactive=False,
            )
            inference_metrics = gr.JSON(label="Action chunk error")
            inference_plot = gr.Plot(label="Action chunk plot")
            inference_button.click(
                _predict_openvla,
                inputs=[
                    inference_checkpoint,
                    use_loaded_inference,
                    loaded_sample,
                    inference_front,
                    inference_wrist,
                    inference_state,
                    inference_instruction,
                    target_chunk_json,
                ],
                outputs=[
                    inference_result,
                    inference_rows,
                    inference_metrics,
                    inference_plot,
                ],
            )

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
            with gr.Row():
                planner_type = gr.Dropdown(
                    label="Planner",
                    choices=["rule_based", "openvla"],
                    value="rule_based",
                )
                agent_checkpoint = gr.Textbox(
                    label="OpenVLA Checkpoint directory",
                    placeholder=f"{DRIVE_ROOT}/30_models/...",
                )
                use_loaded_agent = gr.Checkbox(
                    label="Dataset Explorer Sampleを使用",
                    value=False,
                )
            with gr.Row():
                observation_image = gr.Image(
                    label="Manual Front Image",
                    type="filepath",
                )
                wrist_observation_image = gr.Image(
                    label="Manual Wrist Image",
                    type="filepath",
                )
            with gr.Row():
                step_id = gr.Number(label="Step ID", value=0, precision=0)
                mode = gr.Dropdown(
                    label="自律レベル",
                    choices=[item.value for item in AgentMode],
                    value=AgentMode.PROPOSE.value,
                )
                approved = gr.Checkbox(label="人がActionを承認", value=False)
            current_state = gr.Textbox(
                label="Manual Current State JSON",
                value="[0, 0, 0, 0, 0, 0, 0, 0]",
            )
            after_state = gr.Textbox(
                label="After State JSON（未実行なら空欄）",
                value="",
            )
            run_step_button = gr.Button("次のActionを提案・記録", variant="primary")
            proposal = gr.JSON(label="Goal / Subgoal / Selected Action / Policy metadata")
            candidates = gr.Dataframe(
                headers=["action_id", "label", "score", "action", "expected_result"],
                datatype=["str", "str", "number", "str", "str"],
                label="候補Action",
                interactive=False,
            )
            with gr.Row():
                safety = gr.JSON(label="Safety Validator")
                evaluation = gr.JSON(label="Evaluator")
            agent_plot = gr.Plot(label="OpenVLA Action chunk")
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
                    planner_type,
                    agent_checkpoint,
                    use_loaded_agent,
                    loaded_sample,
                    observation_image,
                    wrist_observation_image,
                ],
                outputs=[
                    proposal,
                    candidates,
                    safety,
                    evaluation,
                    status,
                    agent_plot,
                ],
            )

        with gr.Tab("Autonomous Replay"):
            gr.Markdown(
                "記録済みRLDS Episodeを順に観測し、各FrameでAIが次Actionを再計画します。"
                "これは実機や因果的シミュレーションではなく、オフライン軌跡評価です。"
            )
            replay_dataset_dir = gr.Textbox(
                label="TFDS Builder directory",
                placeholder="Dataset Explorerと同じBuilder directory",
            )
            with gr.Row():
                replay_split = gr.Dropdown(
                    label="Split",
                    choices=["train", "val"],
                    value="train",
                )
                replay_episode_offset = gr.Number(
                    label="Episode offset",
                    value=0,
                    precision=0,
                )
                replay_start_frame = gr.Number(
                    label="Start frame",
                    value=0,
                    precision=0,
                )
            with gr.Row():
                replay_planner = gr.Dropdown(
                    label="Planner",
                    choices=["rule_based", "openvla"],
                    value="rule_based",
                )
                replay_checkpoint = gr.Textbox(
                    label="OpenVLA Checkpoint directory",
                    placeholder=f"{DRIVE_ROOT}/30_models/...",
                )
            replay_goal = gr.Textbox(
                label="最終目標",
                value="記録Episodeのタスクを完了する",
            )
            with gr.Row():
                replay_max_steps = gr.Number(
                    label="最大Step",
                    value=5,
                    precision=0,
                )
                replay_mae_threshold = gr.Number(
                    label="Action MAE閾値",
                    value=0.25,
                )
                replay_max_mismatches = gr.Number(
                    label="連続誤差超過で停止",
                    value=3,
                    precision=0,
                )
                replay_max_repeated = gr.Number(
                    label="同一Action反復で停止",
                    value=3,
                    precision=0,
                )
            replay_stop_safety = gr.Checkbox(
                label="Safety違反で即停止",
                value=True,
            )
            replay_button = gr.Button("自律Replayを実行", variant="primary")
            replay_summary = gr.JSON(label="Replay summary")
            replay_rows = gr.Dataframe(
                headers=[
                    "loop_step",
                    "frame_id",
                    "action_mae",
                    "action_rmse",
                    "action_match",
                    "safety_passed",
                    "repeat_count",
                    "mismatch_count",
                    "latency_ms",
                    "stop_reason",
                ],
                datatype=[
                    "number",
                    "number",
                    "number",
                    "number",
                    "bool",
                    "bool",
                    "number",
                    "number",
                    "number",
                    "str",
                ],
                label="Observe → Plan → Evaluate → Replan",
                interactive=False,
            )
            replay_plot = gr.Plot(label="Action MAE timeline")
            replay_status = gr.Textbox(label="Replay保存結果", interactive=False)
            replay_button.click(
                _run_autonomous_replay,
                inputs=[
                    run_id,
                    replay_dataset_dir,
                    replay_split,
                    replay_episode_offset,
                    replay_start_frame,
                    replay_goal,
                    replay_planner,
                    replay_checkpoint,
                    replay_max_steps,
                    replay_mae_threshold,
                    replay_max_mismatches,
                    replay_max_repeated,
                    replay_stop_safety,
                ],
                outputs=[
                    replay_summary,
                    replay_rows,
                    replay_plot,
                    replay_status,
                ],
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
