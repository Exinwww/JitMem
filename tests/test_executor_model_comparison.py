"""Synthetic recorded runs test fixed-judge isolation and the source behavior bridge."""

import copy
import hashlib
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import pytest
from test_paper_storage_analysis import convert_to_paper
from test_paper_storage_analysis import synthetic_assets as synthetic_assets_fixture
from test_storage_ablation_analysis import create_arm, rewrite_episode, save

synthetic_assets = synthetic_assets_fixture
SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_executor_model_comparison.py"
SPEC = importlib.util.spec_from_file_location("executor_model_comparison_analysis", SCRIPT)
comparison = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = comparison
SPEC.loader.exec_module(comparison)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def executor_arms(tmp_path, synthetic_assets, monkeypatch):
    outcomes = (
        {
            0: [True, True, False, False, True, False],
            1: [True, False, False, False, True, False],
            2: [True, False, True, False, True, False],
        },
        {
            0: [True, False, False, False, True, False],
            1: [True, True, True, False, True, False],
            2: [True, True, False, False, True, False],
        },
    )
    old_sources, new_sources = {}, {}
    for name in comparison.SOURCE_FILES:
        old_bytes = f"# synthetic old module {name}\n".encode()
        new_bytes = old_bytes + (
            b"# synthetic judge routing change\n" if name in comparison.SOURCE_CHANGES else b""
        )
        old_sources[name] = hashlib.sha256(old_bytes).hexdigest()
        new_sources[name] = hashlib.sha256(new_bytes).hexdigest()
        file = tmp_path / "src/jitmem" / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(new_bytes)
    roots = []
    for index, (name, results) in enumerate(zip(("baseline", "candidate"), outcomes)):
        root = create_arm(tmp_path / "outputs" / name, "all", results)
        convert_to_paper(root, synthetic_assets)
        manifest = json.loads((root / "manifest.json").read_text())
        manifest["config"]["curator"]["model"] = "gpt-6.1-sol"
        if index:
            manifest["config"]["judge"] = copy.deepcopy(manifest["config"]["executor"])
            manifest["config"]["executor"]["model"] = "gpt-6.1-sol"
        manifest["source_hashes"] = new_sources if index else old_sources
        content = {
            "config": manifest["config"],
            "tasks": manifest["tasks"],
            "source_hashes": manifest["source_hashes"],
            "packages": manifest["packages"],
            "python_version": list(sys.version_info[:3]),
            "warm_start_sha256": None,
            "prompt_assets": manifest["prompt_assets"],
        }
        manifest["fingerprint"] = hashlib.sha256(
            json.dumps(content, sort_keys=True).encode()
        ).hexdigest()
        save(root / "manifest.json", manifest)
        for seed in (0, 1, 2):
            checkpoint = json.loads((root / f"seed_{seed}/checkpoint.json").read_text())
            checkpoint["fingerprint"] = manifest["fingerprint"]
            save(root / f"seed_{seed}/checkpoint.json", checkpoint)
        roots.append(root)
    baseline, candidate = [json.loads((root / "manifest.json").read_text()) for root in roots]
    calls = sum(
        len(json.loads(file.read_text())["calls"])
        for file in roots[0].glob("seed_*/episodes/*.json")
    )
    bridge = tmp_path / "outputs/bridge.json"
    save(
        bridge,
        {
            "kind": "executor_gpt61_recorded_baseline_source_behavior_bridge",
            "passed": True,
            "errors": [],
            "benchmark": False,
            "replay_kind": "recorded_api_and_environment_fixture",
            "complete_420_baseline_episode_replay": False,
            "baseline_episodes_checked": 18,
            "recorded_calls_checked": calls,
            "baseline_fingerprint": baseline["fingerprint"],
            "baseline_manifest_sha256": digest(roots[0] / "manifest.json"),
            "baseline_source_hashes": old_sources,
            "candidate_source_hashes": new_sources,
            "allowed_changed_sources": comparison.SOURCE_CHANGES,
            "source_difference_paths": comparison.SOURCE_CHANGES,
            "prompt_assets": baseline["prompt_assets"],
            "variants_checked": comparison.BRIDGE_VARIANTS,
            "variants": {
                name: {
                    "episodes_checked": 18,
                    "calls_checked": calls,
                    "calls_by_role": {"curator": 18, "executor": calls - 36, "judge": 18},
                    "batches_checked": 9,
                    "effective_model_by_role": {
                        "curator": "gpt-6.1-sol",
                        "executor": "gpt-5.5",
                        "judge": "gpt-5.5",
                    },
                    "effective_judge_model": "gpt-5.5",
                    "all_messages_actions_storage_usage_batches_match": True,
                }
                for name in comparison.BRIDGE_VARIANTS
            },
            "model_api_calls": 0,
            "external_api_calls": 0,
            "native_environment_steps": 0,
        },
    )
    preflight = tmp_path / "outputs/preflight.json"
    save(
        preflight,
        {
            "passed": True,
            "freeze": {
                "baseline_manifest_sha256": digest(roots[0] / "manifest.json"),
                "baseline_fingerprint": baseline["fingerprint"],
                "config": candidate["config"],
                "effective_config_difference_paths": comparison.CONFIG_CHANGES,
                "source_hashes": new_sources,
                "baseline_source_hashes": old_sources,
                "source_difference_paths": comparison.SOURCE_CHANGES,
                "bridge_proof_sha256": digest(bridge),
                "prompt_assets": candidate["prompt_assets"],
                "tasks": candidate["tasks"],
                "packages": candidate["packages"],
                "python_version": list(sys.version_info[:3]),
                "expected_episodes": 18,
            },
        },
    )
    monkeypatch.setattr(comparison, "PROJECT", tmp_path)
    monkeypatch.setattr(comparison.storage, "load_paper_assets", lambda directory: synthetic_assets)
    return roots, bridge, preflight


def compare(fixture):
    roots, bridge, preflight = fixture
    return comparison.build_comparison(
        *roots, expected_tasks=6, expected_batch_size=2, bridge_proof=bridge, preflight=preflight
    )


def test_fixed_judge_full_storage_comparison_is_signed_descriptive_and_readonly(executor_arms):
    roots, bridge, preflight = executor_arms
    before = {path: digest(path) for root in roots for path in root.rglob("*") if path.is_file()}
    report = compare(executor_arms)
    assert {path: digest(path) for path in before} == before
    assert report["kind"] == "matched_native_alfworld_executor_model_comparison_fixed_judge"
    assert report["benchmark"] is False
    assert report["executor_models"] == {"baseline": "gpt-5.5", "candidate": "gpt-6.1-sol"}
    assert report["curator_model"] == "gpt-6.1-sol" and report["judge_model"] == "gpt-5.5"
    assert report["protocol"]["matched_effective_config_except"] == comparison.CONFIG_CHANGES
    assert (
        report["protocol"]["store_policy"] == "all" and report["protocol"]["labels_visible"] is True
    )
    assert report["protocol"]["same_core_sources"] is False
    assert report["protocol"]["source_difference_paths"] == comparison.SOURCE_CHANGES
    assert report["source_behavior_bridge"]["sha256"] == digest(bridge)
    assert report["preflight"]["sha256"] == digest(preflight)
    assert report["total_episodes"] == 36 and report["new_model_evaluated_episodes"] == 18
    delta = [-100 / 6, 100 / 3, 0]
    assert [run["delta_pp"] for run in report["per_seed"]] == pytest.approx(delta)
    assert report["across_seed"]["delta_pp"] == pytest.approx(
        {"mean": statistics.mean(delta), "std": statistics.stdev(delta)}
    )
    assert report["per_seed"][0]["paired_counts"] == {
        "both_success": 2,
        "candidate_only": 0,
        "baseline_only": 1,
        "both_failure": 3,
    }
    for run in report["per_seed"]:
        for name in ("baseline", "candidate"):
            assert run[name]["storage"]["final_bank_size"] == 6
            assert run[name]["storage"]["judge_failed_stored_entries"] == 3
    assert report["overall"]["candidate"]["paper_efficiency"][
        "mean_executor_input_tokens_k"
    ] == pytest.approx(900 / 18 / 1000)
    assert report["overall"]["candidate"]["usage_by_role"]["judge"]["prompt_tokens"] == 36000
    assert report["statistical_scope"]["pooled_significance_test"] is None
    assert report["analysis_model_api_calls"] == report["analysis_environment_steps"] == 0
    text = comparison.render_markdown(report)
    assert "核心源码并非完全相同" in text and "judge输出不决定" in text
    assert "新 candidate 评测18条" in text and "不代表本次新增36条模型评测" in text
    assert "非 benchmark" in text


@pytest.mark.parametrize(
    "section,field,value,error",
    [
        ("judge", "model", "gpt-6.1-sol", "Effective judge"),
        ("judge", "max_tokens", 123, "Effective config mismatch"),
        ("judge", "base_url", "https://different.example/v1", "Effective config mismatch"),
        ("executor", "temperature", 0.0, "Effective config mismatch"),
        ("executor", "max_tokens", 123, "Effective config mismatch"),
        ("curator", "model", "gpt-5.5", "Curator must"),
        ("experiment", "store_policy", "judge", "full storage required"),
        ("experiment", "warm_start", "bank.jsonl", "cold start"),
        ("experiment", "prompt_profile", "legacy-paraphrase", "paper-v1 is required"),
    ],
)
def test_changes_outside_effective_executor_model_are_rejected(
    executor_arms, section, field, value, error
):
    path = executor_arms[0][1] / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["config"][section][field] = value
    save(path, manifest)
    with pytest.raises(comparison.AnalysisError, match=error):
        compare(executor_arms)


def test_candidate_cannot_silently_route_judge_to_new_executor(executor_arms):
    path = executor_arms[0][1] / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["config"].pop("judge")
    save(path, manifest)
    with pytest.raises(comparison.AnalysisError, match="Effective judge"):
        compare(executor_arms)


@pytest.mark.parametrize("field", ["packages", "python", "tasks", "unexpected_routing_metadata"])
def test_all_other_manifest_metadata_remains_matched(executor_arms, field):
    path = executor_arms[0][1] / "manifest.json"
    manifest = json.loads(path.read_text())
    if field == "tasks":
        manifest[field][0]["game_sha256"] = "different-game"
    else:
        manifest[field] = "different-provenance"
    save(path, manifest)
    with pytest.raises(comparison.AnalysisError, match="mismatch beyond"):
        compare(executor_arms)


def test_fifth_source_change_cannot_hide_behind_bridge(executor_arms):
    path = executor_arms[0][1] / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["source_hashes"]["api.py"] = "1" * 64
    save(path, manifest)
    with pytest.raises(comparison.AnalysisError, match="exactly the four"):
        compare(executor_arms)


@pytest.mark.parametrize(
    "field,value",
    [
        ("passed", False),
        ("benchmark", True),
        ("baseline_episodes_checked", 17),
        ("recorded_calls_checked", 1),
        ("complete_420_baseline_episode_replay", True),
        ("model_api_calls", True),
    ],
)
def test_bridge_coverage_flags_and_api_counters_fail_closed(executor_arms, field, value):
    path = executor_arms[1]
    proof = json.loads(path.read_text())
    proof[field] = value
    save(path, proof)
    with pytest.raises(comparison.AnalysisError, match="Source behavior bridge differs"):
        compare(executor_arms)


def test_explicit_same_judge_variant_is_required(executor_arms):
    path = executor_arms[1]
    proof = json.loads(path.read_text())
    proof["variants"]["compatible_explicit_same_judge"][
        "all_messages_actions_storage_usage_batches_match"
    ] = False
    save(path, proof)
    with pytest.raises(comparison.AnalysisError, match="Source bridge compatible_explicit"):
        compare(executor_arms)


def test_preflight_must_bind_bridge_file_content(executor_arms):
    path = executor_arms[1]
    proof = json.loads(path.read_text())
    proof["additional_safe_diagnostic"] = "changed"
    save(path, proof)
    with pytest.raises(comparison.AnalysisError, match="bridge_proof_sha256"):
        compare(executor_arms)


def test_preflight_cannot_change_generation_budget(executor_arms):
    path = executor_arms[2]
    proof = json.loads(path.read_text())
    proof["freeze"]["config"]["judge"]["max_tokens"] = 123
    save(path, proof)
    with pytest.raises(comparison.AnalysisError, match="preflight freeze differs: config"):
        compare(executor_arms)


def test_current_candidate_core_must_match_frozen_bridge(executor_arms, tmp_path):
    (tmp_path / "src/jitmem/pipeline.py").write_text("# later change\n")
    with pytest.raises(comparison.AnalysisError, match="Current core differs"):
        compare(executor_arms)


def test_original_paper_messages_are_still_validated(executor_arms):
    rewrite_episode(
        executor_arms[0][1],
        0,
        0,
        lambda episode: episode["calls"][1]["messages"][0].update(
            content="Extra invented guidance."
        ),
    )
    with pytest.raises(comparison.AnalysisError, match="executor messages"):
        compare(executor_arms)


def test_full_labels_cannot_be_changed_independently_of_recorded_judge(executor_arms):
    def tamper(episode):
        episode["calls"][0]["messages"][-1]["content"] += "\nExecutor judge label: success"

    rewrite_episode(executor_arms[0][1], 0, 2, tamper)
    with pytest.raises(comparison.AnalysisError, match="visible labels disagree"):
        compare(executor_arms)


def test_unknown_executor_usage_remains_null_and_separate_from_judge(executor_arms):
    def omit(episode):
        episode["calls"][1]["prompt_tokens"] = None
        episode["usage"]["prompt_tokens"] = None
        episode["usage"]["by_role"]["executor"]["prompt_tokens"] = None
        episode["usage"]["complete"] = False

    rewrite_episode(executor_arms[0][1], 0, 0, omit)
    report = compare(executor_arms)
    assert (
        report["overall"]["candidate"]["paper_efficiency"]["mean_executor_input_tokens_k"] is None
    )
    assert report["across_seed"]["candidate_mean_executor_input_tokens_k"] == {
        "mean": None,
        "std": None,
    }
    assert report["overall"]["candidate"]["usage_by_role"]["judge"]["prompt_tokens"] == 36000
    assert "未知" in comparison.render_markdown(report)


def test_partial_records_cannot_write_a_report(executor_arms, tmp_path):
    roots, bridge, preflight = executor_arms
    results = roots[1] / "seed_2/results.jsonl"
    results.write_text("\n".join(results.read_text().splitlines()[:-1]) + "\n")
    output = tmp_path / "outputs/comparison"
    with pytest.raises(SystemExit) as error:
        comparison.main(
            [
                *map(str, roots),
                "--bridge-proof",
                str(bridge),
                "--preflight",
                str(preflight),
                "--expected-tasks",
                "6",
                "--expected-batch-size",
                "2",
                "--output-dir",
                str(output),
            ]
        )
    assert error.value.code == 1 and not output.exists()


@pytest.mark.parametrize("destination", ["docs/results", "outputs/candidate/analysis"])
def test_raw_comparison_cannot_write_public_or_evaluation_directories(
    executor_arms, tmp_path, destination
):
    roots, bridge, preflight = executor_arms
    output = tmp_path / destination
    with pytest.raises(SystemExit) as error:
        comparison.main(
            [
                *map(str, roots),
                "--bridge-proof",
                str(bridge),
                "--preflight",
                str(preflight),
                "--output-dir",
                str(output),
            ]
        )
    assert error.value.code == 1 and not output.exists()


def test_cli_writes_only_completed_ignored_local_analysis(executor_arms, tmp_path):
    roots, bridge, preflight = executor_arms
    output = tmp_path / "outputs/comparison"
    assert (
        comparison.main(
            [
                *map(str, roots),
                "--bridge-proof",
                str(bridge),
                "--preflight",
                str(preflight),
                "--expected-tasks",
                "6",
                "--expected-batch-size",
                "2",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads((output / "executor_comparison.json").read_text())["benchmark"] is False
    assert (output / "executor_comparison.md").exists()
