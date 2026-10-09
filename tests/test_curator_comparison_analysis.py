"""Synthetic local records test model isolation, audit reuse and signed comparisons."""

import hashlib
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import pytest
from test_paper_storage_analysis import (
    convert_to_paper,
)
from test_paper_storage_analysis import (
    synthetic_assets as synthetic_assets_fixture,
)
from test_storage_ablation_analysis import create_arm, rewrite_episode, save

synthetic_assets = synthetic_assets_fixture
SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_curator_comparison.py"
SPEC = importlib.util.spec_from_file_location("curator_comparison_analysis", SCRIPT)
comparison = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = comparison
SPEC.loader.exec_module(comparison)


@pytest.fixture
def curator_arms(tmp_path, synthetic_assets, monkeypatch):
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
    roots = []
    for name, results in zip(("baseline", "candidate"), outcomes):
        root = create_arm(tmp_path / name, "judge", results)
        convert_to_paper(root, synthetic_assets)
        roots.append(root)
    path = roots[1] / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["config"]["curator"]["model"] = "gpt-6.1-sol"
    save(path, manifest)
    monkeypatch.setattr(comparison.storage, "load_paper_assets", lambda directory: synthetic_assets)
    return roots


def compare(arms):
    return comparison.build_comparison(*arms, expected_tasks=6, expected_batch_size=2)


def test_signed_curator_comparison_reuses_quality_gate_and_preserves_input_files(curator_arms):
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for root in curator_arms
        for path in root.rglob("*")
        if path.is_file()
    }
    report = compare(curator_arms)
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before} == before
    assert report["kind"] == "matched_native_alfworld_curator_model_comparison"
    assert report["benchmark"] is False  # A six-task fixture cannot claim the seen140 benchmark.
    assert report["difference_definition"] == "candidate minus baseline"
    assert report["executor_model"] == report["judge_model"] == "gpt-5.5"
    assert report["curator_models"] == {"baseline": "gpt-5.5", "candidate": "gpt-6.1-sol"}
    assert report["protocol"]["matched_config_except"] == [
        "curator.model",
        "experiment.output_dir",
    ]
    assert report["protocol"]["store_policy"] == "judge"
    assert report["protocol"]["labels_visible"] is False
    assert report["total_episodes"] == 36
    assert report["baseline_episodes"] == report["candidate_episodes"] == 18
    assert report["new_model_evaluated_episodes"] == 18
    deltas = [-100 / 6, 100 / 3, 0]
    assert [run["delta_pp"] for run in report["per_seed"]] == pytest.approx(deltas)
    assert report["across_seed"]["delta_pp"] == pytest.approx(
        {"mean": statistics.mean(deltas), "std": statistics.stdev(deltas)}
    )
    assert report["per_seed"][0]["paired_counts"] == {
        "both_success": 2,
        "candidate_only": 0,
        "baseline_only": 1,
        "both_failure": 3,
    }
    for run in report["per_seed"]:
        for name in ("baseline", "candidate"):
            assert run[name]["storage"]["final_bank_size"] == 3
            assert run[name]["storage"]["judge_failed_stored_entries"] == 0
            assert run[name]["storage"]["retrieval"]["judge_failed_references"] == 0
    assert report["overall"]["candidate"]["usage_by_role"]["curator"]["prompt_tokens"] == 18000
    efficiency = report["overall"]["candidate"]["paper_efficiency"]
    assert efficiency["mean_executor_input_tokens_k"] == pytest.approx(900 / 18 / 1000)
    assert efficiency["mean_executor_output_tokens_k"] == pytest.approx(60 / 18 / 1000)
    assert efficiency["mean_executor_interaction_turns"] == pytest.approx(30 / 18)
    assert report["statistical_scope"]["pooled_significance_test"] is None
    assert report["analysis_model_api_calls"] == report["analysis_environment_steps"] == 0
    text = comparison.render_markdown(report)
    assert "仅 curator.model 和输出目录不同" in text
    assert "新 candidate 评测18条" in text
    assert "不代表本次新增36条模型评测" in text
    assert "非 benchmark" in text


@pytest.mark.parametrize(
    "section,field,value,error",
    [
        ("executor", "model", "different-executor", "mismatch beyond"),
        ("curator", "temperature", 0.0, "mismatch beyond"),
        ("curator", "max_tokens", 4096, "mismatch beyond"),
        ("curator", "base_url", "https://other.example/v1", "mismatch beyond"),
        ("experiment", "store_policy", "all", "judge filtering"),
        ("experiment", "warm_start", "warm-memory.jsonl", "cold start"),
        ("experiment", "prompt_profile", "legacy-paraphrase", "paper-v1 is required"),
    ],
)
def test_other_config_changes_are_rejected(curator_arms, section, field, value, error):
    path = curator_arms[1] / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["config"][section][field] = value
    save(path, manifest)
    with pytest.raises(comparison.AnalysisError, match=error):
        compare(curator_arms)


@pytest.mark.parametrize("field", ["source_hashes", "packages", "python", "bm25"])
def test_source_and_runtime_changes_are_rejected(curator_arms, field):
    path = curator_arms[1] / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = {"changed": True}
    save(path, manifest)
    with pytest.raises(comparison.AnalysisError, match="mismatch beyond"):
        compare(curator_arms)


def test_same_curator_is_not_a_model_comparison(curator_arms):
    path = curator_arms[1] / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["config"]["curator"]["model"] = "gpt-5.5"
    save(path, manifest)
    with pytest.raises(comparison.AnalysisError, match="Curator models must differ"):
        compare(curator_arms)


def test_changed_game_data_provenance_is_rejected(curator_arms):
    path = curator_arms[1] / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["tasks"][0]["game_sha256"] = "changed-game-data"
    save(path, manifest)
    with pytest.raises(comparison.AnalysisError, match="mismatch beyond"):
        compare(curator_arms)


def test_exact_paper_message_tampering_is_rejected(curator_arms):
    def tamper(episode):
        episode["calls"][1]["messages"][0]["content"] += "Invented extra guidance."

    rewrite_episode(curator_arms[1], 0, 0, tamper)
    with pytest.raises(comparison.AnalysisError, match="executor messages"):
        compare(curator_arms)


def test_judge_gate_inconsistency_is_rejected(curator_arms):
    checkpoint = json.loads((curator_arms[1] / "seed_0/checkpoint.json").read_text())
    index = next(index for index, row in enumerate(checkpoint["results"]) if row["stored"])

    def tamper(episode):
        episode["judge"]["success"] = False
        episode["calls"][-1]["response"] = json.dumps(episode["judge"])

    rewrite_episode(curator_arms[1], 0, index, tamper)
    with pytest.raises(comparison.AnalysisError, match="storage gate"):
        compare(curator_arms)


def test_missing_executor_usage_propagates_null_without_losing_curator_cost(curator_arms):
    def omit(episode):
        episode["calls"][1]["prompt_tokens"] = None
        episode["usage"]["prompt_tokens"] = None
        episode["usage"]["by_role"]["executor"]["prompt_tokens"] = None
        episode["usage"]["complete"] = False

    rewrite_episode(curator_arms[1], 0, 0, omit)
    report = compare(curator_arms)
    assert (
        report["overall"]["candidate"]["paper_efficiency"]["mean_executor_input_tokens_k"] is None
    )
    assert report["across_seed"]["candidate_mean_executor_input_tokens_k"] == {
        "mean": None,
        "std": None,
    }
    assert report["overall"]["candidate"]["all_role_prompt_tokens"] is None
    assert report["overall"]["candidate"]["usage_by_role"]["curator"]["prompt_tokens"] == 18000
    assert (
        report["overall"]["baseline"]["paper_efficiency"]["mean_executor_input_tokens_k"]
        is not None
    )
    assert "| candidate | 50.00 ± 16.67% | 9 / 18 | 未知 |" in comparison.render_markdown(report)


def test_partial_candidate_cannot_produce_a_benchmark_report(curator_arms, tmp_path, monkeypatch):
    results = curator_arms[1] / "seed_2/results.jsonl"
    results.write_text("\n".join(results.read_text().splitlines()[:-1]) + "\n")
    monkeypatch.setattr(comparison, "PROJECT", tmp_path)
    output = tmp_path / "outputs/comparison"
    with pytest.raises(SystemExit) as error:
        comparison.main(
            [
                *map(str, curator_arms),
                "--expected-tasks",
                "6",
                "--expected-batch-size",
                "2",
                "--output-dir",
                str(output),
            ]
        )
    assert error.value.code == 1
    assert not output.exists()


def test_raw_local_report_cannot_be_written_to_public_docs(curator_arms, tmp_path, monkeypatch):
    monkeypatch.setattr(comparison, "PROJECT", tmp_path)
    output = tmp_path / "docs/results"
    with pytest.raises(SystemExit) as error:
        comparison.main([*map(str, curator_arms), "--output-dir", str(output)])
    assert error.value.code == 1
    assert not output.exists()


def test_analysis_cannot_write_inside_an_evaluation_directory(curator_arms, tmp_path, monkeypatch):
    monkeypatch.setattr(comparison, "PROJECT", tmp_path)
    inside_outputs = tmp_path / "outputs"
    inside_outputs.mkdir()
    protected = inside_outputs / "protected-arm"
    protected.mkdir()
    with pytest.raises(SystemExit) as error:
        comparison.main(
            [str(protected), str(curator_arms[1]), "--output-dir", str(protected / "analysis")]
        )
    assert error.value.code == 1
    assert not (protected / "analysis").exists()


def test_cli_writes_only_completed_local_comparison(curator_arms, tmp_path, monkeypatch):
    monkeypatch.setattr(comparison, "PROJECT", tmp_path)
    output = tmp_path / "outputs/comparison"
    assert (
        comparison.main(
            [
                *map(str, curator_arms),
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
    result = json.loads((output / "curator_comparison.json").read_text())
    assert result["benchmark"] is False
    assert result["new_model_evaluated_episodes"] == 18
    assert (output / "curator_comparison.md").exists()
