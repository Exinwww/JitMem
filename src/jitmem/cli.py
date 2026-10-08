from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .api import ChatAPIError, ChatClient
from .config import ConfigError, EnvironmentConfig, ExperimentConfig, RunConfig, load_config
from .environments import (
    SPLITS,
    ALFWorldEnvironment,
    FakeHouseholdEnvironment,
    discover_tasks,
    fake_tasks,
)
from .evaluation import evaluate, write_json
from .pipeline import MockChatClient, Pipeline


def _tasks(config):
    env = config.environment
    if env.backend == "mock":
        return fake_tasks(env.limit or 3)
    return discover_tasks(env.data_root, env.split, env.task_types, env.annotation_index, env.limit)


def _overrides(config, args):
    environment = {}
    experiment = {}
    for argument in ["limit", "split"]:
        if getattr(args, argument, None) is not None:
            environment[argument] = getattr(args, argument)
    for argument in ["method", "output_dir", "batch_size", "workers"]:
        if getattr(args, argument, None) is not None:
            experiment[argument] = getattr(args, argument)
    if getattr(args, "seeds", None) is not None:
        experiment["seeds"] = [int(value) for value in args.seeds.split(",")]
    return replace(
        config,
        environment=replace(config.environment, **environment),
        experiment=replace(config.experiment, **experiment),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="API-backed JITMEM ALFWorld reproduction")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["doctor", "list-tasks", "env-smoke", "evaluate", "api-check"]:
        sub = commands.add_parser(name)
        sub.add_argument("--config", default="configs/alfworld.example.toml")
        sub.add_argument("--split", choices=SPLITS)
        if name in {"list-tasks", "evaluate"}:
            sub.add_argument("--limit", type=int)
        if name == "list-tasks":
            sub.add_argument("--output")
        if name == "evaluate":
            sub.add_argument(
                "--method", choices=["jitmem", "no-memory", "raw-memory", "write-summary"]
            )
            sub.add_argument("--seeds", help="Comma-separated task-order seeds, e.g. 0,1,2")
            sub.add_argument("--batch-size", type=int)
            sub.add_argument(
                "--workers", type=int, help="Isolated episode processes within each frozen batch"
            )
            sub.add_argument("--output-dir")
            sub.add_argument("--resume", action="store_true")
    smoke = commands.add_parser("smoke", help="Offline fixture run; never an ALFWorld score")
    smoke.add_argument("--output-dir", default="outputs/mock_smoke")
    smoke.add_argument("--resume", action="store_true")
    report = commands.add_parser("compare", help="Compare completed experiment summaries")
    report.add_argument("outputs", nargs="+")
    args = parser.parse_args()
    try:
        if args.command == "compare":
            for output in args.outputs:
                data = json.loads((Path(output) / "summary.json").read_text())
                label = "ALFWorld" if data["benchmark"] else "FIXTURE"
                print(
                    f"{label} {data['method']} split={data['split']} "
                    f"SR={100 * data['success_rate_mean']:.2f}% "
                    f"std={100 * data['success_rate_std']:.2f} "
                    f"runs={len(data['runs'])} ({output})"
                )
            return
        if args.command == "smoke":
            config = RunConfig(
                environment=EnvironmentConfig(backend="mock", limit=3),
                experiment=ExperimentConfig(seeds=[0], batch_size=2, output_dir=args.output_dir),
            )
        else:
            config = _overrides(load_config(args.config), args)
        if args.command == "doctor":
            inventory = {}
            for split in SPLITS:
                tasks = discover_tasks(
                    config.environment.data_root, split, config.environment.task_types
                )
                inventory[split] = {
                    "tasks": len(tasks),
                    "by_type": dict(Counter(t.task_type for t in tasks)),
                }
            runtime = {}
            for package in ["alfworld", "textworld", "fast-downward-textworld"]:
                try:
                    runtime[package] = version(package)
                except PackageNotFoundError:
                    runtime[package] = "not installed"
            print(
                json.dumps(
                    {
                        "python": sys.version,
                        "runtime": runtime,
                        "dataset": inventory,
                        "executor_configured": bool(config.executor.model),
                        "curator_configured": bool(config.curator.model),
                    },
                    indent=2,
                )
            )
            return
        if args.command == "api-check":
            for role, model in [("executor", config.executor), ("curator", config.curator)]:
                result = ChatClient(model).complete(
                    [{"role": "user", "content": "Reply with API ready."}]
                )
                print(
                    f"{role}: response received; usage={result.prompt_tokens}+{result.completion_tokens}"
                )
            return
        tasks = _tasks(config)
        if args.command == "list-tasks":
            manifest = [asdict(task) for task in tasks]
            if args.output:
                write_json(Path(args.output), manifest)
                print(f"Saved {len(tasks)} tasks to {args.output}")
            else:
                print(json.dumps(manifest, ensure_ascii=False, indent=2))
            return
        factory = (
            ALFWorldEnvironment
            if config.environment.backend == "alfworld"
            else FakeHouseholdEnvironment
        )
        if args.command == "env-smoke":
            if not tasks:
                raise ValueError("No tasks discovered.")
            environment = factory(max_steps=config.experiment.max_steps)
            try:
                initial = environment.reset(tasks[0])
                next_state = environment.step("look")
                print(
                    json.dumps(
                        {
                            "runtime_interactive": True,
                            "benchmark_evaluation": False,
                            "task_id": tasks[0].task_id,
                            "initial_observation": initial.observation,
                            "initial_actions": list(initial.admissible_actions),
                            "look_observation": next_state.observation,
                            "native_success": next_state.success,
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
            finally:
                environment.close()
            return
        if config.environment.backend == "mock":
            executor = curator = MockChatClient()
        else:
            executor = ChatClient(config.executor)
            curator = (
                ChatClient(config.curator)
                if config.experiment.method in {"jitmem", "write-summary"}
                else executor
            )
        pipeline = Pipeline(config, executor, curator)
        summary = evaluate(
            config,
            tasks,
            pipeline,
            lambda: factory(max_steps=config.experiment.max_steps),
            resume=args.resume,
        )
        label = "ALFWorld" if summary["benchmark"] else "OFFLINE FIXTURE (not benchmark)"
        print(
            f"{label}: SR={100 * summary['success_rate_mean']:.2f}% "
            f"std={100 * summary['success_rate_std']:.2f}; output={config.experiment.output_dir}"
        )
    except (ConfigError, ChatAPIError, ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
