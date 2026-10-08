"""Validated TOML configuration for API-based JitMem experiments.

Model names may be omitted while inspecting a dataset or running a mock smoke
test. The API client validates that a model and any required API key are present
when it is constructed. Judging uses the executor configuration; there is no
separate judge model setting.
"""

from __future__ import annotations

import copy
import json
import os
import re
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


class ConfigError(ValueError):
    """A configuration value is missing, malformed, or unsupported."""


def _integer(name: str, value: Any, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{name} must be an integer >= {minimum}")


def _number(name: str, value: Any, minimum: float, maximum: float | None = None) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number")
    if not (value >= minimum) or (maximum is not None and not (value <= maximum)):
        bounds = f" between {minimum} and {maximum}" if maximum is not None else f" >= {minimum}"
        raise ConfigError(f"{name} must be{bounds}")
    if value == float("inf"):
        raise ConfigError(f"{name} must be finite")


def _nonempty_string(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a nonempty string")


@dataclass(frozen=True, slots=True)
class ModelConfig:
    base_url: str = "https://api.openai.com/v1"
    model: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float | None = 1.0
    omit_temperature: bool = False
    max_tokens: int = 4096
    timeout_seconds: float = 90.0
    retries: int = 3
    token_limit_parameter: str = "max_tokens"
    extra_body: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _nonempty_string("base_url", self.base_url)
        try:
            parsed = urlsplit(self.base_url)
            port = parsed.port
        except ValueError as exc:
            raise ConfigError("base_url must be a valid HTTP(S) URL") from exc
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ConfigError("base_url must be a valid HTTP(S) URL")
        if (
            parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ConfigError(
                "base_url cannot contain credentials, query parameters, or a fragment"
            )
        if port is not None and port <= 0:
            raise ConfigError("base_url must have a valid port")
        if not isinstance(self.model, str):
            raise ConfigError("model must be a string")
        if not isinstance(self.api_key_env, str) or (
            self.api_key_env and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.api_key_env)
        ):
            raise ConfigError(
                "api_key_env must be an environment variable name, or empty for an unauthenticated endpoint"
            )
        if self.temperature is not None:
            _number("temperature", self.temperature, 0, 2)
        if not isinstance(self.omit_temperature, bool):
            raise ConfigError("omit_temperature must be a boolean")
        _integer("max_tokens", self.max_tokens, 1)
        _number("timeout_seconds", self.timeout_seconds, 0)
        if self.timeout_seconds == 0:
            raise ConfigError("timeout_seconds must be > 0")
        _integer("retries", self.retries)
        if self.retries > 10:
            raise ConfigError("retries must be <= 10")
        if not isinstance(self.token_limit_parameter, str) or self.token_limit_parameter not in {
            "max_tokens",
            "max_completion_tokens",
        }:
            raise ConfigError(
                "token_limit_parameter must be 'max_tokens' or 'max_completion_tokens'"
            )
        if not isinstance(self.extra_body, dict) or not all(
            isinstance(key, str) for key in self.extra_body
        ):
            raise ConfigError("extra_body must be a table with string keys")
        reserved = {
            "model",
            "messages",
            "temperature",
            "max_tokens",
            "max_completion_tokens",
            "stream",
            "n",
        }
        if reserved.intersection(self.extra_body):
            raise ConfigError(
                "extra_body cannot override model, messages, temperature, token limits, stream, or n"
            )
        try:
            json.dumps(self.extra_body, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ConfigError("extra_body must contain finite JSON-compatible values") from exc
        object.__setattr__(self, "extra_body", copy.deepcopy(self.extra_body))


@dataclass(frozen=True, slots=True)
class EnvironmentConfig:
    backend: str = "alfworld"
    data_root: str = "/Users/linbei/workspace/experiential_memory/data/alfworld"
    split: str = "valid_seen"
    task_types: list[int] = field(default_factory=lambda: [1, 2, 3, 4, 5, 6])
    limit: int | None = None
    annotation_index: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or self.backend not in {"alfworld", "mock"}:
            raise ConfigError("backend must be 'alfworld' or 'mock'")
        _nonempty_string("data_root", self.data_root)
        if not isinstance(self.split, str) or self.split not in {
            "train",
            "valid_train",
            "valid_seen",
            "valid_unseen",
        }:
            raise ConfigError("split must be train, valid_train, valid_seen, or valid_unseen")
        if not isinstance(self.task_types, list) or not self.task_types:
            raise ConfigError("task_types must be a nonempty list of task IDs 1 through 6")
        for value in self.task_types:
            _integer("task_types entry", value, 1)
            if value > 6:
                raise ConfigError("task_types entries must be in 1 through 6")
        if len(set(self.task_types)) != len(self.task_types):
            raise ConfigError("task_types cannot contain duplicate IDs")
        object.__setattr__(self, "task_types", list(self.task_types))
        if self.limit is not None:
            _integer("limit", self.limit, 1)
        _integer("annotation_index", self.annotation_index)


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    method: str = "jitmem"
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    batch_size: int = 10
    workers: int = 1
    max_steps: int = 30
    history_window: int = 3
    retrieval_k: int = 3
    store_policy: str = "judge"
    task_adaptive: bool = True
    output_dir: str = "outputs/alfworld"
    warm_start: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.method, str) or self.method not in {
            "jitmem",
            "no-memory",
            "raw-memory",
            "write-summary",
        }:
            raise ConfigError("method must be jitmem, no-memory, raw-memory, or write-summary")
        if not isinstance(self.seeds, list) or not self.seeds:
            raise ConfigError("seeds must be a nonempty list of integers")
        for seed in self.seeds:
            _integer("seed", seed)
        if len(set(self.seeds)) != len(self.seeds):
            raise ConfigError("seeds cannot contain duplicates")
        object.__setattr__(self, "seeds", list(self.seeds))
        _integer("batch_size", self.batch_size, 1)
        _integer("workers", self.workers, 1)
        _integer("max_steps", self.max_steps, 1)
        _integer("history_window", self.history_window)
        _integer("retrieval_k", self.retrieval_k)
        if not isinstance(self.store_policy, str) or self.store_policy not in {"judge", "all"}:
            raise ConfigError(
                "store_policy must be 'judge' or 'all'; ground truth cannot gate memory storage"
            )
        if not isinstance(self.task_adaptive, bool):
            raise ConfigError("task_adaptive must be a boolean")
        _nonempty_string("output_dir", self.output_dir)
        if self.warm_start is not None:
            _nonempty_string("warm_start", self.warm_start)


@dataclass(frozen=True, slots=True)
class RunConfig:
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    executor: ModelConfig = field(default_factory=ModelConfig)
    curator: ModelConfig = field(default_factory=lambda: ModelConfig(max_tokens=8192))

    def __post_init__(self) -> None:
        for name, expected in (
            ("environment", EnvironmentConfig),
            ("experiment", ExperimentConfig),
            ("executor", ModelConfig),
            ("curator", ModelConfig),
        ):
            if not isinstance(getattr(self, name), expected):
                raise ConfigError(f"{name} must be a {expected.__name__}")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible settings; only API key variable names are stored."""
        return asdict(self)


def _section(
    raw: dict[str, Any],
    name: str,
    cls: type[Any],
    defaults: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
) -> Any:
    values = raw.get(name, {})
    if not isinstance(values, dict):
        raise ConfigError(f"{name} must be a TOML table")
    known = {item.name for item in fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ConfigError(f"Unknown {name} settings: {', '.join(sorted(unknown))}")
    merged = dict(defaults or {})
    merged.update(values)
    merged.update(overrides or {})
    try:
        return cls(**merged)
    except ConfigError as exc:
        raise ConfigError(f"{name}: {exc}") from exc


def _environment_model_overrides(role: str) -> dict[str, str]:
    """Resolve connection settings while retaining only credential variable names."""
    overrides = {}
    prefix = f"JITMEM_{role.upper()}"
    for field_name, suffix in (("base_url", "BASE_URL"), ("model", "MODEL")):
        for variable in (f"{prefix}_{suffix}", f"OPENAI_{suffix}"):
            value = os.environ.get(variable, "").strip()
            if value:
                overrides[field_name] = value
                break
    for variable in (f"{prefix}_API_KEY", "OPENAI_API_KEY"):
        if os.environ.get(variable, "").strip():
            overrides["api_key_env"] = variable
            break
    return overrides


def load_config(path: str | Path) -> RunConfig:
    """Read configuration without requiring an API key or contacting a provider.

    For each role, nonempty JITMEM_<ROLE>_BASE_URL/MODEL/API_KEY environment
    variables override common OPENAI_BASE_URL/MODEL/API_KEY variables, which
    override optional TOML settings. Empty environment values are ignored.
    Omitted curator settings inherit executor settings, with an 8192-token curator
    budget, including the resolved executor credential variable name when no
    curator/common key is provided. Only the API key's variable name enters
    the configuration.
    A dedicated ``judge`` section is rejected because the paper's judge must
    use the executor model. Paths are interpreted by the caller relative to its
    working directory, so ``outputs/alfworld`` retains its ordinary CLI meaning.
    """
    config_path = Path(path).expanduser()
    try:
        with config_path.open("rb") as source:
            raw = tomllib.load(source)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {config_path}") from exc
    unknown = set(raw) - {"environment", "experiment", "executor", "curator"}
    if unknown:
        if "judge" in unknown:
            raise ConfigError("A judge section is unsupported: judging must use the executor model")
        raise ConfigError(f"Unknown configuration sections: {', '.join(sorted(unknown))}")
    executor = _section(
        raw, "executor", ModelConfig, overrides=_environment_model_overrides("executor")
    )
    curator_defaults = asdict(executor)
    curator_defaults["max_tokens"] = 8192
    return RunConfig(
        environment=_section(raw, "environment", EnvironmentConfig),
        experiment=_section(raw, "experiment", ExperimentConfig),
        executor=executor,
        curator=_section(
            raw, "curator", ModelConfig, curator_defaults, _environment_model_overrides("curator")
        ),
    )
