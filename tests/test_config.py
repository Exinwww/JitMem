"""Configuration validation has no dependency on API credentials or ALFWorld."""

import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from jitmem.config import (
    ConfigError,
    EnvironmentConfig,
    ExperimentConfig,
    ModelConfig,
    RunConfig,
    load_config,
)


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def load(self, text: str) -> RunConfig:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment.toml"
            path.write_text(text, encoding="utf-8")
            return load_config(path)

    def test_defaults_allow_non_api_commands(self) -> None:
        config = self.load("")
        self.assertEqual(config.environment.backend, "alfworld")
        self.assertEqual(config.environment.split, "valid_seen")
        self.assertEqual(config.environment.task_types, [1, 2, 3, 4, 5, 6])
        self.assertEqual(config.experiment.seeds, [0, 1, 2])
        self.assertEqual(config.executor.model, "")
        self.assertEqual(config.executor.base_url, "https://api.openai.com/v1")
        self.assertEqual(config.curator.base_url, "https://api.openai.com/v1")
        self.assertEqual(config.executor.api_key_env, "OPENAI_API_KEY")
        self.assertEqual(config.curator.api_key_env, "OPENAI_API_KEY")
        self.assertFalse(config.executor.omit_temperature)
        self.assertEqual(config.executor.max_tokens, 4096)
        self.assertEqual(config.curator.max_tokens, 8192)
        json.dumps(config.to_dict())

    def test_explicit_values_and_curator_inheritance(self) -> None:
        config = self.load("""
[environment]
backend = "mock"
split = "valid_train"
task_types = [1, 3]
limit = 5
annotation_index = 2
[experiment]
method = "raw-memory"
seeds = [7]
batch_size = 2
retrieval_k = 0
task_adaptive = false
store_policy = "all"
warm_start = "memory.json"
[executor]
base_url = "https://example.test/v1"
model = "executor-model"
api_key_env = "TEST_JITMEM_KEY"
max_tokens = 100
token_limit_parameter = "max_completion_tokens"
[executor.extra_body]
top_p = 0.95
[curator]
model = "curator-model"
""")
        self.assertEqual(config.environment.limit, 5)
        self.assertEqual(config.environment.annotation_index, 2)
        self.assertEqual(config.experiment.retrieval_k, 0)
        self.assertFalse(config.experiment.task_adaptive)
        self.assertEqual(config.curator.model, "curator-model")
        self.assertEqual(config.curator.base_url, config.executor.base_url)
        self.assertEqual(config.curator.api_key_env, "TEST_JITMEM_KEY")
        self.assertEqual(config.curator.max_tokens, 8192)
        self.assertEqual(config.curator.extra_body, {"top_p": 0.95})
        config.curator.extra_body["top_p"] = 1.0
        self.assertEqual(config.executor.extra_body["top_p"], 0.95)

    def test_worker_count_defaults_and_toml_override(self) -> None:
        self.assertEqual(self.load("").experiment.workers, 1)
        config = self.load("[experiment]\nworkers=10")
        self.assertEqual(config.experiment.workers, 10)
        self.assertEqual(config.to_dict()["experiment"]["workers"], 10)

    def test_worker_count_must_be_a_positive_integer(self) -> None:
        for value in (0, -1, True, False, 1.5, "2"):
            with self.subTest(value=value):
                with self.assertRaises(ConfigError):
                    replace(ExperimentConfig(), workers=value)

    def test_common_environment_settings_override_both_toml_roles(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENAI_BASE_URL": "https://common.example.test/v1",
                "OPENAI_MODEL": "shared-model",
                "OPENAI_API_KEY": "common-secret",
            },
        ):
            config = self.load("""
[executor]
base_url = "https://toml-executor.example.test/v1"
model = "toml-executor"
api_key_env = "TOML_KEY"
[curator]
base_url = "https://toml-curator.example.test/v1"
model = "toml-curator"
api_key_env = "TOML_CURATOR_KEY"
""")
        for role in (config.executor, config.curator):
            self.assertEqual(role.base_url, "https://common.example.test/v1")
            self.assertEqual(role.model, "shared-model")
            self.assertEqual(role.api_key_env, "OPENAI_API_KEY")
        self.assertNotIn("common-secret", json.dumps(config.to_dict()))
        self.assertNotIn("common-secret", repr(config))

    def test_role_environment_settings_override_common_and_toml(self) -> None:
        values = {
            "OPENAI_BASE_URL": "https://common.example.test/v1",
            "OPENAI_MODEL": "shared-model",
            "OPENAI_API_KEY": "common-secret",
            "JITMEM_EXECUTOR_BASE_URL": "https://executor.example.test/v1",
            "JITMEM_EXECUTOR_MODEL": "executor-env-model",
            "JITMEM_EXECUTOR_API_KEY": "executor-secret",
            "JITMEM_CURATOR_BASE_URL": "https://curator.example.test/v1",
            "JITMEM_CURATOR_MODEL": "curator-env-model",
            "JITMEM_CURATOR_API_KEY": "curator-secret",
        }
        with patch.dict(os.environ, values):
            config = self.load('[executor]\nmodel="toml-executor"\n[curator]\nmodel="toml-curator"')
        for name, role in (("executor", config.executor), ("curator", config.curator)):
            self.assertEqual(role.base_url, f"https://{name}.example.test/v1")
            self.assertEqual(role.model, f"{name}-env-model")
            self.assertEqual(role.api_key_env, f"JITMEM_{name.upper()}_API_KEY")
        serialized = json.dumps(config.to_dict())
        for secret in ("common-secret", "executor-secret", "curator-secret"):
            self.assertNotIn(secret, serialized)
            self.assertNotIn(secret, repr(config))

    def test_common_curator_value_takes_precedence_over_executor_inheritance(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENAI_MODEL": "common-curator-model",
                "OPENAI_BASE_URL": "https://common.example.test/v1",
                "JITMEM_EXECUTOR_MODEL": "executor-only-model",
                "JITMEM_EXECUTOR_BASE_URL": "https://executor.example.test/v1",
            },
        ):
            config = self.load("")
        self.assertEqual(config.executor.model, "executor-only-model")
        self.assertEqual(config.curator.model, "common-curator-model")
        self.assertEqual(config.curator.base_url, "https://common.example.test/v1")

    def test_curator_inherits_executor_connection_and_role_specific_credentials(self) -> None:
        with patch.dict(
            os.environ,
            {
                "JITMEM_EXECUTOR_MODEL": "executor-only-model",
                "JITMEM_EXECUTOR_BASE_URL": "https://executor.example.test/v1",
                "JITMEM_EXECUTOR_API_KEY": "executor-only-secret",
            },
        ):
            config = self.load("")
        self.assertEqual(config.curator.model, config.executor.model)
        self.assertEqual(config.curator.base_url, config.executor.base_url)
        self.assertEqual(config.executor.api_key_env, "JITMEM_EXECUTOR_API_KEY")
        self.assertEqual(config.curator.api_key_env, "JITMEM_EXECUTOR_API_KEY")
        self.assertNotIn("executor-only-secret", json.dumps(config.to_dict()))

    def test_blank_role_environment_settings_fall_back_to_common(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENAI_MODEL": "shared-model",
                "OPENAI_API_KEY": "common-secret",
                "JITMEM_EXECUTOR_MODEL": "",
                "JITMEM_EXECUTOR_BASE_URL": "   ",
                "JITMEM_EXECUTOR_API_KEY": " ",
                "JITMEM_CURATOR_MODEL": " ",
                "JITMEM_CURATOR_API_KEY": "",
            },
        ):
            config = self.load("")
        for role in (config.executor, config.curator):
            self.assertEqual(role.model, "shared-model")
            self.assertEqual(role.api_key_env, "OPENAI_API_KEY")
            self.assertEqual(role.base_url, "https://api.openai.com/v1")

    def test_explicit_curator_credentials_override_executor_inheritance(self) -> None:
        with patch.dict(os.environ, {"JITMEM_EXECUTOR_API_KEY": "executor-only-secret"}):
            config = self.load('[curator]\napi_key_env="CUSTOM_CURATOR_KEY"')
        self.assertEqual(config.executor.api_key_env, "JITMEM_EXECUTOR_API_KEY")
        self.assertEqual(config.curator.api_key_env, "CUSTOM_CURATOR_KEY")
        self.assertNotIn("executor-only-secret", json.dumps(config.to_dict()))

    def test_environment_values_are_validated_and_can_replace_toml_defaults(self) -> None:
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "not-an-http-url"}):
            with self.assertRaises(ConfigError):
                self.load("")
        with patch.dict(os.environ, {"OPENAI_BASE_URL": "https://env.example.test/v1"}):
            config = self.load('[executor]\nbase_url="invalid-optional-value"')
            self.assertEqual(config.executor.base_url, "https://env.example.test/v1")

    def test_omit_temperature_can_be_set_in_toml(self) -> None:
        config = self.load("[executor]\nomit_temperature=true\n[curator]\nomit_temperature=false")
        self.assertTrue(config.executor.omit_temperature)
        self.assertFalse(config.curator.omit_temperature)

    def test_unknown_and_malformed_settings_are_rejected(self) -> None:
        cases = [
            "[judge]\nmodel = 'different-model'",
            "[unknown]\nvalue = 1",
            "[environment]\nlimt = 5",
            "environment = 'not-table'",
            "[experiment]\nmethod = 'react'",
            "[experiment]\nstore_policy = 'success'",
            "[executor]\nmax_tokens = true",
            "[executor]\nretries = -1",
            "[executor.extra_body]\nmessages = []",
            "[executor]\nbase_url = 'https://user:secret@example.test/v1'",
            "[executor]\nbase_url = 'file:///tmp/local'",
            "[executor]\ntemperature = 4.0",
            "[executor]\nomit_temperature = 'true'",
            "[environment]\nbackend = []",
            "[experiment]\nseeds = []",
            "[environment]\ntask_types = [0]",
            "[environment]\nlimit = 0",
            "[experiment]\ntask_adaptive = 'false'",
            "[environment]\nsplit = 'test'",
            "[broken",
        ]
        for text in cases:
            with self.subTest(text=text):
                with self.assertRaises(ConfigError):
                    self.load(text)

    def test_dataclass_validation_catches_invalid_numeric_ranges(self) -> None:
        cases = [
            (EnvironmentConfig(), {"annotation_index": -1}),
            (EnvironmentConfig(), {"task_types": [1, 1]}),
            (ExperimentConfig(), {"seeds": [0, 0]}),
            (ExperimentConfig(), {"batch_size": 0}),
            (ExperimentConfig(), {"max_steps": -2}),
            (ExperimentConfig(), {"retrieval_k": -1}),
            (ModelConfig(), {"timeout_seconds": 0}),
            (ModelConfig(), {"timeout_seconds": float("inf")}),
            (ModelConfig(), {"temperature": float("nan")}),
            (ModelConfig(), {"omit_temperature": 1}),
            (ModelConfig(), {"max_tokens": 0}),
            (ModelConfig(), {"retries": 11}),
            (ModelConfig(), {"api_key_env": "literal api key"}),
            (ModelConfig(), {"token_limit_parameter": "other"}),
            (ModelConfig(), {"extra_body": {"top_p": float("nan")}}),
        ]
        for config, changes in cases:
            with self.subTest(changes=changes):
                with self.assertRaises(ConfigError):
                    replace(config, **changes)

    def test_default_mutable_values_are_independent(self) -> None:
        first, second = RunConfig(), RunConfig()
        first.environment.task_types.append(1)
        first.experiment.seeds.append(9)
        first.executor.extra_body["top_p"] = 0.5
        self.assertEqual(second.environment.task_types, [1, 2, 3, 4, 5, 6])
        self.assertEqual(second.experiment.seeds, [0, 1, 2])
        self.assertEqual(second.executor.extra_body, {})


if __name__ == "__main__":
    unittest.main()
