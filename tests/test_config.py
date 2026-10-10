"""Configuration validation has no dependency on API credentials or ALFWorld."""

import hashlib
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
        self.assertIsNone(config.judge)
        json.dumps(config.to_dict())

    def test_default_judge_serialization_preserves_exact_historical_shape(self) -> None:
        config = RunConfig(
            environment=EnvironmentConfig(data_root="/dataset"),
            experiment=ExperimentConfig(output_dir="/output", prompt_assets="/prompts"),
            executor=ModelConfig(model="executor-model"),
            curator=ModelConfig(model="curator-model", max_tokens=8192),
        )
        # Captured before the optional judge field was introduced. Preserving
        # this shape preserves the configuration portion of historical fingerprints;
        # changes to source hashes still correctly prevent a cross-version resume.
        serialized = config.to_dict()
        self.assertEqual(set(serialized), {"environment", "experiment", "executor", "curator"})
        self.assertEqual(
            hashlib.sha256(json.dumps(serialized, sort_keys=True).encode()).hexdigest(),
            "358c0ed05003991a79b32cc8dd2cd42cbc78000fcc3f2eaa701e1ea68b9f8498",
        )

    def test_common_variables_do_not_activate_independent_judge(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENAI_MODEL": "common-model",
                "OPENAI_BASE_URL": "https://common.example.test/v1",
                "OPENAI_API_KEY": "common-secret",
                "JITMEM_EXECUTOR_MODEL": "executor-only-model",
                "JITMEM_EXECUTOR_API_KEY": "executor-only-secret",
            },
        ):
            config = self.load("")
        self.assertIsNone(config.judge)
        self.assertNotIn("judge", config.to_dict())
        self.assertEqual(config.executor.model, "executor-only-model")
        self.assertEqual(config.executor.api_key_env, "JITMEM_EXECUTOR_API_KEY")

    def test_blank_dedicated_variables_do_not_activate_independent_judge(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENAI_MODEL": "common-model",
                "JITMEM_JUDGE_MODEL": " ",
                "JITMEM_JUDGE_BASE_URL": "",
                "JITMEM_JUDGE_API_KEY": "  ",
            },
        ):
            config = self.load("")
        self.assertIsNone(config.judge)
        self.assertNotIn("judge", config.to_dict())

    def test_explicit_independent_judge_inherits_resolved_executor_parameters(self) -> None:
        with patch.dict(os.environ, {"JITMEM_EXECUTOR_API_KEY": "executor-only-secret"}):
            config = self.load("""
[executor]
base_url = "https://executor.example.test/v1"
model = "gpt-6.1-sol"
max_tokens = 1234
temperature = 0.25
timeout_seconds = 12
retries = 1
token_limit_parameter = "max_completion_tokens"
[executor.extra_body]
top_p = 0.95
[judge]
model = "gpt-5.5"
""")
        self.assertEqual(config.executor.model, "gpt-6.1-sol")
        self.assertEqual(config.judge.model, "gpt-5.5")
        self.assertEqual(config.judge.base_url, config.executor.base_url)
        self.assertEqual(config.judge.api_key_env, "JITMEM_EXECUTOR_API_KEY")
        for field in (
            "max_tokens",
            "temperature",
            "timeout_seconds",
            "retries",
            "token_limit_parameter",
        ):
            self.assertEqual(getattr(config.judge, field), getattr(config.executor, field))
        self.assertEqual(config.judge.extra_body, {"top_p": 0.95})
        config.judge.extra_body["top_p"] = 0.5
        self.assertEqual(config.executor.extra_body["top_p"], 0.95)
        self.assertEqual(config.curator.extra_body["top_p"], 0.95)
        self.assertEqual(config.to_dict()["judge"]["model"], "gpt-5.5")
        self.assertNotIn("executor-only-secret", json.dumps(config.to_dict()))

    def test_explicit_empty_judge_section_activates_matching_independent_config(self) -> None:
        config = self.load('[executor]\nmodel="executor-model"\n[judge]\n')
        self.assertEqual(config.judge, config.executor)
        self.assertIsNot(config.judge, config.executor)
        self.assertIn("judge", config.to_dict())

    def test_explicit_judge_common_environment_precedence_matches_other_roles(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENAI_MODEL": "common-model",
                "OPENAI_BASE_URL": "https://common.example.test/v1",
                "OPENAI_API_KEY": "common-secret",
                "JITMEM_EXECUTOR_MODEL": "executor-only-model",
            },
        ):
            config = self.load("""
[judge]
model = "toml-judge"
base_url = "https://toml-judge.example.test/v1"
api_key_env = "TOML_JUDGE_KEY"
""")
        self.assertEqual(config.executor.model, "executor-only-model")
        self.assertEqual(config.judge.model, "common-model")
        self.assertEqual(config.judge.base_url, "https://common.example.test/v1")
        self.assertEqual(config.judge.api_key_env, "OPENAI_API_KEY")

    def test_judge_environment_overrides_common_and_toml_without_persisting_keys(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENAI_MODEL": "common-model",
                "OPENAI_BASE_URL": "https://common.example.test/v1",
                "OPENAI_API_KEY": "common-secret",
                "JITMEM_JUDGE_MODEL": " judge-env-model ",
                "JITMEM_JUDGE_BASE_URL": " https://judge.example.test/v1 ",
                "JITMEM_JUDGE_API_KEY": "judge-only-secret",
            },
        ):
            config = self.load('[judge]\nmodel="toml-judge"\napi_key_env="TOML_JUDGE_KEY"')
        self.assertEqual(config.judge.model, "judge-env-model")
        self.assertEqual(config.judge.base_url, "https://judge.example.test/v1")
        self.assertEqual(config.judge.api_key_env, "JITMEM_JUDGE_API_KEY")
        self.assertEqual(config.executor.model, "common-model")
        self.assertEqual(config.curator.model, "common-model")
        for value in ("common-secret", "judge-only-secret"):
            self.assertNotIn(value, json.dumps(config.to_dict()))
            self.assertNotIn(value, repr(config))

    def test_each_dedicated_judge_variable_can_activate_the_role_without_toml(self) -> None:
        cases = (
            ("JITMEM_JUDGE_MODEL", "judge-model", "model", "judge-model"),
            (
                "JITMEM_JUDGE_BASE_URL",
                "https://judge.example.test/v1",
                "base_url",
                "https://judge.example.test/v1",
            ),
            ("JITMEM_JUDGE_API_KEY", "judge-only-secret", "api_key_env", "JITMEM_JUDGE_API_KEY"),
        )
        for variable, value, field, expected in cases:
            with self.subTest(variable=variable):
                with patch.dict(
                    os.environ, {variable: value, "JITMEM_EXECUTOR_API_KEY": "executor-only-secret"}
                ):
                    config = self.load('[executor]\nmodel="executor-model"\nmax_tokens=321')
                self.assertIsInstance(config.judge, ModelConfig)
                self.assertEqual(getattr(config.judge, field), expected)
                self.assertEqual(config.judge.max_tokens, 321)
                if field != "model":
                    self.assertEqual(config.judge.model, "executor-model")
                if field != "api_key_env":
                    self.assertEqual(config.judge.api_key_env, "JITMEM_EXECUTOR_API_KEY")
                self.assertNotIn("judge-only-secret", json.dumps(config.to_dict()))

    def test_explicit_judge_credential_name_can_override_executor_inheritance(self) -> None:
        with patch.dict(os.environ, {"JITMEM_EXECUTOR_API_KEY": "executor-only-secret"}):
            config = self.load('[judge]\nmodel="judge-model"\napi_key_env="CUSTOM_JUDGE_KEY"')
        self.assertEqual(config.judge.api_key_env, "CUSTOM_JUDGE_KEY")
        self.assertEqual(config.executor.api_key_env, "JITMEM_EXECUTOR_API_KEY")

    def test_judge_environment_connection_is_validated(self) -> None:
        for endpoint in (
            "not-http",
            "https://user:private-fixture@example.test/v1",
            "https://judge.example.test/v1?api_key=private-fixture",
        ):
            with self.subTest(endpoint=endpoint):
                with patch.dict(os.environ, {"JITMEM_JUDGE_BASE_URL": endpoint}):
                    with self.assertRaises(ConfigError) as caught:
                        self.load("")
                self.assertNotIn("private-fixture", str(caught.exception))

    def test_run_config_judge_type_validation(self) -> None:
        self.assertIsNone(RunConfig(judge=None).judge)
        self.assertEqual(
            RunConfig(judge=ModelConfig(model="judge-model")).judge.model, "judge-model"
        )
        for value in ({}, "judge-model", False, 1, EnvironmentConfig()):
            with self.subTest(value_type=type(value).__name__):
                with self.assertRaisesRegex(ConfigError, "judge must be a ModelConfig or None"):
                    RunConfig(judge=value)

    def test_judge_cannot_embed_nested_credentials(self) -> None:
        cases = (
            '[judge.extra_body]\napi_key="private-fixture"',
            '[judge.extra_body.headers]\nAUTHORIZATION="private-fixture"',
            '[judge.extra_body]\nprovider=[{access_token="private-fixture"}]',
        )
        for text in cases:
            with self.subTest(text=text):
                with self.assertRaisesRegex(ConfigError, "judge: Credentials cannot") as caught:
                    self.load(text)
                self.assertNotIn("private-fixture", str(caught.exception))

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
            "judge = 'not-table'",
            "[judge]\nunsupported = 1",
            "[judge]\nmax_tokens = false",
            "[judge]\napi_key_env = 'literal api key'",
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

    def test_provider_parameters_cannot_persist_embedded_credentials(self) -> None:
        fake = "not-a-real-credential"
        cases = [
            {"api_key": fake},
            {"headers": {"AUTHORIZATION": fake}},
            {"provider": [{"access-token": fake}]},
            {"client_secret": fake},
            {"AWS_SECRET_ACCESS_KEY": fake},
        ]
        for extra in cases:
            with self.subTest(field=next(iter(extra))):
                with self.assertRaisesRegex(ConfigError, "Credentials cannot") as caught:
                    replace(ModelConfig(), extra_body=extra)
                self.assertNotIn(fake, str(caught.exception))
        with self.assertRaises(ConfigError) as caught:
            self.load('[executor.extra_body]\napi_key="not-a-real-credential"')
        self.assertNotIn(fake, str(caught.exception))
        allowed = replace(
            ModelConfig(),
            extra_body={"top_p": 0.95, "chat_template_kwargs": {"enable_thinking": False}},
        )
        self.assertFalse(allowed.extra_body["chat_template_kwargs"]["enable_thinking"])


if __name__ == "__main__":
    unittest.main()
