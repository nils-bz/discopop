# This file is part of the DiscoPoP software (http://www.discopop.tu-darmstadt.de)
#
# Copyright (c) 2020, Technische Universitaet Darmstadt, Germany
#
# This software may be modified and distributed under the terms of
# the 3-Clause BSD License.  See the LICENSE file in the package base
# directory for details.

import pytest

from discopop_library.ProjectManager.services.assistant_backends import (
    OPENAI_DEFAULT_BASE_URL,
    PLACEHOLDER_API_KEY,
    AssistantBackendConfig,
    ResolvedBackend,
    make_session,
    resolve_backend,
)
from discopop_library.ProjectManager.services.assistant_openai import OpenAICompatibleSession

LOCAL = {
    "DISCOPOP_ASSISTANT_PROVIDER": "openai",
    "DISCOPOP_ASSISTANT_MODEL": "Qwen/Qwen3.8-27B-FP8",
    "DISCOPOP_ASSISTANT_BASE_URL": "http://localhost:18000/v1",
}


def test_default_is_anthropic_with_the_sdks_own_settings() -> None:
    assert resolve_backend(AssistantBackendConfig(), {}) == ResolvedBackend("anthropic", None, None, None)


def test_environment_selects_an_openai_compatible_server() -> None:
    backend = resolve_backend(AssistantBackendConfig(), {**LOCAL, "DISCOPOP_ASSISTANT_API_KEY": "secret"})
    assert backend == ResolvedBackend("openai", "Qwen/Qwen3.8-27B-FP8", "http://localhost:18000/v1", "secret")
    assert "secret" not in backend.describe()


def test_flags_win_over_the_environment() -> None:
    config = AssistantBackendConfig(provider="openai", model="other", base_url="http://gpu:8000/v1")
    backend = resolve_backend(config, {**LOCAL, "DISCOPOP_ASSISTANT_PROVIDER": "anthropic"})
    assert (backend.provider, backend.model, backend.base_url) == ("openai", "other", "http://gpu:8000/v1")


def test_openai_defaults_url_and_placeholder_key() -> None:
    backend = resolve_backend(AssistantBackendConfig(provider="openai", model="gpt"), {})
    assert backend.base_url == OPENAI_DEFAULT_BASE_URL and backend.api_key == PLACEHOLDER_API_KEY


def test_provider_key_variable_is_the_fallback() -> None:
    backend = resolve_backend(AssistantBackendConfig(provider="openai", model="gpt"), {"OPENAI_API_KEY": "o"})
    assert backend.api_key == "o"
    backend = resolve_backend(AssistantBackendConfig(), {"ANTHROPIC_API_KEY": "a"})
    assert backend.api_key == "a"
    backend = resolve_backend(AssistantBackendConfig(), {"ANTHROPIC_API_KEY": "a", "DISCOPOP_ASSISTANT_API_KEY": "d"})
    assert backend.api_key == "d"


def test_blank_values_count_as_not_given() -> None:
    backend = resolve_backend(AssistantBackendConfig(provider=" "), {"DISCOPOP_ASSISTANT_MODEL": ""})
    assert backend == ResolvedBackend("anthropic", None, None, None)


def test_openai_without_model_is_an_error() -> None:
    with pytest.raises(ValueError, match="--assistant-model"):
        resolve_backend(AssistantBackendConfig(provider="openai"), {})


def test_unknown_provider_is_an_error() -> None:
    with pytest.raises(ValueError, match="Unknown assistant provider 'gemini'"):
        resolve_backend(AssistantBackendConfig(), {"DISCOPOP_ASSISTANT_PROVIDER": "gemini"})


def test_make_session_for_openai() -> None:
    session = make_session(resolve_backend(AssistantBackendConfig(), LOCAL))
    assert isinstance(session, OpenAICompatibleSession)
