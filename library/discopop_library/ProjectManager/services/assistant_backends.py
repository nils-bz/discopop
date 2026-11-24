# This file is part of the DiscoPoP software (http://www.discopop.tu-darmstadt.de)
#
# Copyright (c) 2020, Technische Universitaet Darmstadt, Germany
#
# This software may be modified and distributed under the terms of
# the 3-Clause BSD License.  See the LICENSE file in the package base
# directory for details.

"""PROTOTYPE: which model backs the assistant -- chosen by command line flags and environment variables.

Each setting is taken from the first of: the flag of ``discopop_project_manager --gui`` (``--assistant-provider``,
``--assistant-model``, ``--assistant-base-url``), the environment variable (``DISCOPOP_ASSISTANT_PROVIDER``,
``DISCOPOP_ASSISTANT_MODEL``, ``DISCOPOP_ASSISTANT_BASE_URL``), the default of the provider. The API key is never a
flag (it would show up in the process list and the shell history): ``DISCOPOP_ASSISTANT_API_KEY``, or the
provider's own variable (``ANTHROPIC_API_KEY`` resp. ``OPENAI_API_KEY``).

Providers:

- ``anthropic`` (default): Claude through the Anthropic SDK (``assistant_anthropic.py``)
- ``openai``: any server speaking the OpenAI Chat Completions API (``assistant_openai.py``); OpenAI itself when no
  base URL is given, else e.g. a vLLM, Ollama or LM Studio server; a model name is required
"""

import os
from dataclasses import dataclass
from typing import Mapping, Optional

from discopop_library.ProjectManager.services.assistant import ModelSession

PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_OPENAI = "openai"
PROVIDERS = (PROVIDER_ANTHROPIC, PROVIDER_OPENAI)
DEFAULT_PROVIDER = PROVIDER_ANTHROPIC

ENV_PROVIDER = "DISCOPOP_ASSISTANT_PROVIDER"
ENV_MODEL = "DISCOPOP_ASSISTANT_MODEL"
ENV_BASE_URL = "DISCOPOP_ASSISTANT_BASE_URL"
ENV_API_KEY = "DISCOPOP_ASSISTANT_API_KEY"
# the variables the providers' own tools read; used when ENV_API_KEY is not set
_PROVIDER_API_KEY_ENV = {PROVIDER_ANTHROPIC: "ANTHROPIC_API_KEY", PROVIDER_OPENAI: "OPENAI_API_KEY"}

OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
# PLACEHOLDER until a real key is set: sent when no key is configured, so a server that checks keys answers "401
# Unauthorized" (the error names DISCOPOP_ASSISTANT_API_KEY) and one that does not (a local vLLM or Ollama) works
PLACEHOLDER_API_KEY = "sk-discopop-placeholder"


@dataclass(frozen=True)
class AssistantBackendConfig:
    """The settings as given: ``None`` means "not given", resolved by :func:`resolve_backend`."""

    provider: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None


@dataclass(frozen=True)
class ResolvedBackend:
    provider: str
    model: Optional[str]  # None: the provider's default model (anthropic only)
    base_url: Optional[str]  # None: the provider's default URL
    api_key: Optional[str]  # None: the SDK finds it itself (anthropic only)

    def describe(self) -> str:
        """For the panel, without the key."""
        model = self.model or "default model"
        return f"{self.provider}: {model}" + (f" at {self.base_url}" if self.base_url else "")


def _given(value: Optional[str]) -> Optional[str]:
    return value.strip() if value and value.strip() else None


def resolve_backend() -> ResolvedBackend:
    """Flag before environment before default; ValueError for an unknown provider or a missing model."""
    provider = (_given(os.environ.get(ENV_PROVIDER) or DEFAULT_PROVIDER)).lower()
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown assistant provider {provider!r}; choose one of {', '.join(PROVIDERS)}")
    model = _given(os.environ.get(ENV_MODEL))
    base_url = _given(os.environ.get(ENV_BASE_URL))
    api_key = _given(os.environ.get(ENV_API_KEY)) or _given(os.environ.get(_PROVIDER_API_KEY_ENV[provider]))

    if provider == PROVIDER_OPENAI:
        if model is None:
            raise ValueError(
                f"The provider {PROVIDER_OPENAI!r} needs a model: {ENV_MODEL} "
                "(e.g. Qwen/Qwen3.8-27B-FP8, as listed by <base URL>/models)"
            )
        return ResolvedBackend(provider, model, base_url or OPENAI_DEFAULT_BASE_URL, api_key or PLACEHOLDER_API_KEY)
    # anthropic: without a key the SDK looks for its own credentials (ANTHROPIC_API_KEY, an `ant auth login` profile)
    return ResolvedBackend(provider, model, base_url, api_key)


def make_session(backend: ResolvedBackend) -> ModelSession:
    """A new conversation with the model of ``backend``; ImportError when its SDK is not installed."""
    if backend.provider == PROVIDER_OPENAI:
        from discopop_library.ProjectManager.services.assistant_openai import OpenAICompatibleSession

        assert backend.model is not None and backend.base_url is not None  # resolve_backend guarantees them
        return OpenAICompatibleSession(backend.base_url, backend.model, backend.api_key)

    from discopop_library.ProjectManager.services.assistant_anthropic import AnthropicSession

    return AnthropicSession(model=backend.model, base_url=backend.base_url, api_key=backend.api_key)
