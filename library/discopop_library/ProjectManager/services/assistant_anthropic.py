# This file is part of the DiscoPoP software (http://www.discopop.tu-darmstadt.de)
#
# Copyright (c) 2020, Technische Universitaet Darmstadt, Germany
#
# This software may be modified and distributed under the terms of
# the 3-Clause BSD License.  See the LICENSE file in the package base
# directory for details.

"""PROTOTYPE: :class:`ModelSession` for Claude, through the Anthropic Python SDK.

The SDK is optional (``pip install anthropic``) and imported only when a session is created; credentials come from
the environment (``ANTHROPIC_API_KEY`` or an ``ant auth login`` profile).
"""

import importlib
from typing import Any, Dict, List, Optional, Sequence

from discopop_library.ProjectManager.services.assistant import ModelTurn, ToolCall, ToolResult, ToolSpec, ModelSession

MODEL = "claude-opus-5-5"
MAX_TOKENS = 64000
# agentic multi-step tool use; Claude Opus 5.5 defaults to "medium"
EFFORT = "high"
# on a refusal, the API re-runs the request on a fallback model chosen by the refusal category
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicSession(ModelSession):
    def __init__(
        self, model: Optional[str] = None, base_url: Optional[str] = None, api_key: Optional[str] = None
    ) -> None:
        anthropic = importlib.import_module("anthropic")  # ImportError: the SDK is not installed
        # what is not given, the SDK takes from the environment (ANTHROPIC_BASE_URL, ANTHROPIC_API_KEY, a profile)
        options: Dict[str, Any] = {}
        if base_url is not None:
            options["base_url"] = base_url
        if api_key is not None:
            options["api_key"] = api_key
        self._client: Any = anthropic.Anthropic(**options)
        self._model = model or MODEL
        self._messages: List[Dict[str, Any]] = []

    def add_user_message(self, text: str) -> None:
        self._messages.append({"role": "user", "content": text})

    def next_turn(self, system: str, tools: Sequence[ToolSpec]) -> ModelTurn:
        with self._client.beta.messages.stream(
            model=self._model,
            max_tokens=MAX_TOKENS,
            betas=[FALLBACK_BETA],
            fallbacks="default",
            output_config={"effort": EFFORT},
            system=system,
            tools=[_tool_definition(tool) for tool in tools],
            messages=self._messages,
        ) as stream:
            response = stream.get_final_message()
        # the whole content goes back on the next request, thinking and fallback blocks included
        self._messages.append({"role": "assistant", "content": response.content})
        text = "\n".join(block.text for block in response.content if block.type == "text")
        if response.stop_reason == "refusal":
            return ModelTurn(text + "\n[the model declined to answer]", [], "refusal")
        if response.stop_reason == "max_tokens":
            # a tool call cut off at the limit may be incomplete: run none
            return ModelTurn(text + "\n[the answer was cut off]", [], "max_tokens")
        calls = [
            ToolCall(block.id, block.name, block.input if isinstance(block.input, dict) else {})
            for block in response.content
            if block.type == "tool_use"
        ]
        return ModelTurn(text, calls, str(response.stop_reason))

    def add_tool_results(self, results: Sequence[ToolResult]) -> None:
        self._messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": r.call_id, "content": r.text, "is_error": r.is_error}
                    for r in results
                ],
            }
        )


def _tool_definition(tool: ToolSpec) -> Dict[str, Any]:
    # eager_input_streaming: inputs are not validated by the API then -- Assistant validates every call against the
    # schema before it runs (mcp_server.argument_coercion), and the local tools fail on a missing argument
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
        "eager_input_streaming": True,
    }
