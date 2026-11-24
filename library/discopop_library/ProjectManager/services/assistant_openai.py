# This file is part of the DiscoPoP software (http://www.discopop.tu-darmstadt.de)
#
# Copyright (c) 2020, Technische Universitaet Darmstadt, Germany
#
# This software may be modified and distributed under the terms of
# the 3-Clause BSD License.  See the LICENSE file in the package base
# directory for details.

"""PROTOTYPE: :class:`ModelSession` for any server speaking the OpenAI Chat Completions API with tool calling.

That is OpenAI itself and the servers imitating it: vLLM (``--enable-auto-tool-choice`` and a ``--tool-call-parser``
are needed for tools), Ollama, LM Studio, llama.cpp's server, OpenRouter, ... -- the same kind of provider opencode
calls ``@ai-sdk/openai-compatible``. One ``POST <base_url>/chat/completions`` per turn, with the standard library
only: no SDK to install.
"""

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from discopop_library.ProjectManager.services import openai_adapter
from discopop_library.ProjectManager.services.assistant import ModelTurn, ToolCall, ToolResult, ToolSpec, ModelSession
from discopop_library.ProjectManager.services.openai_adapter import OpenAIResponse

# a model calling tools on a large project can take minutes for one answer on a small GPU
TIMEOUT_SECONDS = 600.0
# reasoning models served without a reasoning parser (e.g. Qwen 3 on a plain vLLM) put their thinking into the text
_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_ERROR_EXCERPT = 500

# (url, headers, body) -> the decoded JSON answer; replaced by a fake in the tests
Transport = Callable[[str, Dict[str, str], Dict[str, Any]], Dict[str, Any]]


class OpenAICompatibleSession(ModelSession):
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str],
        max_tokens: Optional[int] = None,
        transport: Optional[Transport] = None,
    ) -> None:
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._model = model
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._transport: Transport = transport or _post_json
        self._messages: List[Dict[str, Any]] = []

    def add_user_message(self, text: str) -> None:
        self._messages.append({"role": "user", "content": text})

    def next_turn(self, system: str, tools: Sequence[ToolSpec]) -> ModelTurn:
        body: Dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "system", "content": system}] + self._messages,
        }
        if tools:
            body["tools"] = openai_adapter.to_openai_tool_defs(tools)
            body["tool_choice"] = "auto"
        if self._max_tokens is not None:
            body["max_tokens"] = self._max_tokens
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        answer: Dict[str, any] = self._transport(self._url, headers, body)
        parsed_response: OpenAIResponse = openai_adapter.to_response(answer)

        if parsed_response.finish_reason == "length": # token limit reached
            # a tool call cut off at the limit may be incomplete: run none, and keep none in the history (every
            # tool call kept there needs a result)
            self._messages.append({"role": "assistant", "content": parsed_response.text})
            return ModelTurn(
                parsed_response.text + "\n[the answer was cut off]",
                [],
                parsed_response.finish_reason
            )
        if parsed_response.finish_reason == "content_filter": # server's filter blocked answer
            self._messages.append({"role": "assistant", "content": parsed_response.text})
            return ModelTurn(
                parsed_response.text + "\n[the model declined to answer]",
                [],
                parsed_response.finish_reason
            )

        tool_calls: list[ToolCall] = openai_adapter.to_tool_calls(parsed_response.raw_tool_calls)
        # the history gets the calls as the server sent them, with the ids given to calls without one
        assistant: Dict[str, Any] = {"role": "assistant", "content": parsed_response.text or None}
        if tool_calls:
            assistant["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": (raw.get("function") or {}).get("arguments") or "{}"},
                }
                for call, raw in zip(tool_calls, parsed_response.raw_tool_calls)
            ]
        self._messages.append(assistant)
        return ModelTurn(parsed_response.text, tool_calls, parsed_response.finish_reason)

    def add_tool_results(self, results: Sequence[ToolResult]) -> None:
        # the API has no error flag for a tool result: the model reads it from the text
        for result in results:
            content = ("ERROR: " + result.text) if result.is_error else result.text
            self._messages.append({"role": "tool", "tool_call_id": result.call_id, "content": content})

def _tool_definition(tool: ToolSpec) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": tool.name, "description": tool.description, "parameters": tool.input_schema},
    }


def _post_json(url: str, headers: Dict[str, str], body: Dict[str, Any]) -> Dict[str, Any]:
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            answer = json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:_ERROR_EXCERPT]
        hint = ""
        if e.code in (401, 403):
            hint = " Check the API key (environment variable DISCOPOP_ASSISTANT_API_KEY)."
        elif e.code == 404:
            hint = " Check the base URL (it usually ends in /v1) and the model name."
        raise RuntimeError(f"The model server at {url} answered {e.code} {e.reason}.{hint} {detail}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Cannot reach the model server at {url}: {e.reason}") from e
    if not isinstance(answer, dict):
        raise RuntimeError(f"The model server at {url} sent no JSON object")
    return answer
