import re
from typing import List, Dict, Any, Sequence, Tuple

import json

from discopop_library.ProjectManager.services.assistant import ToolCall, ToolSpec
from discopop_library.ProjectManager.services.assistant_openai import OpenAIResponse


# reasoning models served without a reasoning parser (e.g. Qwen 3 on a plain vLLM) put their thinking into the text
_THINK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_ERROR_EXCERPT = 500


def to_tool_calls(raw_openai_calls: List[Dict[str, any]]) -> List[ToolCall]:
    def _to_tool_call(raw: Dict[str, Any], index: int) -> ToolCall:
        function = raw.get("function") or {}
        call_id = str(raw.get("id") or f"call_{index}")
        name = str(function.get("name") or "")
        arguments_text = function.get("arguments") or "{}"
        try:
            arguments = json.loads(arguments_text) if isinstance(arguments_text, str) else arguments_text
        except ValueError as e:
            return ToolCall(call_id, name, {}, error=f"The arguments are not valid JSON ({e}): {arguments_text!r}")
        if not isinstance(arguments, dict):
            return ToolCall(call_id, name, {}, error=f"The arguments have to be a JSON object, not {arguments_text!r}")
        return ToolCall(call_id, name, arguments)

    return [_to_tool_call(raw, index) for index, raw in enumerate(raw_openai_calls)]


def to_openai_tool_defs(tools: Sequence[ToolSpec]) -> List[Dict[str, Any]]:
    def _to_tool_definition(tool: ToolSpec) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": tool.name, "description": tool.description, "parameters": tool.input_schema},
        }

    return [_to_tool_definition(tool) for tool in tools]

def to_response(openai_answer: Dict[str, any]) -> OpenAIResponse:
    choices = openai_answer.get("choices") or []
    if not choices:
        raise RuntimeError(f"The model server sent no answer: {json.dumps(openai_answer)[:_ERROR_EXCERPT]}")
    choice = choices[0]
    message = choice.get("message") or {}
    finish = str(choice.get("finish_reason") or "")
    text = _THINK.sub("", message.get("content") or "").strip()  # remove reasoning parts
    raw_calls: list = message.get("tool_calls") or []
    return OpenAIResponse(text, raw_calls, finish)