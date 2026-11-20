# This file is part of the DiscoPoP software (http://www.discopop.tu-darmstadt.de)
#
# Copyright (c) 2020, Technische Universitaet Darmstadt, Germany
#
# This software may be modified and distributed under the terms of
# the 3-Clause BSD License.  See the LICENSE file in the package base
# directory for details.

import json
import queue
import threading
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from discopop_library.ProjectManager.services.assistant import Assistant, ToolResult, ToolSpec
from discopop_library.ProjectManager.services.assistant_openai import OpenAICompatibleSession
from discopop_library.ProjectManager.services.jobs import WorkContext, WorkEvent

TOOL = ToolSpec("read_source", "Read a file.", {"type": "object", "properties": {"file_path": {"type": "string"}}})


class FakeServer:
    """Answers the requests with the queued answers and records them (copies: the session reuses its lists)."""

    def __init__(self, *answers: Dict[str, Any]) -> None:
        self.answers = list(answers)
        self.requests: List[Tuple[str, Dict[str, str], Dict[str, Any]]] = []

    def __call__(self, url: str, headers: Dict[str, str], body: Dict[str, Any]) -> Dict[str, Any]:
        self.requests.append((url, dict(headers), json.loads(json.dumps(body))))
        return self.answers.pop(0)


def answer(content: Any = None, tool_calls: Any = None, finish: str = "stop") -> Dict[str, Any]:
    message: Dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {"choices": [{"index": 0, "message": message, "finish_reason": finish}]}


def call(call_id: str, name: str, arguments: str) -> Dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def test_request_has_url_key_system_prompt_and_tools() -> None:
    server = FakeServer(answer("Hello"))
    session = OpenAICompatibleSession("http://localhost:18000/v1/", "Qwen/Qwen3.8-27B-FP8", "secret", transport=server)
    session.add_user_message("Hi")

    turn = session.next_turn("You are the assistant.", [TOOL])

    assert turn.text == "Hello" and turn.tool_calls == []
    url, headers, body = server.requests[0]
    assert url == "http://localhost:18000/v1/chat/completions"
    assert headers["Authorization"] == "Bearer secret"
    assert body["model"] == "Qwen/Qwen3.8-27B-FP8"
    assert body["messages"] == [
        {"role": "system", "content": "You are the assistant."},
        {"role": "user", "content": "Hi"},
    ]
    assert body["tools"] == [
        {
            "type": "function",
            "function": {"name": "read_source", "description": "Read a file.", "parameters": TOOL.input_schema},
        }
    ]


def test_without_key_no_authorization_header() -> None:
    server = FakeServer(answer("ok"))
    session = OpenAICompatibleSession("http://h/v1", "m", None, transport=server)
    session.add_user_message("Hi")
    session.next_turn("s", [])
    assert "Authorization" not in server.requests[0][1]
    assert "tools" not in server.requests[0][2]


def test_tool_calls_and_results_go_into_the_history() -> None:
    server = FakeServer(
        answer(None, [call("c1", "read_source", '{"file_path": "main.cpp"}')], finish="tool_calls"),
        answer("Done."),
    )
    session = OpenAICompatibleSession("http://h/v1", "m", "k", transport=server)
    session.add_user_message("Read main.cpp")

    turn = session.next_turn("s", [TOOL])
    assert [(c.id, c.name, c.arguments, c.error) for c in turn.tool_calls] == [
        ("c1", "read_source", {"file_path": "main.cpp"}, None)
    ]
    session.add_tool_results([ToolResult("c1", "int main() {}"), ToolResult("c1", "boom", is_error=True)])
    session.next_turn("s", [TOOL])

    messages = server.requests[1][2]["messages"]
    assert messages[2] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [call("c1", "read_source", '{"file_path": "main.cpp"}')],
    }
    assert messages[3] == {"role": "tool", "tool_call_id": "c1", "content": "int main() {}"}
    assert messages[4] == {"role": "tool", "tool_call_id": "c1", "content": "ERROR: boom"}


def test_thinking_is_removed_from_the_text() -> None:
    server = FakeServer(answer("<think>\nlet me see\n</think>\n\nThe loop is a doall."))
    session = OpenAICompatibleSession("http://h/v1", "m", "k", transport=server)
    session.add_user_message("?")
    assert session.next_turn("s", []).text == "The loop is a doall."


@pytest.mark.parametrize("arguments", ['{"file_path": ', '["main.cpp"]'])
def test_unreadable_arguments_become_an_error_call(arguments: str) -> None:
    server = FakeServer(answer(None, [call("c1", "read_source", arguments)], finish="tool_calls"))
    session = OpenAICompatibleSession("http://h/v1", "m", "k", transport=server)
    session.add_user_message("?")
    (tool_call,) = session.next_turn("s", [TOOL]).tool_calls
    assert tool_call.error is not None and tool_call.arguments == {}


def test_cut_off_answer_runs_no_tool_and_keeps_none_in_the_history() -> None:
    server = FakeServer(answer("partial", [call("c1", "read_source", '{"file_')], finish="length"), answer("ok"))
    session = OpenAICompatibleSession("http://h/v1", "m", "k", transport=server)
    session.add_user_message("?")

    turn = session.next_turn("s", [TOOL])
    assert turn.tool_calls == [] and "cut off" in turn.text
    session.add_user_message("go on")
    session.next_turn("s", [TOOL])
    assert server.requests[1][2]["messages"][2] == {"role": "assistant", "content": "partial"}


def test_answer_without_choices_is_an_error() -> None:
    session = OpenAICompatibleSession("http://h/v1", "m", "k", transport=FakeServer({"error": "overloaded"}))
    session.add_user_message("?")
    with pytest.raises(RuntimeError, match="overloaded"):
        session.next_turn("s", [])


def test_assistant_does_not_run_a_call_with_unreadable_arguments(tmp_path: Path) -> None:
    (tmp_path / "main.cpp").write_text("int main() {}\n")
    server = FakeServer(
        answer(None, [call("c1", "propose_edit", '{"file_path": "main.cpp", "old_text": ')], finish="tool_calls"),
        answer("Sorry."),
    )
    session = OpenAICompatibleSession("http://h/v1", "m", "k", transport=server)
    assistant = Assistant(str(tmp_path), session, [])
    events: "queue.Queue[WorkEvent]" = queue.Queue()
    work = WorkContext(events, threading.Event())

    result = assistant.run_turn("Change it", work)

    assert result.proposals == []
    tool_message = server.requests[1][2]["messages"][-1]
    assert tool_message["role"] == "tool" and tool_message["content"].startswith("ERROR: The arguments are not valid")
