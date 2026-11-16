# This file is part of the DiscoPoP software (http://www.discopop.tu-darmstadt.de)
#
# Copyright (c) 2020, Technische Universitaet Darmstadt, Germany
#
# This software may be modified and distributed under the terms of
# the 3-Clause BSD License.  See the LICENSE file in the package base
# directory for details.

import queue
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence

import pytest

from discopop_library.ProjectManager.services.assistant import (
    Assistant,
    EditProposal,
    ModelTurn,
    ToolCall,
    ToolResult,
    ToolSpec,
    apply_proposal,
)
from discopop_library.ProjectManager.services.jobs import WorkContext, WorkEvent, WorkOutput
from discopop_library.ProjectManager.services.project_state import HotspotsChanged, PatternsChanged


class ScriptedModel:
    """Answers with the given turns in order and records what it was sent."""

    def __init__(self, *turns: ModelTurn) -> None:
        self.turns = list(turns)
        self.user_messages: List[str] = []
        self.tool_results: List[List[ToolResult]] = []
        self.tools_seen: List[ToolSpec] = []

    def add_user_message(self, text: str) -> None:
        self.user_messages.append(text)

    def next_turn(self, system: str, tools: Sequence[ToolSpec]) -> ModelTurn:
        self.tools_seen = list(tools)
        return self.turns.pop(0)

    def add_tool_results(self, results: Sequence[ToolResult]) -> None:
        self.tool_results.append(list(results))


def fake_tool(name: str, answer: str, calls: List[Dict[str, Any]]) -> Any:
    def handle(arguments: Dict[str, Any], ctx: Any) -> Any:
        calls.append(arguments)
        return [SimpleNamespace(text=answer)]

    schema = {
        "type": "object",
        "properties": {"project_path": {"type": "string"}, "limit": {"type": "integer"}},
        "required": ["project_path"],
        "additionalProperties": False,
    }
    return SimpleNamespace(TOOL=SimpleNamespace(name=name, description=name, inputSchema=schema), handle=handle)


def work_context() -> "tuple[WorkContext, queue.Queue[WorkEvent]]":
    events: "queue.Queue[WorkEvent]" = queue.Queue()
    return WorkContext(events, threading.Event()), events


def outputs(events: "queue.Queue[WorkEvent]") -> List[str]:
    return [e.text for e in list(events.queue) if isinstance(e, WorkOutput)]


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "main.cpp").write_text("int main() {\n  for (int i = 0; i < n; i++)\n    a[i] = 0;\n}\n")
    return tmp_path


def test_discopop_tool_gets_project_path_and_coerced_arguments(project: Path) -> None:
    calls: List[Dict[str, Any]] = []
    model = ScriptedModel(
        ModelTurn("", [ToolCall("t1", "gather_data", {"limit": "3"})]),
        ModelTurn("Done.", []),
    )
    assistant = Assistant(str(project), model, [fake_tool("gather_data", "pipeline ran", calls)])
    work, events = work_context()

    result = assistant.run_turn("profile it", work)

    assert calls == [{"limit": 3, "project_path": str(project)}]
    assert model.tool_results == [[ToolResult("t1", "pipeline ran", False)]]
    assert "project_path" not in next(t for t in model.tools_seen if t.name == "gather_data").input_schema["properties"]
    assert result.changes == [HotspotsChanged, PatternsChanged]
    assert outputs(events) == ["▶ gather_data(limit='3')", "Done."]


def test_error_of_a_tool_is_passed_to_the_model(project: Path) -> None:
    model = ScriptedModel(
        ModelTurn("", [ToolCall("t1", "get_hotspots", {"limit": "many"}), ToolCall("t2", "nope", {})]),
        ModelTurn("Sorry.", []),
    )
    assistant = Assistant(str(project), model, [fake_tool("get_hotspots", "", [])])

    assistant.run_turn("hotspots?", work_context()[0])

    assert [r.is_error for r in model.tool_results[0]] == [True, True]


def test_read_source_and_propose_edit(project: Path) -> None:
    old = "  for (int i = 0; i < n; i++)\n"
    model = ScriptedModel(
        ModelTurn("", [ToolCall("t1", "read_source", {"file_path": "main.cpp", "start_line": 2, "end_line": 3})]),
        ModelTurn(
            "",
            [
                ToolCall(
                    "t2",
                    "propose_edit",
                    {
                        "file_path": "main.cpp",
                        "old_text": old,
                        "new_text": "#pragma omp parallel for\n" + old,
                        "reason": "doall loop",
                    },
                )
            ],
        ),
        ModelTurn("See the proposal.", []),
    )
    assistant = Assistant(str(project), model, [])

    result = assistant.run_turn("parallelize", work_context()[0])

    assert (
        model.tool_results[0][0].text
        == "    2    for (int i = 0; i < n; i++)\n    3      a[i] = 0;\n(the file has 4 lines)"
    )
    assert [p.reason for p in result.proposals] == ["doall loop"]
    assert "#pragma" not in (project / "main.cpp").read_text(), "a proposal does not change the file"

    apply_proposal(str(project), result.proposals[0])
    assert "#pragma omp parallel for\n  for" in (project / "main.cpp").read_text()
    (project / "main.cpp").write_text("int main() {}\n")  # changed since the proposal
    with pytest.raises(ValueError):
        apply_proposal(str(project), result.proposals[0])


def test_files_outside_of_the_project_are_refused(project: Path) -> None:
    model = ScriptedModel(ModelTurn("", [ToolCall("t1", "read_source", {"file_path": "../secret"})]), ModelTurn("", []))
    Assistant(str(project), model, []).run_turn("x", work_context()[0])
    assert model.tool_results[0][0].is_error
    with pytest.raises(ValueError):
        apply_proposal(str(project), EditProposal(1, "/etc/passwd", "a", "b", ""))


def test_cancel_stops_before_the_next_model_call(project: Path) -> None:
    model = ScriptedModel(ModelTurn("never sent", []))
    work, events = work_context()
    work.cancel_event.set()

    Assistant(str(project), model, []).run_turn("x", work)

    assert model.turns, "the model was not called"
    assert outputs(events) == ["[stopped]"]
