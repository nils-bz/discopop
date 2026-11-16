# This file is part of the DiscoPoP software (http://www.discopop.tu-darmstadt.de)
#
# Copyright (c) 2020, Technische Universitaet Darmstadt, Germany
#
# This software may be modified and distributed under the terms of
# the 3-Clause BSD License.  See the LICENSE file in the package base
# directory for details.

from pathlib import Path
from typing import List, Sequence

from pytestqt.qtbot import QtBot

from discopop_library.ProjectManager.qt_gui.conftest import DialogRecorder
from discopop_library.ProjectManager.qt_gui.panels.assistant_panel import AssistantPanel
from discopop_library.ProjectManager.qt_gui.status import StatusLine
from discopop_library.ProjectManager.qt_gui.testing import make_arguments
from discopop_library.ProjectManager.services.assistant import ModelTurn, ToolCall, ToolResult, ToolSpec
from discopop_library.ProjectManager.services.project_state import ProjectState

OLD = "  for (int i = 0; i < n; i++)\n"


class ProposingModel:
    """Proposes one pragma, then answers."""

    def __init__(self) -> None:
        self.turns = [
            ModelTurn(
                "Let me propose a change.",
                [
                    ToolCall(
                        "t1",
                        "propose_edit",
                        {
                            "file_path": "main.cpp",
                            "old_text": OLD,
                            "new_text": "#pragma omp parallel for\n" + OLD,
                            "reason": "doall loop",
                        },
                    )
                ],
            ),
            ModelTurn("Review proposal 1.", []),
        ]
        self.results: List[ToolResult] = []

    def add_user_message(self, text: str) -> None:
        pass

    def next_turn(self, system: str, tools: Sequence[ToolSpec]) -> ModelTurn:
        return self.turns.pop(0)

    def add_tool_results(self, results: Sequence[ToolResult]) -> None:
        self.results.extend(results)


def make_panel(qtbot: QtBot, tmp_path: Path) -> AssistantPanel:
    (tmp_path / "main.cpp").write_text("int main() {\n" + OLD + "    a[i] = 0;\n}\n")
    status = StatusLine()
    qtbot.addWidget(status)
    panel = AssistantPanel(
        make_arguments(tmp_path), ProjectState(), status, model_factory=ProposingModel, tool_loader=lambda: []
    )
    qtbot.addWidget(panel)
    return panel


def test_turn_shows_transcript_and_proposal_and_applies_it(qtbot: QtBot, tmp_path: Path) -> None:
    panel = make_panel(qtbot, tmp_path)
    panel.input.setPlainText("Parallelize the loop")
    panel.send()
    assert not panel.send_button.isEnabled()

    qtbot.waitUntil(lambda: panel.send_button.isEnabled(), timeout=5000)

    transcript = panel.transcript.toPlainText()
    assert "Parallelize the loop" in transcript and "propose_edit" in transcript and "Review proposal 1." in transcript
    assert panel.proposal_list.count() == 1
    assert "+#pragma omp parallel for" in panel.diff_view.toPlainText()
    assert "#pragma" not in (tmp_path / "main.cpp").read_text()

    panel.apply_selected()

    assert "#pragma omp parallel for\n" + OLD in (tmp_path / "main.cpp").read_text()
    assert panel.proposal_list.count() == 0 and not panel.apply_button.isEnabled()


def test_missing_backend_is_reported(qtbot: QtBot, tmp_path: Path, dialog_answers: DialogRecorder) -> None:
    panel = make_panel(qtbot, tmp_path)

    def broken() -> List[object]:
        raise ImportError("No module named 'mcp_server'")

    panel._tool_loader = broken  # type: ignore[assignment]
    panel.input.setPlainText("hi")
    panel.send()

    assert dialog_answers.titles("show_error") == ["Assistant"]
    assert panel.send_button.isEnabled()
