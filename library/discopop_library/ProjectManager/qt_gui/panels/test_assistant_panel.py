# This file is part of the DiscoPoP software (http://www.discopop.tu-darmstadt.de)
#
# Copyright (c) 2020, Technische Universitaet Darmstadt, Germany
#
# This software may be modified and distributed under the terms of
# the 3-Clause BSD License.  See the LICENSE file in the package base
# directory for details.

from pathlib import Path
import threading
from typing import Callable, List, Sequence

import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from pytestqt.qtbot import QtBot

from discopop_library.ProjectManager.qt_gui.conftest import DialogRecorder
from discopop_library.ProjectManager.qt_gui.panels.assistant_panel import AssistantPanel, model_factory_from_arguments
from discopop_library.ProjectManager.qt_gui.status import StatusLine
from discopop_library.ProjectManager.qt_gui.testing import make_arguments
from discopop_library.ProjectManager.services.assistant import ModelTurn, ToolCall, ToolModule, ToolResult, ToolSpec
from discopop_library.ProjectManager.services.assistant_openai import OpenAICompatibleSession
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


def make_panel(
    qtbot: QtBot, tmp_path: Path, tool_loader: Callable[[], List[ToolModule]] = lambda: []
) -> AssistantPanel:
    """A panel whose assistant has been created (in the background): Send is enabled then."""
    (tmp_path / "main.cpp").write_text("int main() {\n" + OLD + "    a[i] = 0;\n}\n")
    status = StatusLine()
    qtbot.addWidget(status)
    panel = AssistantPanel(
        make_arguments(tmp_path), ProjectState(), status, model_factory=ProposingModel, tool_loader=tool_loader
    )
    qtbot.addWidget(panel)
    qtbot.waitUntil(panel.send_button.isEnabled, timeout=5000)
    return panel


def test_send_is_disabled_until_the_assistant_exists(qtbot: QtBot, tmp_path: Path) -> None:
    status = StatusLine()
    qtbot.addWidget(status)
    created = threading.Event()

    def slow_model() -> ProposingModel:
        created.wait(5)
        return ProposingModel()

    panel = AssistantPanel(
        make_arguments(tmp_path), ProjectState(), status, model_factory=slow_model, tool_loader=lambda: []
    )
    qtbot.addWidget(panel)
    assert not panel.send_button.isEnabled()
    created.set()
    qtbot.waitUntil(panel.send_button.isEnabled, timeout=5000)


def test_deleting_the_panel_while_the_assistant_is_created_is_harmless(qtbot: QtBot, tmp_path: Path) -> None:
    # was a QThread owned by the panel: destroying it while running aborted the process
    status = StatusLine()
    qtbot.addWidget(status)
    created = threading.Event()

    def slow_model() -> ProposingModel:
        created.wait(5)
        return ProposingModel()

    panel = AssistantPanel(
        make_arguments(tmp_path), ProjectState(), status, model_factory=slow_model, tool_loader=lambda: []
    )
    panel.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    created.set()
    qtbot.wait(200)  # the creation finishes; nobody polls its result any more


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
    def broken() -> List[ToolModule]:
        raise ImportError("No module named 'mcp_server'")

    panel = make_panel(qtbot, tmp_path, tool_loader=broken)
    panel.input.setPlainText("hi")
    panel.send()

    assert dialog_answers.titles("show_error") == ["Assistant"]
    assert "No module named 'mcp_server'" in dialog_answers.messages("show_error")[0]
    assert panel.send_button.isEnabled()


def test_model_factory_follows_the_assistant_arguments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("DISCOPOP_ASSISTANT_PROVIDER", "DISCOPOP_ASSISTANT_MODEL", "DISCOPOP_ASSISTANT_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    arguments = make_arguments(tmp_path)
    arguments.assistant_provider = "openai"
    arguments.assistant_model = "Qwen/Qwen3.8-27B-FP8"
    arguments.assistant_base_url = "http://localhost:18000/v1"
    assert isinstance(model_factory_from_arguments(arguments)(), OpenAICompatibleSession)

    arguments.assistant_model = None
    with pytest.raises(ValueError, match="needs a model"):
        model_factory_from_arguments(arguments)()
