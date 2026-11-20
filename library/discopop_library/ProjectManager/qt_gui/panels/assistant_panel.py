# This file is part of the DiscoPoP software (http://www.discopop.tu-darmstadt.de)
#
# Copyright (c) 2020, Technische Universitaet Darmstadt, Germany
#
# This software may be modified and distributed under the terms of
# the 3-Clause BSD License.  See the LICENSE file in the package base
# directory for details.

"""PROTOTYPE: the Assistant tab -- a chat with an LLM that drives DiscoPoP (``services/assistant.py``).

A turn runs on a ``WorkRunner``; its text and tool calls arrive as ``WorkOutput`` lines. The edits the assistant
proposes are listed below the chat and only change a file when the user applies them.
"""

import difflib
from typing import Callable, Dict, List, Optional, Sequence

from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
    QListWidgetItem,
)
from PySide6.QtCore import Qt

from discopop_library.ProjectManager.ProjectManagerArguments import ProjectManagerArguments
from discopop_library.ProjectManager.qt_gui import dialogs, widgets
from discopop_library.ProjectManager.qt_gui.job_adapter import start_work
from discopop_library.ProjectManager.qt_gui.panel import Panel
from discopop_library.ProjectManager.qt_gui.status import StatusLine
from discopop_library.ProjectManager.services.assistant import (
    Assistant,
    EditProposal,
    ModelSession,
    ToolModule,
    TurnResult,
    apply_proposal,
    load_mcp_tools,
)
from discopop_library.ProjectManager.services.assistant_backends import (
    AssistantBackendConfig,
    make_session,
    resolve_backend,
)
from discopop_library.ProjectManager.services.jobs import (
    WorkContext,
    WorkEvent,
    WorkFinished,
    WorkOutput,
    WorkRunner,
    WorkStatus,
)
from discopop_library.ProjectManager.services.project_lock import ProjectLock
from discopop_library.ProjectManager.services.project_state import ProjectState


def model_factory_from_arguments(arguments: ProjectManagerArguments) -> Callable[[], ModelSession]:
    """The backend chosen by the ``--assistant-*`` flags and the environment (``services/assistant_backends.py``).

    Resolved when a session is created (on the initializer's thread), so a wrong setting is reported in the panel.
    """
    config = AssistantBackendConfig(
        provider=arguments.assistant_provider,
        model=arguments.assistant_model,
        base_url=arguments.assistant_base_url,
    )
    return lambda: make_session(resolve_backend(config))


class AssistantPanel(Panel):
    def __init__(
        self,
        arguments: ProjectManagerArguments,
        project_state: ProjectState,
        status: StatusLine,
        model_factory: Optional[Callable[[], ModelSession]] = None,
        tool_loader: Callable[[], List[ToolModule]] = load_mcp_tools,
        parent: Optional[QWidget] = None,
        project_lock: Optional[ProjectLock] = None,
    ) -> None:
        super().__init__(arguments, project_state, status, parent, project_lock=project_lock)
        self._model_factory: Callable[[], ModelSession] = model_factory or model_factory_from_arguments(arguments)
        self._tool_loader = tool_loader
        self._assistant: Optional[Assistant] = None
        self._runner: Optional[WorkRunner] = None
        self._proposals: Dict[int, EditProposal] = {}

        # the assistant is created in the background (importing the SDK and the MCP tools takes a moment); until
        # then, Send is disabled
        self._init_runner: Optional[WorkRunner] = None
        self._assistant_init_error: Optional[str] = None

        self.transcript = QTextBrowser()
        self.input = QPlainTextEdit()
        self.input.setPlaceholderText("e.g. Where does the program spend its time, and what can be parallelized?")
        self.input.setMaximumHeight(90)
        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send)
        self.send_button.setEnabled(False)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.cancel_jobs)

        self.proposal_list = QListWidget()
        self.proposal_list.currentRowChanged.connect(lambda _row: self._show_diff())
        self.diff_view = QPlainTextEdit()
        self.diff_view.setReadOnly(True)
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_selected)
        self.reject_button = QPushButton("Reject")
        self.reject_button.clicked.connect(self.reject_selected)

        chat = QWidget()
        chat_layout = QVBoxLayout(chat)
        chat_layout.addWidget(self.transcript)
        row = QHBoxLayout()
        row.addWidget(self.input, 1)
        row.addWidget(self.send_button)
        row.addWidget(self.stop_button)
        chat_layout.addLayout(row)

        review = QWidget()
        review_layout = QVBoxLayout(review)
        review_layout.addWidget(QLabel("Proposed changes (not applied until you apply them):"))
        lists = QHBoxLayout()
        lists.addWidget(self.proposal_list, 1)
        lists.addWidget(self.diff_view, 3)
        review_layout.addLayout(lists)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.reject_button)
        buttons.addWidget(self.apply_button)
        review_layout.addLayout(buttons)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(chat)
        splitter.addWidget(review)
        splitter.setSizes([500, 220])
        layout = QVBoxLayout(self)
        layout.addWidget(splitter)
        self._update_buttons()
        # on a plain worker thread, not a QThread owned by the panel: deleting the panel while the work still runs
        # would destroy a running QThread, which aborts the process; the adapter's timer just dies with the panel
        self._init_runner = start_work(self, self._create_assistant, self._on_assistant_created)

    def send(self) -> None:
        text = self.input.toPlainText().strip()
        if not text or self._runner is not None:
            return
        if self._assistant is None:
            dialogs.show_error(self, "Assistant", self._build_assistant_init_error_msg())
            return
        assistant = self._assistant
        self.input.clear()
        self._append("You", text)
        self.status.set("⏳ Assistant working ...", widgets.STATUS_BUSY)
        self._runner = start_work(self, lambda work: assistant.run_turn(text, work), self._handle)
        self._update_buttons()

    def cancel_jobs(self) -> None:
        if self._runner is not None:
            self._runner.cancel()
            self.status.set("Stopping the assistant ...", widgets.STATUS_STOP)
        if self._init_runner is not None:
            self._init_runner.cancel()  # the creation does not stop early, but its result is no longer awaited

    def _handle(self, event: WorkEvent) -> None:
        if isinstance(event, WorkOutput):
            if event.text.startswith("▶ "):
                self._append("tool", event.text[2:])
            else:
                self._append("Assistant", event.text)
        elif isinstance(event, WorkStatus):
            self.status.set(event.text, widgets.STATUS_BUSY)
        elif isinstance(event, WorkFinished):
            self._runner = None
            self._update_buttons()
            if event.error is not None:
                self._append("error", event.error)
                self.status.set("Assistant failed", widgets.STATUS_FAIL, reset_ms=5000)
                return
            self.status.set("Assistant done", widgets.STATUS_OK, reset_ms=3000)
            self._finish_turn(event.result)

    def _finish_turn(self, result: TurnResult) -> None:
        for proposal in result.proposals:
            self._proposals[proposal.id] = proposal
            item = QListWidgetItem(f"{proposal.id}: {proposal.path} – {proposal.reason}")
            item.setData(Qt.ItemDataRole.UserRole, proposal.id)
            self.proposal_list.addItem(item)

        if result.proposals and self.proposal_list.currentRow() < 0:
            self.proposal_list.setCurrentRow(0)

        # the other panels show what the assistant's tool calls produced
        for event_type in result.changes:
            self.project_state.publish(event_type())

        self._update_buttons()

    def _append(self, who: str, text: str) -> None:
        style = {"You": "font-weight:bold", "tool": "color:gray", "error": "color:#b00"}.get(who, "")
        escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
        self.transcript.append(f'<p style="{style}"><b>{who}:</b> {escaped}</p>')

    def _create_assistant(self, _work: WorkContext) -> Assistant:
        # on the worker thread: touches no widget
        return Assistant(self.arguments.project_root, self._model_factory(), self._tool_loader())

    def _on_assistant_created(self, event: WorkEvent) -> None:
        if not isinstance(event, WorkFinished):
            return
        self._init_runner = None
        if event.error is not None:
            # Send stays enabled: it reports the error
            self._assistant_init_error = event.error
        else:
            self._assistant = event.result
        self._update_buttons()

    def _build_assistant_init_error_msg(self) -> str:
        if self._assistant_init_error is not None:
            return f"Could not start assistant: {self._assistant_init_error}"
        return "Could not start assistant."

    # --- the review of proposed changes ---

    def _selected(self) -> Optional[EditProposal]:
        # which edit proposal is highlighted in the review list
        item = self.proposal_list.currentItem()
        if item is None:
            return None
        proposal_id = item.data(Qt.ItemDataRole.UserRole)
        return self._proposals.get(proposal_id)

    def _show_diff(self) -> None:
        proposal = self._selected()
        if proposal is None:
            self.diff_view.clear()
        else:
            diff = difflib.unified_diff(
                proposal.old.splitlines(), proposal.new.splitlines(), proposal.path, proposal.path, lineterm=""
            )
            self.diff_view.setPlainText("\n".join(diff))
        self._update_buttons()

    def apply_selected(self) -> None:
        proposal = self._selected()
        if proposal is None:
            return
        try:
            apply_proposal(self.arguments.project_root, proposal)
        except (OSError, ValueError) as e:
            dialogs.show_error(self, "Apply change", str(e))
            return
        # the model learns about it with the next message; the results are stale now (get_project_status says so)
        self._append("You", f"[applied proposal {proposal.id} to {proposal.path}]")
        self._remove_selected()

    def reject_selected(self) -> None:
        proposal = self._selected()
        if proposal is not None:
            self._append("You", f"[rejected proposal {proposal.id}]")
            self._remove_selected()

    def _remove_selected(self) -> None:
        proposal = self._selected()
        if proposal is not None:
            del self._proposals[proposal.id]
        self.proposal_list.takeItem(self.proposal_list.currentRow())
        self._show_diff()

    def _update_buttons(self) -> None:
        is_running = self._runner is not None
        is_assistant_init_done = self._assistant is not None or self._assistant_init_error is not None
        self.send_button.setEnabled(not is_running and is_assistant_init_done)
        self.stop_button.setEnabled(is_running)
        # no change while a turn runs: its tools read the files
        has_selection = self._selected() is not None
        self.apply_button.setEnabled(has_selection and not is_running)
        self.reject_button.setEnabled(has_selection)
