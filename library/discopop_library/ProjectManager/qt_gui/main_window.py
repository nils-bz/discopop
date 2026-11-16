# This file is part of the DiscoPoP software (http://www.discopop.tu-darmstadt.de)
#
# Copyright (c) 2020, Technische Universitaet Darmstadt, Germany
#
# This software may be modified and distributed under the terms of
# the 3-Clause BSD License.  See the LICENSE file in the package base
# directory for details.

"""The main window: the configuration list on the left, the panels as tabs on the right, the status line below.

The window assembles the panels and wires them: which panel reacts to which :class:`ProjectState` event, and the
few callbacks one panel offers another. The panels do not know each other.
"""

import os
from typing import Dict, List, Optional, Tuple

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QCloseEvent, QIcon, QKeySequence, QResizeEvent, QShortcut
from PySide6.QtWidgets import QLabel, QMainWindow, QSplitter, QTabWidget, QWidget, QToolButton

from discopop_library.ProjectManager.ProjectManagerArguments import ProjectManagerArguments
from discopop_library.ProjectManager.qt_gui import widgets
from discopop_library.ProjectManager.qt_gui.panel import Panel
from discopop_library.ProjectManager.qt_gui.panels.autotuning_panel import AutotuningPanel
from discopop_library.ProjectManager.qt_gui.panels.config_list_panel import ConfigListPanel
from discopop_library.ProjectManager.qt_gui.panels.editor_panel import EditorPanel
from discopop_library.ProjectManager.qt_gui.panels.execute_panel import ExecutePanel
from discopop_library.ProjectManager.qt_gui.panels.hotspot_panel import HotspotPanel
from discopop_library.ProjectManager.qt_gui.panels.patch_repair_panel import PatchRepairPanel
from discopop_library.ProjectManager.qt_gui.panels.pattern_detection_panel import PatternDetectionPanel
from discopop_library.ProjectManager.qt_gui.panels.report_panel import ReportPanel
from discopop_library.ProjectManager.qt_gui.panels.assistant_panel import AssistantPanel
from discopop_library.ProjectManager.qt_gui.status import StatusLine

# re-exported: the names were defined here first
from discopop_library.ProjectManager.qt_gui.tabs import TAB_ASSISTANT
from discopop_library.ProjectManager.qt_gui.tabs import (
    TAB_AUTOTUNING as TAB_AUTOTUNING,
    TAB_EDITOR as TAB_EDITOR,
    TAB_EXECUTE as TAB_EXECUTE,
    TAB_HOTSPOT as TAB_HOTSPOT,
    TAB_ORDER as TAB_ORDER,
    TAB_PATCH_REPAIR as TAB_PATCH_REPAIR,
    TAB_PATTERN_DETECTION as TAB_PATTERN_DETECTION,
    TAB_REPORT as TAB_REPORT,
)
from discopop_library.ProjectManager.services.project_lock import LockHolder, ProjectLock
from discopop_library.ProjectManager.services.project_state import (
    AutotuningFinished,
    ConfigurationsChanged,
    ProjectBusyChanged,
    ProjectState,
    ResultsChanged,
)

ICON_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "icon.png")


class ElidedPathLabel(QLabel):
    """Shows a path, cut from the left with an ellipsis when it does not fit; the full path is the tooltip."""

    def __init__(self, path: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._path = path
        self.setToolTip(path)
        widgets.set_label_color(self, widgets.STATUS_IDLE)
        self._elide()

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 (Qt's name)
        super().resizeEvent(event)
        self._elide()

    def _elide(self) -> None:
        width = self.width() if self.width() > 1 else 10_000
        self.setText(self.fontMetrics().elidedText(self._path, Qt.TextElideMode.ElideLeft, width))


class _LockChanges(QObject):
    """Brings the changes of the :class:`ProjectLock` to the main thread.

    The lock calls its ``on_change`` on the thread that acquires or releases it -- a worker thread for the assistant's
    tool calls -- but ``ProjectState`` subscribers touch widgets. A signal emitted on another thread reaches a slot of
    this object, which lives on the main thread, as a queued call.
    """

    changed = Signal(object)


class MainWindow(QMainWindow):
    """The window; it is the :class:`TabHost` of the panels."""

    def __init__(self, arguments: ProjectManagerArguments, was_initialized: bool = False) -> None:
        super().__init__()
        self.setWindowTitle("DiscoPoP Project Manager")
        self.setWindowIcon(QIcon(ICON_PATH))
        self.resize(1900, 1000)
        self.setMinimumSize(1100, 700)

        self.arguments = arguments
        self.was_initialized = was_initialized
        self.project_state = ProjectState()
        self._lock_changes = _LockChanges(self)
        self._lock_changes.changed.connect(self._publish_project_busy)
        self.project_lock = ProjectLock(self._lock_changes.changed.emit)
        self.panels: List[Panel] = []

        self.status = StatusLine()
        self.dot_dp_label = ElidedPathLabel(arguments.dot_dp)
        self.statusBar().addWidget(self.status, 1)
        self.statusBar().addPermanentWidget(self.dot_dp_label)

        self._assistant_width = 300
        self.right_toggle = QToolButton()
        self.right_toggle.setText("Assistant")
        self.right_toggle.setCheckable(True)
        self.right_toggle.setChecked(True)
        self.right_toggle.clicked.connect(self.toggle_right_panel)
        self.statusBar().addPermanentWidget(self.right_toggle)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(self.splitter)
        self.tabs = QTabWidget()
        self._tab_widgets: Dict[str, QWidget] = {}
        self._pending_tab_state: Dict[str, Tuple[bool, str]] = {}
        self._pending_tab_titles: Dict[str, str] = {}

        self._build_panels()
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setStretchFactor(2, 0)
        self.splitter.setCollapsible(2, True)
        self.splitter.setSizes([280, 1320, self._assistant_width])

    # --- assembly ---

    def _build_panels(self) -> None:
        """Create the panels and wire them: the events each one reacts to beyond its own, and the callbacks."""
        args, state, status, lock = self.arguments, self.project_state, self.status, self.project_lock
        self.execute_panel = ExecutePanel(
            args,
            state,
            status,
            self,
            run_state_changed=lambda running: self.report_panel.set_execution_running(running),
            project_lock=lock,
        )
        self.editor_panel = EditorPanel(args, state, status, self, project_lock=lock)
        self.report_panel = ReportPanel(args, state, status, self.execute_panel.console.append_text, project_lock=lock)
        self.hotspot_panel = HotspotPanel(args, state, status, self.execute_panel.timeouts, project_lock=lock)
        self.pattern_detection_panel = PatternDetectionPanel(args, state, status, self, project_lock=lock)
        self.patch_repair_panel = PatchRepairPanel(args, state, status, project_lock=lock)
        self.autotuning_panel = AutotuningPanel(args, state, status, self, project_lock=lock)
        self.assistant_panel = AssistantPanel(args, state, status, parent=self, project_lock=lock)
        self.config_list = ConfigListPanel(
            args,
            state,
            status,
            confirm_leave_config=self.editor_panel.confirm_leave,
            open_compilation_editor=self.editor_panel.open_compilation_editor,
            tabs=self,
            project_lock=lock,
        )
        self.panels.append(self.config_list)

        self.splitter.addWidget(self.config_list)
        self.splitter.addWidget(self.tabs)
        for name, panel in (
            (TAB_EDITOR, self.editor_panel),
            (TAB_EXECUTE, self.execute_panel),
            (TAB_REPORT, self.report_panel),
            (TAB_HOTSPOT, self.hotspot_panel),
            (TAB_PATTERN_DETECTION, self.pattern_detection_panel),
            (TAB_PATCH_REPAIR, self.patch_repair_panel),
            (TAB_AUTOTUNING, self.autotuning_panel)
        ):
            self.add_tab(name, panel)
        self.splitter.addWidget(self.assistant_panel)

        # the autotuner records its measurements as execution results
        state.subscribe(AutotuningFinished, lambda _event: self.report_panel.update_display())
        # a profiling run (results) or a new configuration can make the Autotuning tab usable
        state.subscribe(ResultsChanged, lambda _event: self.autotuning_panel.update_tab_state())
        state.subscribe(ConfigurationsChanged, lambda _event: self.autotuning_panel.update_tab_state())
        self.autotuning_panel.update_tab_state()
        self.pattern_detection_panel.refresh()
        QShortcut(QKeySequence.StandardKey.Save, self, self.editor_panel.save_shortcut)

    def after_show(self) -> None:
        """Called once the window is visible: the wizard of a fresh project opens on top of it."""
        self.config_list.maybe_open_wizard(self.was_initialized)

    def add_tab(self, name: str, widget: QWidget) -> None:
        """Put ``widget`` at the place of tab ``name`` (replacing what is there) and register it if it is a panel.

        A state set before the tab existed (enabled, tooltip, title) is applied to it.
        """
        if name in self._tab_widgets:
            index = self.tabs.indexOf(self._tab_widgets[name])
            self.tabs.removeTab(index)
            self.tabs.insertTab(index, widget, name)
        else:
            self.tabs.addTab(widget, name)
        self._tab_widgets[name] = widget
        if isinstance(widget, Panel):
            self.panels.append(widget)
        if name in self._pending_tab_state:
            self.set_tab_enabled(name, *self._pending_tab_state.pop(name))
        if name in self._pending_tab_titles:
            self.set_tab_title(name, self._pending_tab_titles.pop(name))

    def tab(self, name: str) -> QWidget:
        return self._tab_widgets[name]

    def show_tab(self, name: str) -> None:
        self.tabs.setCurrentWidget(self._tab_widgets[name])

    def current_tab(self) -> str:
        current = self.tabs.currentWidget()
        return next((name for name, widget in self._tab_widgets.items() if widget is current), "")

    def set_tab_title(self, name: str, title: str) -> None:
        if name not in self._tab_widgets:
            self._pending_tab_titles[name] = title
            return
        self.tabs.setTabText(self.tabs.indexOf(self._tab_widgets[name]), title)

    def set_tab_enabled(self, name: str, enabled: bool, tooltip: str = "") -> None:
        if name not in self._tab_widgets:
            # a panel decides about its tab in its constructor, before the window added it
            self._pending_tab_state[name] = (enabled, tooltip)
            return
        index = self.tabs.indexOf(self._tab_widgets[name])
        self.tabs.setTabEnabled(index, enabled)
        self.tabs.setTabToolTip(index, tooltip)

    def _publish_project_busy(self, _holder: Optional[LockHolder]) -> None:
        self.project_state.publish(ProjectBusyChanged())

    # --- lifecycle ---

    def cancel_running_jobs(self) -> None:
        """Stop the jobs the panels started.

        Several of them run in a session of their own, which neither the closed window nor a Ctrl+C in the terminal
        reaches.
        """
        for panel in self.panels:
            panel.cancel_jobs()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 (Qt's name)
        self.cancel_running_jobs()
        super().closeEvent(event)

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 (Qt's name)
        super().resizeEvent(event)
        # the path takes at most half of the status bar; the status text keeps the rest
        self.dot_dp_label.setMaximumWidth(max(self.width() // 2, 100))

    def toggle_right_panel(self) -> None:
        sizes = self.splitter.sizes()
        if sizes[2] > 0:
            self._assistant_width = sizes[2]
            sizes[1] += sizes[2]
            sizes[2] = 0
        else:
            width = self._assistant_width
            sizes[1] -= width
            sizes[2] = width
        self.splitter.setSizes(sizes)
        self._sync_right_toggle()

    def _sync_right_toggle(self) -> None:
        width = self.splitter.sizes()[2]
        if width > 0:
            self._right_width = width
        self.right_toggle.setChecked(width > 0)
