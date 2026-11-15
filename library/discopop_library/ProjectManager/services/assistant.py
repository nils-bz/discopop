# This file is part of the DiscoPoP software (http://www.discopop.tu-darmstadt.de)
#
# Copyright (c) 2020, Technische Universitaet Darmstadt, Germany
#
# This software may be modified and distributed under the terms of
# the 3-Clause BSD License.  See the LICENSE file in the package base
# directory for details.

"""PROTOTYPE: an LLM assistant for the GUI, driving DiscoPoP through the tools of the MCP server.

The agent loop runs here, without any GUI toolkit, on a ``WorkRunner``: send the conversation and the tools to the
model, run the tools it calls, send the results back, until it answers without a tool call.

- the model is behind :class:`ModelSession` (one conversation); ``assistant_anthropic.py`` implements it for Claude,
  a test with a scripted fake, another provider with its own class
- the DiscoPoP tools are the tool modules of ``mcp_server`` (``TOOL`` + ``handle``), called *in-process*: no MCP
  subprocess, and their work runs on the GUI's own runner, so it is cancelled like any other GUI job
- the project is the GUI's: ``project_path`` is filled in here and not shown to the model
- the assistant never writes a source file: ``propose_edit`` records a change, the user applies it in the panel
  (:func:`apply_proposal`, on the main thread)
"""

import importlib
import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Set

from discopop_library.ProjectManager.services.jobs import WorkContext
from discopop_library.ProjectManager.services.project_state import HotspotsChanged, PatternsChanged, ResultsChanged

# the tools of the MCP server offered to the model: analysis, pipeline and checking -- not the setup tools, the GUI
# has its own wizard and editors for those
MCP_TOOL_MODULES = (
    "get_project_status",
    "get_configurations",
    "gather_data",
    "get_hotspots",
    "get_parallelization_suggestions",
    "explain_parallelization",
    "get_side_effects",
    "run_configuration",
)

# the ProjectState events to publish after a turn that called the tool
_CHANGES = {
    "gather_data": (HotspotsChanged, PatternsChanged),
    "run_configuration": (ResultsChanged,),
}

# an answer that never ends calls tools forever; the turn stops after this many model calls
MAX_STEPS = 25
MAX_READ_LINES = 400

SYSTEM_PROMPT = (
    "You are the assistant of the DiscoPoP GUI, a tool that finds parallelism in C/C++ programs by profiling them. "
    "The user has the project at {project} open. Use the tools to answer from measured data: get_project_status "
    "first, gather_data when results are missing or stale, then get_hotspots, get_parallelization_suggestions, "
    "explain_parallelization and get_side_effects. Read code with read_source. You cannot change files: propose "
    "every change with propose_edit; the user reviews and applies it, then run_configuration checks that it builds, "
    "gives the same results and runs faster. Answer briefly; name files and lines."
)


@dataclass(frozen=True)
class ToolSpec:
    """A tool as the model sees it."""

    name: str
    description: str
    input_schema: Dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    text: str
    is_error: bool = False


@dataclass(frozen=True)
class ModelTurn:
    """One answer of the model: its text, the tools it calls, and why it stopped (``stop`` is informative only)."""

    text: str
    tool_calls: List[ToolCall]
    stop: str = ""


class ModelSession(Protocol):
    """One conversation with a model; the session keeps the history in the format of its provider."""

    def add_user_message(self, text: str) -> None: ...

    def next_turn(self, system: str, tools: Sequence[ToolSpec]) -> ModelTurn:
        """Send the conversation so far; the answer becomes part of it."""
        ...

    def add_tool_results(self, results: Sequence[ToolResult]) -> None:
        """The results of all calls of the last turn, in one message."""
        ...


@dataclass(frozen=True)
class EditProposal:
    """A change the assistant proposes: replace ``old`` (once in the file) by ``new``."""

    id: int
    path: str  # relative to the project
    old: str
    new: str
    reason: str


@dataclass
class TurnResult:
    """What a turn left for the panel: proposals to review and the ProjectState events to publish."""

    proposals: List[EditProposal] = field(default_factory=list)
    changes: List[type] = field(default_factory=list)


class ToolModule(Protocol):
    """A tool module of ``mcp_server.tools``: ``TOOL`` (an ``mcp.types.Tool``) and ``handle``."""

    TOOL: Any

    def handle(self, arguments: Dict[str, Any], ctx: Any) -> Any: ...


def load_mcp_tools() -> List[Any]:
    """The tool modules of the installed MCP server; ImportError when it is not installed."""
    return [importlib.import_module(f"mcp_server.tools.{name}") for name in MCP_TOOL_MODULES]


def _without_project_path(schema: Dict[str, Any]) -> Dict[str, Any]:
    properties = {k: v for k, v in schema.get("properties", {}).items() if k != "project_path"}
    required = [r for r in schema.get("required", []) if r != "project_path"]
    return {**schema, "properties": properties, "required": required}


_READ_SOURCE = ToolSpec(
    "read_source",
    "Read lines of a source file of the project, with line numbers.",
    {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Relative to the project root."},
            "start_line": {"type": "integer", "description": "First line, default 1."},
            "end_line": {"type": "integer", "description": f"Last line; at most {MAX_READ_LINES} lines are shown."},
        },
        "required": ["file_path"],
        "additionalProperties": False,
    },
)

_PROPOSE_EDIT = ToolSpec(
    "propose_edit",
    "Propose a change of a source file: replace old_text, which must occur exactly once in the file, by new_text "
    "(e.g. the same lines with an OpenMP pragma above them). The user reviews and applies it; until then the file "
    "is unchanged.",
    {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Relative to the project root."},
            "old_text": {"type": "string"},
            "new_text": {"type": "string"},
            "reason": {"type": "string", "description": "One sentence for the user: what the change does and why."},
        },
        "required": ["file_path", "old_text", "new_text", "reason"],
        "additionalProperties": False,
    },
)


def resolve_in_project(project_root: str, relative: str) -> Path:
    """``relative`` below the project root; ValueError for a path outside of it."""
    root = Path(project_root).resolve()
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError(f"{relative} is not a file of the project")
    return path


def apply_proposal(project_root: str, proposal: EditProposal) -> None:
    """Make the change of ``proposal``; ValueError when the file no longer contains ``old`` exactly once."""
    path = resolve_in_project(project_root, proposal.path)
    text = path.read_text()
    if text.count(proposal.old) != 1:
        raise ValueError(f"{proposal.path} has changed: the text to replace is not in it exactly once")
    path.write_text(text.replace(proposal.old, proposal.new, 1))


class Assistant:
    """The conversation of the panel. ``run_turn`` runs on a worker thread, one turn at a time."""
    def __init__(self, project_root: str, model: ModelSession, tool_modules: Sequence[ToolModule]) -> None:
        self.project_root = project_root
        self._model = model
        self._modules = {module.TOOL.name: module for module in tool_modules}
        self._next_proposal = 1
        self._ctx: Any = None  # the mcp_server ToolContext, created on the first DiscoPoP tool call (its caches)
        self.tools = [
            ToolSpec(m.TOOL.name, m.TOOL.description, _without_project_path(m.TOOL.inputSchema)) for m in tool_modules
        ] + [_READ_SOURCE, _PROPOSE_EDIT]

    def run_turn(self, user_text: str, work: WorkContext) -> TurnResult:
        result = TurnResult()
        called: Set[str] = set()
        self._model.add_user_message(user_text)
        system = SYSTEM_PROMPT.format(project=self.project_root)
        for _ in range(MAX_STEPS):
            if work.cancel_event.is_set():
                work.output("[stopped]")
                break
            work.status("Waiting for the model ...")
            turn = self._model.next_turn(system, self.tools)
            if turn.text:
                work.output(turn.text)
            if not turn.tool_calls:
                break
            results = []
            for call in turn.tool_calls:
                work.output(f"▶ {call.name}({_summary(call.arguments)})")
                called.add(call.name)
                results.append(self._call(call, work, result))
            self._model.add_tool_results(results)
        else:
            work.output(f"[stopped after {MAX_STEPS} model calls]")
        result.changes = [event for name in sorted(called) for event in _CHANGES.get(name, ())]
        return result

    # --- tools ---

    def _call(self, call: ToolCall, work: WorkContext, turn: TurnResult) -> ToolResult:
        try:
            if call.name == _READ_SOURCE.name:
                return ToolResult(call.id, self._read_source(call.arguments))
            if call.name == _PROPOSE_EDIT.name:
                return ToolResult(call.id, self._propose_edit(call.arguments, turn))
            if call.name in self._modules:
                return self._call_discopop(call, work)
            return ToolResult(call.id, f"There is no tool {call.name}.", is_error=True)
        except Exception as e:  # the model gets the error and can correct its call
            return ToolResult(call.id, f"{type(e).__name__}: {e}", is_error=True)

    def _call_discopop(self, call: ToolCall, work: WorkContext) -> ToolResult:
        from mcp_server.argument_coercion import coerce_arguments, validation_error
        from mcp_server.tools.helpers import ToolContext

        module = self._modules[call.name]
        schema = module.TOOL.inputSchema
        arguments, _ = coerce_arguments({**call.arguments, "project_path": self.project_root}, schema)
        invalid = validation_error(call.name, arguments, schema)
        if invalid:
            return ToolResult(call.id, invalid, is_error=True)
        if self._ctx is None:
            self._ctx = ToolContext()
        ctx = self._ctx

        def progress(_done: float, _total: Optional[float], message: Optional[str]) -> None:
            if message:
                work.status(f"{call.name}: {message}")

        finished = threading.Event()

        def forward_cancel() -> None:
            # the tool's processes are the ToolContext's, not the WorkContext's: pass the cancel on
            while not finished.wait(0.2):
                if work.cancel_event.is_set():
                    ctx.cancel()
                    return

        threading.Thread(target=forward_cancel, daemon=True).start()
        try:
            with ctx.reporting_progress(progress), ctx.cancellable():
                content = module.handle(arguments, ctx)
        finally:
            finished.set()
        text = "\n".join(getattr(block, "text", "") for block in content)
        return ToolResult(call.id, text, is_error=_is_error(text))

    def _read_source(self, arguments: Dict[str, Any]) -> str:
        path = resolve_in_project(self.project_root, str(arguments["file_path"]))
        lines = path.read_text(errors="replace").splitlines()
        start = max(int(arguments.get("start_line", 1)), 1)
        end = min(int(arguments.get("end_line", start + MAX_READ_LINES - 1)), start + MAX_READ_LINES - 1, len(lines))
        shown = "\n".join(f"{n:>5}  {lines[n - 1]}" for n in range(start, end + 1))
        return shown + (f"\n(the file has {len(lines)} lines)" if end < len(lines) else "")

    def _propose_edit(self, arguments: Dict[str, Any], turn: TurnResult) -> str:
        relative = str(arguments["file_path"])
        old, new = str(arguments["old_text"]), str(arguments["new_text"])
        count = resolve_in_project(self.project_root, relative).read_text().count(old)
        if count != 1:
            raise ValueError(f"old_text occurs {count} times in {relative}; it has to occur exactly once")
        proposal = EditProposal(self._next_proposal, relative, old, new, str(arguments.get("reason", "")))
        self._next_proposal += 1
        turn.proposals.append(proposal)
        return f"Proposal {proposal.id} recorded; the user reviews it after your answer. The file is unchanged."


def _summary(arguments: Dict[str, Any]) -> str:
    return ", ".join(f"{k}={v!r}" if len(repr(v)) <= 40 else f"{k}=…" for k, v in arguments.items())


def _is_error(text: str) -> bool:
    """The MCP tools report an error as JSON with status "error"."""
    try:
        data = json.loads(text)
    except ValueError:
        return False
    return isinstance(data, dict) and data.get("status") == "error"


ModelFactory = Callable[[], ModelSession]
