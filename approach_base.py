"""
Common interface for the three selectable execution approaches.

The workflow is: prepare a domain, topology, mesh and model ONCE, then choose one
approach to compile and run. ABM, CompuCell3D and MPC are peers here -- selecting
one must never require, or silently substitute, another.

  * ABM          -- agent/cell-level Cellular Potts dynamics (existing engine)
  * CompuCell3D  -- the external CC3D simulator, driven through its own project files
  * MPC          -- Model Predictive Control: repeated constrained optimisation of a
                    control sequence against a predictive model. MPC is a CONTROL
                    method, not a discretisation scheme, and is modelled as such.

Why adapters run synchronously
------------------------------
Each adapter exposes a blocking ``run()`` that takes a :class:`RunContext`. Threads,
pause/resume, cancellation and state transitions live in ``run_manager``, so an
adapter stays a plain testable function of its inputs and cannot corrupt run state.
An adapter cooperates with control flow only by calling ``context.checkpoint()``.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from geometry import ValidationIssue

# =============================================================================
# Run lifecycle
# =============================================================================
#: Every state a run can occupy. Ordering is documentation, not a progression:
#: a run can go ready -> compiling -> failed without ever running.
RUN_STATES: Tuple[str, ...] = (
    "draft",        # configuration incomplete
    "invalid",      # validation found errors
    "ready",        # validated AND compiled successfully
    "compiling",    # compilation in progress
    "queued",       # accepted, waiting for a worker
    "running",
    "paused",
    "completed",    # finished successfully
    "failed",       # validation, compilation or execution error
    "cancelled",
)

TERMINAL_STATES: Tuple[str, ...] = ("completed", "failed", "cancelled")

#: States from which a run may still be started.
STARTABLE_STATES: Tuple[str, ...] = ("ready", "queued")


class RunCancelled(Exception):
    """Raised inside an adapter when the run was cancelled at a checkpoint.

    A distinct type so ``run_manager`` records 'cancelled' rather than 'failed':
    a user-requested stop is not an error.
    """


class ApproachUnavailable(Exception):
    """Raised when an approach's external dependency is missing.

    Deliberately distinct from a validation error. It means "this approach cannot
    run on this machine", which the UI must report honestly rather than falling
    back to a different approach.
    """


# =============================================================================
# Run context: the only channel between an adapter and the run manager
# =============================================================================
@dataclass
class RunContext:
    """Progress reporting and cooperative control for one run."""

    run_id: str
    seed: Optional[int] = None
    on_progress: Optional[Callable[[float, str], None]] = None
    on_log: Optional[Callable[[str, str], None]] = None
    #: Returns True when the run should stop. Supplied by the run manager.
    is_cancelled: Callable[[], bool] = lambda: False
    #: Blocks while the run is paused. Supplied by the run manager.
    wait_if_paused: Callable[[], None] = lambda: None

    def log(self, message: str, level: str = "info") -> None:
        if self.on_log:
            self.on_log(str(message), str(level))

    def progress(self, fraction: float, message: str = "") -> None:
        if self.on_progress:
            self.on_progress(max(0.0, min(1.0, float(fraction))), str(message))

    def checkpoint(self) -> None:
        """Yield to the run manager. Adapters must call this in their step loop.

        Honours a pause first (so a paused run does not spin) and then raises
        :class:`RunCancelled` if cancellation was requested.
        """
        self.wait_if_paused()
        if self.is_cancelled():
            raise RunCancelled(f"Run {self.run_id} was cancelled.")


# =============================================================================
# Capabilities
# =============================================================================
@dataclass
class Capabilities:
    """What an approach can do on THIS machine, right now.

    ``available`` is resolved at call time, not import time, so installing a
    dependency does not require a restart. When it is False,
    ``unavailable_reason`` must name the missing dependency precisely -- a vague
    "not available" is what drives a user to guess.
    """

    approach_id: str
    label: str
    available: bool = True
    unavailable_reason: str = ""
    #: Spatial dimensions the approach can run, e.g. (1, 2).
    dimensions: Tuple[int, ...] = (2,)
    supports_pause: bool = False
    supports_cancel: bool = True
    supports_fields: bool = False
    supports_seed: bool = False
    deterministic_with_seed: bool = False
    supports_export: bool = False
    engine_name: str = ""
    engine_version: str = ""
    requirements: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "approach_id": self.approach_id,
            "label": self.label,
            "available": bool(self.available),
            "unavailable_reason": self.unavailable_reason,
            "dimensions": list(self.dimensions),
            "supports_pause": bool(self.supports_pause),
            "supports_cancel": bool(self.supports_cancel),
            "supports_fields": bool(self.supports_fields),
            "supports_seed": bool(self.supports_seed),
            "deterministic_with_seed": bool(self.deterministic_with_seed),
            "supports_export": bool(self.supports_export),
            "engine_name": self.engine_name,
            "engine_version": self.engine_version,
            "requirements": list(self.requirements),
            "notes": self.notes,
        }


# =============================================================================
# The adapter interface
# =============================================================================
class ApproachAdapter(ABC):
    """One selectable modelling/execution approach.

    Contract:
      * ``get_capabilities`` -- never raises; reports availability honestly.
      * ``validate``         -- returns issues; never raises on bad user input.
      * ``compile``          -- raises on failure. A caller that catches the
                                exception must NOT report the run as ready.
      * ``run``              -- blocking; calls ``context.checkpoint()`` regularly;
                                raises :class:`RunCancelled` when cancelled.
      * ``export_configuration`` -- a portable description of the configured run,
                                available even when the engine itself is not.
    """

    approach_id: str = ""
    label: str = ""

    @abstractmethod
    def get_capabilities(self) -> Capabilities:
        ...

    @abstractmethod
    def validate(self, project: Dict[str, Any]) -> List[ValidationIssue]:
        ...

    @abstractmethod
    def compile(self, project: Dict[str, Any]) -> Dict[str, Any]:
        ...

    @abstractmethod
    def run(self, compiled: Dict[str, Any], context: RunContext) -> Dict[str, Any]:
        ...

    def export_configuration(self, project: Dict[str, Any]) -> Dict[str, Any]:
        """Portable configuration for this approach. Overridden where a real
        external format exists (CompuCell3D writes a runnable project)."""
        return {
            "approach": self.approach_id,
            "label": self.label,
            "configuration": dict((project or {}).get("approaches", {}).get(self.approach_id, {})),
        }

    # -- shared helpers -----------------------------------------------------
    def approach_config(self, project: Dict[str, Any]) -> Dict[str, Any]:
        """This approach's own settings, isolated from the other two.

        Switching approaches keeps the common geometry/mesh/model and leaves each
        approach's block untouched, which is what lets a user compare approaches
        without re-entering configuration.
        """
        return dict((project or {}).get("approaches", {}).get(self.approach_id, {}) or {})

    def require_available(self) -> None:
        capabilities = self.get_capabilities()
        if not capabilities.available:
            raise ApproachUnavailable(capabilities.unavailable_reason or
                                      f"{self.label} is not available on this machine.")


# =============================================================================
# Registry
# =============================================================================
_REGISTRY: Dict[str, ApproachAdapter] = {}

#: Adapter modules that register themselves on import. Loaded lazily rather than at
#: the top of this file, because each of them imports THIS module.
_BUILTIN_ADAPTER_MODULES: Tuple[str, ...] = (
    "approach_abm", "approach_cc3d", "approach_mpc",
)

_load_attempted = False
#: Import failures, kept so a broken adapter is REPORTED rather than silently absent.
_load_errors: Dict[str, str] = {}


def ensure_approaches_loaded() -> None:
    """Import the built-in adapters once, so the registry never depends on the
    caller's import order.

    Without this, an approach existed only if some module up the call stack happened
    to import it: a caller that imported just the CompuCell3D adapter would be told
    ABM and MPC do not exist. Registration is a side effect of import, so the registry
    has to own that import rather than hope for it.
    """
    global _load_attempted
    if _load_attempted:
        return
    # Set first: an adapter importing this module must not re-enter the loader.
    _load_attempted = True
    import importlib

    for module_name in _BUILTIN_ADAPTER_MODULES:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            # One unimportable adapter must not hide the other two.
            _load_errors[module_name] = f"{type(exc).__name__}: {exc}"


def register_approach(adapter: ApproachAdapter) -> ApproachAdapter:
    if not adapter.approach_id:
        raise ValueError("An approach adapter needs an approach_id.")
    _REGISTRY[adapter.approach_id] = adapter
    return adapter


def get_approach(approach_id: str) -> ApproachAdapter:
    ensure_approaches_loaded()
    key = str(approach_id or "").strip()
    if key not in _REGISTRY:
        known = ", ".join(sorted(_REGISTRY)) or "none registered"
        detail = ""
        if _load_errors:
            detail = (" Some adapters failed to load: "
                      + "; ".join(f"{k} ({v})" for k, v in _load_errors.items()))
        raise KeyError(f"Unknown approach {approach_id!r}. Available: {known}.{detail}")
    return _REGISTRY[key]


def list_approaches() -> List[Dict[str, Any]]:
    """Capabilities for every registered approach, available or not.

    Unavailable approaches are still listed, with the reason, so the UI can show
    an honest disabled state instead of hiding the option.
    """
    ensure_approaches_loaded()
    out: List[Dict[str, Any]] = []
    for key in sorted(_REGISTRY):
        try:
            out.append(_REGISTRY[key].get_capabilities().to_dict())
        except Exception as exc:                       # capabilities must never break the list
            out.append(Capabilities(
                approach_id=key, label=key, available=False,
                unavailable_reason=f"Capability check failed ({type(exc).__name__}).",
            ).to_dict())
    # An adapter that could not even be imported is reported, not omitted.
    for module_name, error in _load_errors.items():
        out.append(Capabilities(
            approach_id=module_name, label=module_name, available=False,
            unavailable_reason=f"This approach module failed to load: {error}",
        ).to_dict())
    return out


def available_approach_ids() -> List[str]:
    return [c["approach_id"] for c in list_approaches() if c["available"]]


def registered_ids() -> List[str]:
    ensure_approaches_loaded()
    return sorted(_REGISTRY)


def load_errors() -> Dict[str, str]:
    ensure_approaches_loaded()
    return dict(_load_errors)
