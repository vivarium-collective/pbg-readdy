"""ReaDDy composite documents + composite-spec discovery.

Two flavors of composite construction live in this package:

1. **Hand-coded factory** -- `make_readdy_document(...)` builds a PBG
   state-dict programmatically for callers that want full control over
   species, reactions, potentials, and initial particles. Used by
   `demo/demo_report.py` for the four ReaDDy demos.

2. **Declarative `*.composite.yaml`** -- sibling files in this directory
   follow the pbg-superpowers composite-spec convention.
   `build_composite()` loads one by name and instantiates
   `process_bigraph.Composite` with parameter substitution. The
   dashboard's composite explorer discovers these automatically once the
   package is installed in a workspace.

Both flavors are equivalent -- pick the one that fits your use case.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Any

import yaml
from process_bigraph import allocate_core
from process_bigraph.emitter import RAMEmitter

from pbg_readdy.processes import ReaDDyProcess


# ---------------------------------------------------------------------------
# Hand-coded composite factory (legacy / programmatic API)
# ---------------------------------------------------------------------------

def make_readdy_document(
    box_size=None,
    species=None,
    reactions=None,
    potentials=None,
    initial_particles=None,
    timestep=0.01,
    observe_stride=100,
    reaction_handler='Gillespie',
    interval=1.0,
):
    """Create a composite document for a ReaDDy reaction-diffusion simulation.

    Returns a document dict ready for use with Composite().

    Args:
        box_size: Simulation box [x, y, z] (default [10, 10, 10])
        species: Dict of species name -> diffusion constant
        reactions: List of reaction dicts with 'descriptor' and 'rate'
        potentials: List of potential dicts
        initial_particles: Dict of species name -> list of [x,y,z] positions
        timestep: Integration timestep
        observe_stride: Observable recording stride
        reaction_handler: 'Gillespie' or 'UncontrolledApproximation'
        interval: Time interval between process updates

    Returns:
        dict: Composite document with ReaDDy process, stores, and emitter
    """
    if box_size is None:
        box_size = [10.0, 10.0, 10.0]
    if species is None:
        species = {'A': 1.0}
    if reactions is None:
        reactions = []
    if potentials is None:
        potentials = []
    if initial_particles is None:
        initial_particles = {}

    return {
        'readdy': {
            '_type': 'process',
            'address': 'local:ReaDDyProcess',
            'config': {
                'box_size': box_size,
                'species': species,
                'reactions': reactions,
                'potentials': potentials,
                'initial_particles': initial_particles,
                'timestep': timestep,
                'observe_stride': observe_stride,
                'reaction_handler': reaction_handler,
            },
            'interval': interval,
            'inputs': {},
            'outputs': {
                'particle_counts': ['stores', 'particle_counts'],
                'total_particles': ['stores', 'total_particles'],
                'positions': ['stores', 'positions'],
                'energy': ['stores', 'energy'],
                'time': ['stores', 'time'],
            },
        },
        'stores': {},
        'emitter': {
            '_type': 'step',
            'address': 'local:ram-emitter',
            'config': {
                'emit': {
                    'total_particles': 'integer',
                    'energy': 'float',
                    'time': 'float',
                },
            },
            'inputs': {
                'total_particles': ['stores', 'total_particles'],
                'energy': ['stores', 'energy'],
                'time': ['global_time'],
            },
        },
    }


# ---------------------------------------------------------------------------
# Core registration
# ---------------------------------------------------------------------------

def register_readdy(core=None):
    """Return a core with ReaDDyProcess, the RAM emitter, and the
    ReaDDy Visualization(s) registered."""
    if core is None:
        core = allocate_core()
    core.register_link('ReaDDyProcess', ReaDDyProcess)
    core.register_link('ram-emitter', RAMEmitter)
    # Also register under the CamelCase class name so *.composite.yaml
    # specs that reference `local:RAMEmitter` resolve identically.
    core.register_link('RAMEmitter', RAMEmitter)
    # Register Visualization Steps so composites can wire them by name.
    from pbg_readdy.visualizations import ReaDDyPlots
    core.register_link('ReaDDyPlots', ReaDDyPlots)
    return core


# ---------------------------------------------------------------------------
# Declarative composite-spec loader (*.composite.yaml)
# ---------------------------------------------------------------------------

_COMPOSITES_DIR = Path(__file__).parent

_FULL_PLACEHOLDER = re.compile(r"^\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}$")
_INLINE_PLACEHOLDER = re.compile(r"\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _cast(value: Any, declared_type: str | None) -> Any:
    if declared_type is None:
        return value
    if declared_type == "float":
        return float(value)
    if declared_type == "int":
        return int(value)
    if declared_type in ("string", "str"):
        return str(value)
    if declared_type == "bool":
        if isinstance(value, str):
            return value.strip().lower() in ("true", "1", "yes")
        return bool(value)
    return value


def _substitute(state: Any, params: dict, overrides: dict) -> Any:
    if isinstance(state, dict):
        return {k: _substitute(v, params, overrides) for k, v in state.items()}
    if isinstance(state, list):
        return [_substitute(v, params, overrides) for v in state]
    if isinstance(state, str):
        m = _FULL_PLACEHOLDER.match(state)
        if m:
            pname = m.group(1)
            pdef = params.get(pname, {})
            raw = overrides.get(pname, pdef.get("default"))
            return _cast(raw, pdef.get("type"))
        if _INLINE_PLACEHOLDER.search(state):
            return _INLINE_PLACEHOLDER.sub(
                lambda mm: str(overrides.get(mm.group(1), params.get(mm.group(1), {}).get("default", ""))),
                state,
            )
    return state


def list_composite_specs() -> list[str]:
    """Return short names of every `*.composite.yaml` shipped in this package."""
    out: list[str] = []
    for path in sorted(_COMPOSITES_DIR.glob("*.composite.yaml")):
        out.append(path.name[: -len(".composite.yaml")])
    return out


def load_composite_spec(name: str) -> dict:
    """Load and parse a named composite spec. `name` is the stem (no suffix)."""
    path = _COMPOSITES_DIR / f"{name}.composite.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"composite spec not found: {path}")
    return yaml.safe_load(path.read_text())


def build_composite(name: str, *, overrides: dict | None = None, core=None):
    """Load a *.composite.yaml by name and instantiate process_bigraph.Composite.

    overrides: parameter overrides (keys must match spec.parameters)
    core:      optional pre-built core; otherwise register_readdy() is used
    """
    from process_bigraph import Composite

    spec = load_composite_spec(name)
    if not isinstance(spec, dict) or "state" not in spec or "name" not in spec:
        raise ValueError(f"composite '{name}' missing required keys (name, state)")

    if core is None:
        core = register_readdy()

    params = spec.get("parameters") or {}
    state = _substitute(spec.get("state") or {}, params, overrides or {})
    return Composite({"state": state}, core=core)
