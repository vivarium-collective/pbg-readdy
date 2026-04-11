# pbg-readdy

Process-bigraph wrapper for [ReaDDy](https://readdy.github.io), a particle-based reaction-diffusion simulator.

Wraps ReaDDy's Brownian dynamics engine as a `process-bigraph` Process, enabling particle-based reaction-diffusion simulations to be composed with other biological models in the vivarium framework.

## Installation

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install
pip install -e .

# For development (tests, visualization)
pip install -e ".[dev]"
```

ReaDDy requires either conda or a recent pip wheel:
```bash
pip install readdy
```

## Quick Start

```python
import numpy as np
from process_bigraph import Composite, allocate_core
from process_bigraph.emitter import RAMEmitter
from pbg_readdy import ReaDDyProcess, make_readdy_document

core = allocate_core()
core.register_link('ReaDDyProcess', ReaDDyProcess)
core.register_link('ram-emitter', RAMEmitter)

# Create a document with 30 particles undergoing fusion
positions = (np.random.random((30, 3)) * 8 - 4).tolist()
doc = make_readdy_document(
    box_size=[10., 10., 10.],
    species={'A': 1.0, 'B': 0.5},
    reactions=[{'descriptor': 'fusion: A +(2) A -> B', 'rate': 5.0}],
    potentials=[{
        'type': 'harmonic_repulsion',
        'species1': 'A', 'species2': 'A',
        'force_constant': 10., 'interaction_distance': 1.5,
    }],
    initial_particles={'A': positions},
    timestep=0.005,
    observe_stride=50,
    interval=1.0,
)

sim = Composite({'state': doc}, core=core)
sim.run(5.0)
print(sim.state['stores']['particle_counts'])
```

## API Reference

### ReaDDyProcess

A time-driven `Process` that bridges ReaDDy's Brownian dynamics simulation.

| Port | Type | Direction | Description |
|------|------|-----------|-------------|
| `particle_counts` | `map[integer]` | output | Count per species |
| `total_particles` | `integer` | output | Total particle count |
| `positions` | `list` | output | Particle positions [[x,y,z], ...] |
| `energy` | `float` | output | Total potential energy |
| `time` | `float` | output | Simulation time |

**Config:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `box_size` | tuple | (10,10,10) | Simulation box dimensions |
| `periodic` | tuple | (True,True,True) | Periodic boundary conditions |
| `kbt` | float | 0.0 | Thermal energy (0 = ReaDDy default) |
| `timestep` | float | 0.01 | Integration timestep |
| `species` | dict | {'A': 1.0} | Species name -> diffusion constant |
| `reactions` | list | [] | Reaction descriptors (see below) |
| `potentials` | list | [] | Pair/external potentials (see below) |
| `initial_particles` | dict | {} | Species -> list of [x,y,z] positions |
| `observe_stride` | int | 1 | Observable recording stride (steps) |
| `reaction_handler` | str | 'Gillespie' | Reaction handler algorithm |

**Reaction formats:**
```python
# Descriptor string (fusion, fission, decay, conversion)
{'descriptor': 'fusion: A +(2) A -> B', 'rate': 5.0}

# Enzymatic (catalyst preserved)
{'method': 'enzymatic', 'name': 'eat', 'catalyst': 'B',
 'type_from': 'A', 'type_to': 'B', 'rate': 1.0, 'educt_distance': 3.0}
```

**Potential types:**
```python
{'type': 'harmonic_repulsion', 'species1': 'A', 'species2': 'A',
 'force_constant': 10., 'interaction_distance': 1.5}

{'type': 'lennard_jones', 'species1': 'A', 'species2': 'B',
 'm': 12, 'n': 6, 'cutoff': 3.0, 'epsilon': 1.0, 'sigma': 1.0}

{'type': 'weak_interaction', 'species1': 'A', 'species2': 'A',
 'force_constant': 5., 'desired_distance': 1.0, 'depth': 2.0, 'cutoff': 3.0}

{'type': 'screened_electrostatics', 'species1': 'A', 'species2': 'B',
 'electrostatic_strength': 1.0, 'inverse_screening_depth': 0.1,
 'repulsion_strength': 1.0, 'repulsion_distance': 1.0, 'cutoff': 5.0}
```

### make_readdy_document()

Factory function that creates a ready-to-run composite document with ReaDDyProcess, stores, and a RAM emitter.

### Extra methods on ReaDDyProcess

- `get_trajectory_data()` — Returns full time-series: `{'times': [...], 'counts': {'A': [...], ...}, 'energy': [...]}`
- `get_position_snapshots(stride=None)` — Returns list of `{'time': t, 'positions': [[x,y,z], ...]}`

## Architecture

ReaDDy manages its own internal particle state. The wrapper uses the **bridge pattern**:

1. **Build** — lazily constructs the ReaDDy system with species, reactions, potentials
2. **Push** — initial particles are set at build time
3. **Run** — each `update(state, interval)` calls `simulation.run()` for the requested interval
4. **Read** — callback-based observables accumulate particle counts, energy, and positions

The callback approach (rather than trajectory files) allows `update()` to be called multiple times, supporting composition in a Composite with other processes.

## Demo

```bash
source .venv/bin/activate
PYTHONPATH=. python demo/demo_report.py
```

Generates `demo/report.html` — an interactive report with:
- 3D particle viewers (Three.js) with play/pause animation
- Population dynamics and energy charts (Plotly)
- Bigraph architecture diagrams (bigraph-viz)
- Interactive composite document trees

## License

MIT
