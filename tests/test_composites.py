"""Integration tests for ReaDDy composites."""

import pytest
import numpy as np
from process_bigraph import Composite, allocate_core
from process_bigraph.emitter import RAMEmitter
from pbg_readdy.processes import ReaDDyProcess
from pbg_readdy.composites import make_readdy_document


@pytest.fixture
def core():
    c = allocate_core()
    c.register_link('ReaDDyProcess', ReaDDyProcess)
    c.register_link('ram-emitter', RAMEmitter)
    return c


def test_make_readdy_document():
    doc = make_readdy_document(
        species={'A': 1.0},
        reactions=[{'descriptor': 'decay: A ->', 'rate': 0.1}],
        initial_particles={'A': [[0, 0, 0]]},
    )
    assert 'readdy' in doc
    assert doc['readdy']['_type'] == 'process'
    assert 'emitter' in doc
    assert doc['emitter']['_type'] == 'step'


def test_composite_run(core):
    np.random.seed(42)
    positions = (np.random.random((15, 3)) * 6 - 3).tolist()
    doc = make_readdy_document(
        species={'A': 1.0},
        initial_particles={'A': positions},
        timestep=0.01,
        observe_stride=10,
        interval=0.5,
    )
    sim = Composite({'state': doc}, core=core)
    sim.run(1.0)
    state = sim.state
    assert 'stores' in state
    assert state['stores']['total_particles'] == 15


def test_composite_with_reactions(core):
    np.random.seed(42)
    positions = (np.random.random((20, 3)) * 4 - 2).tolist()
    doc = make_readdy_document(
        species={'A': 1.0, 'B': 0.5},
        reactions=[
            {'descriptor': 'fusion: A +(3) A -> B', 'rate': 10.0},
        ],
        potentials=[
            {'type': 'harmonic_repulsion', 'species1': 'A', 'species2': 'A',
             'force_constant': 10., 'interaction_distance': 1.5},
        ],
        initial_particles={'A': positions},
        timestep=0.005,
        observe_stride=50,
        interval=1.0,
    )
    sim = Composite({'state': doc}, core=core)
    sim.run(3.0)
    state = sim.state
    counts = state['stores']['particle_counts']
    assert counts.get('B', 0) > 0
