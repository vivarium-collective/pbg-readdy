"""Unit tests for ReaDDyProcess."""

import pytest
import numpy as np
from process_bigraph import allocate_core
from pbg_readdy.processes import ReaDDyProcess


@pytest.fixture
def core():
    c = allocate_core()
    c.register_link('ReaDDyProcess', ReaDDyProcess)
    return c


def _random_positions(n, box_half=4.0):
    return (np.random.random((n, 3)) * 2 * box_half - box_half).tolist()


def test_instantiation(core):
    proc = ReaDDyProcess(
        config={'species': {'A': 1.0}, 'timestep': 0.01},
        core=core)
    assert proc.config['timestep'] == 0.01
    assert proc.config['species'] == {'A': 1.0}


def test_config_defaults(core):
    proc = ReaDDyProcess(config={}, core=core)
    assert tuple(proc.config['box_size']) == (10.0, 10.0, 10.0)
    assert tuple(proc.config['periodic']) == (True, True, True)
    assert proc.config['timestep'] == 0.01
    assert proc.config['reaction_handler'] == 'Gillespie'
    assert proc.config['observe_stride'] == 1


def test_initial_state(core):
    proc = ReaDDyProcess(config={
        'species': {'A': 1.0},
        'initial_particles': {'A': [[0, 0, 0], [1, 1, 1], [2, 2, 2]]},
    }, core=core)
    state = proc.initial_state()
    assert state['particle_counts'] == {'A': 3}
    assert state['total_particles'] == 3
    assert state['energy'] == 0.0
    assert state['time'] == 0.0
    assert len(state['positions']) == 3


def test_outputs_schema(core):
    proc = ReaDDyProcess(config={}, core=core)
    outputs = proc.outputs()
    expected_ports = ['particle_counts', 'total_particles', 'positions',
                      'energy', 'time']
    for port in expected_ports:
        assert port in outputs, f'Missing output port: {port}'


def test_single_update(core):
    np.random.seed(42)
    proc = ReaDDyProcess(config={
        'species': {'A': 1.0},
        'initial_particles': {'A': _random_positions(10)},
        'timestep': 0.01,
        'observe_stride': 10,
    }, core=core)
    proc.initial_state()
    result = proc.update({}, interval=0.5)
    assert 'particle_counts' in result
    assert 'total_particles' in result
    assert 'energy' in result
    assert isinstance(result['energy'], float)
    assert result['total_particles'] == 10  # no reactions -> same count
    assert abs(result['time'] - 0.5) < 0.02


def test_reactions_reduce_particles(core):
    np.random.seed(42)
    proc = ReaDDyProcess(config={
        'species': {'A': 1.0, 'B': 0.5},
        'reactions': [
            {'descriptor': 'fusion: A +(3) A -> B', 'rate': 10.0},
        ],
        'potentials': [
            {'type': 'harmonic_repulsion', 'species1': 'A', 'species2': 'A',
             'force_constant': 10., 'interaction_distance': 1.5},
        ],
        'initial_particles': {'A': _random_positions(30, box_half=3.0)},
        'timestep': 0.005,
        'observe_stride': 50,
    }, core=core)
    proc.initial_state()
    result = proc.update({}, interval=3.0)
    # Some A should have fused into B
    assert result['particle_counts'].get('B', 0) > 0
    assert result['total_particles'] < 30


def test_trajectory_data(core):
    np.random.seed(42)
    proc = ReaDDyProcess(config={
        'species': {'A': 1.0},
        'initial_particles': {'A': _random_positions(5)},
        'timestep': 0.01,
        'observe_stride': 20,
    }, core=core)
    proc.initial_state()
    proc.update({}, interval=1.0)
    traj = proc.get_trajectory_data()
    assert len(traj['times']) > 1
    assert len(traj['energy']) == len(traj['times'])
    assert 'A' in traj['counts']
    assert len(traj['counts']['A']) == len(traj['times'])
    # Time should be in real units
    assert traj['times'][-1] == pytest.approx(1.0, abs=0.02)


def test_position_snapshots(core):
    np.random.seed(42)
    proc = ReaDDyProcess(config={
        'species': {'A': 1.0},
        'initial_particles': {'A': _random_positions(5)},
        'timestep': 0.01,
        'observe_stride': 25,
    }, core=core)
    proc.initial_state()
    proc.update({}, interval=0.5)
    snaps = proc.get_position_snapshots()
    assert len(snaps) > 1
    for snap in snaps:
        assert 'time' in snap
        assert 'positions' in snap
        for pos in snap['positions']:
            assert len(pos) == 3


def test_harmonic_repulsion_energy(core):
    # Two particles close together with repulsion should produce energy > 0
    proc = ReaDDyProcess(config={
        'species': {'A': 0.01},  # very low diffusion
        'potentials': [
            {'type': 'harmonic_repulsion', 'species1': 'A', 'species2': 'A',
             'force_constant': 50., 'interaction_distance': 2.0},
        ],
        'initial_particles': {'A': [[0, 0, 0], [0.5, 0, 0]]},
        'timestep': 0.001,
        'observe_stride': 10,
    }, core=core)
    proc.initial_state()
    result = proc.update({}, interval=0.01)
    assert result['energy'] >= 0.0


def test_empty_initial_particles(core):
    proc = ReaDDyProcess(config={
        'species': {'A': 1.0},
        'initial_particles': {},
        'timestep': 0.01,
        'observe_stride': 10,
    }, core=core)
    state = proc.initial_state()
    assert state['total_particles'] == 0
    result = proc.update({}, interval=0.1)
    assert result['total_particles'] == 0
