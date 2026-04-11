"""Demo: ReaDDy multi-configuration reaction-diffusion report with 3D viewers.

Runs four distinct particle-based reaction-diffusion simulations
(actin treadmilling, Lotka-Volterra, living polymers, crowded diffusion),
generates interactive 3D particle viewers with Three.js, Plotly charts,
bigraph-viz diagrams, and navigatable PBG document trees in a single
self-contained HTML.
"""

import json
import os
import time
import base64
import tempfile
import numpy as np
from process_bigraph import allocate_core
from pbg_readdy.processes import ReaDDyProcess
from pbg_readdy.composites import make_readdy_document


# ── Helpers ────────────────────────────────────────────────────────

def _random_positions(n, box_half=4.0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return (np.random.random((n, 3)) * 2 * box_half - box_half).tolist()


def _random_positions_sphere(n, radius=6.0, seed=None):
    """Generate n random positions uniformly inside a sphere of given radius."""
    if seed is not None:
        np.random.seed(seed)
    positions = []
    while len(positions) < n:
        p = (np.random.random(3) * 2 - 1) * radius
        if np.linalg.norm(p) < radius:
            positions.append(p.tolist())
    return positions


def _downsample_snapshots(snapshots, max_snaps=50):
    if len(snapshots) > max_snaps:
        step = len(snapshots) // max_snaps
        return snapshots[::step]
    return snapshots


# ── Simulation 1: Actin Filament Treadmilling (topology) ──────────

def run_simulation_actin():
    """Run actin filament treadmilling using ReaDDy topologies directly."""
    import readdy

    np.random.seed(42)
    t0 = time.perf_counter()

    system = readdy.ReactionDiffusionSystem(box_size=[20., 20., 20.], unit_system=None)
    system.add_species('substrate', 0.5)
    system.add_topology_species('head', 0.1)
    system.add_topology_species('core', 0.1)
    system.add_topology_species('tail', 0.1)
    system.topologies.add_type('filament')

    # All bond types (needed for robustness during type changes)
    for t1, t2 in [('head', 'core'), ('core', 'core'), ('core', 'tail'),
                   ('head', 'tail'), ('tail', 'tail'), ('head', 'head')]:
        system.topologies.configure_harmonic_bond(t1, t2, force_constant=100, length=1.)

    # All angle types for rigidity
    all_types = ['head', 'core', 'tail']
    for t1 in all_types:
        for t2 in all_types:
            for t3 in all_types:
                system.topologies.configure_harmonic_angle(
                    t1, t2, t3, force_constant=20., equilibrium_angle=np.pi)

    # Polymerization at barbed (head) end
    system.topologies.add_spatial_reaction(
        'attach: filament(head) + (substrate) -> filament(core--head)',
        rate=15.0, radius=1.5)

    # Depolymerization at pointed (tail) end - structural reaction
    def depoly_rate(topology):
        vertices = topology.get_graph().get_vertices()
        return 0.02 if len(vertices) > 3 else 0.

    def depoly_reaction(topology):
        recipe = readdy.StructuralReactionRecipe(topology)
        vertices = topology.get_graph().get_vertices()
        for v in vertices:
            if topology.particle_type_of_vertex(v) == 'tail':
                neighbors = v.neighbors()
                if neighbors:
                    adj_idx = neighbors[0].get().particle_index
                    recipe.separate_vertex(v.particle_index)
                    recipe.change_particle_type(v.particle_index, 'substrate')
                    for vi, vv in enumerate(vertices):
                        if vv.particle_index == adj_idx:
                            recipe.change_particle_type(vi, 'tail')
                            break
                break
        return recipe

    system.topologies.add_structural_reaction(
        'depolymerize', 'filament', depoly_reaction, depoly_rate)

    system.potentials.add_harmonic_repulsion(
        'substrate', 'substrate', force_constant=10., interaction_distance=0.8)

    # Create simulation
    sim = system.simulation(kernel='CPU')
    sim.show_progress = False

    # Initial filament: head-core-core-tail
    positions = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=float)
    top = sim.add_topology('filament', ['head', 'core', 'core', 'tail'], positions)
    g = top.get_graph()
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(2, 3)

    # 300 substrate particles
    sub_pos = (np.random.random((300, 3)) - 0.5) * 18
    for p in sub_pos:
        sim.add_particle('substrate', p.tolist())

    # Set up observables via callbacks
    n_steps = 30000
    timestep = 0.005
    observe_stride = 100

    count_data = []
    energy_data = []

    species_order = ['head', 'core', 'substrate', 'tail']

    sim.observe.number_of_particles(
        stride=observe_stride, types=species_order,
        callback=lambda x: count_data.append(np.array(x)))
    sim.observe.energy(
        stride=observe_stride,
        callback=lambda x: energy_data.append(float(x)))

    # File-based trajectory + topology recording for bond extraction
    tmpf = tempfile.mktemp(suffix='.h5')
    sim.output_file = tmpf
    sim.record_trajectory(stride=observe_stride)
    sim.observe.topologies(stride=observe_stride)

    sim.run(n_steps, timestep=timestep)
    runtime = time.perf_counter() - t0

    # Build trajectory data
    n_points = len(energy_data)
    times = [i * observe_stride * timestep for i in range(n_points)]
    counts = {}
    for i, sp in enumerate(species_order):
        counts[sp] = [int(c[i]) for c in count_data]

    # Derive filament length series (head + core + tail)
    filament_length = []
    for j in range(n_points):
        fl = counts['head'][j] + counts['core'][j] + counts['tail'][j]
        filament_length.append(fl)
    counts['filament_length'] = filament_length

    traj_data = {
        'times': times,
        'counts': counts,
        'energy': list(energy_data),
    }

    # Read positions and topology bonds from HDF5 file
    import readdy as readdy_traj
    traj = readdy_traj.Trajectory(tmpf)
    time_top, topology_records = traj.read_observable_topologies()
    frames = list(traj.read())

    pos_snapshots = []
    for frame_idx, frame in enumerate(frames):
        positions = []
        id_to_idx = {}
        for fi, p in enumerate(frame):
            positions.append([float(p.position[0]), float(p.position[1]), float(p.position[2])])
            id_to_idx[p.id] = fi

        bonds = []
        if frame_idx < len(topology_records):
            for t in topology_records[frame_idx]:
                for e in t.edges:
                    pid1 = t.particles[e[0]]
                    pid2 = t.particles[e[1]]
                    if pid1 in id_to_idx and pid2 in id_to_idx:
                        bonds.append([id_to_idx[pid1], id_to_idx[pid2]])

        real_time = frame_idx * observe_stride * timestep
        pos_snapshots.append({
            'time': round(real_time, 6),
            'positions': positions,
            'bonds': bonds,
        })

    os.unlink(tmpf)

    return traj_data, pos_snapshots, runtime


# ── Simulation 2: Lotka-Volterra Predator-Prey (wrapper) ──────────

def run_simulation_lotka_volterra():
    """Run Lotka-Volterra predator-prey using the PBG wrapper."""
    np.random.seed(100)
    core = allocate_core()
    core.register_link('ReaDDyProcess', ReaDDyProcess)

    config = {
        'box_size': (15., 15., 15.),
        'species': {'A': 1.5, 'B': 1.5},
        'reactions': [
            {'descriptor': 'reproduce: A -> A +(2) A', 'rate': 0.3},
            {'method': 'enzymatic', 'name': 'eat', 'catalyst': 'B',
             'type_from': 'A', 'type_to': 'B', 'rate': 0.8,
             'educt_distance': 3.0},
            {'descriptor': 'death: B ->', 'rate': 0.4},
        ],
        'potentials': [
            {'type': 'harmonic_repulsion', 'species1': 'A',
             'species2': 'A', 'force_constant': 5.,
             'interaction_distance': 1.0},
            {'type': 'harmonic_repulsion', 'species1': 'B',
             'species2': 'B', 'force_constant': 5.,
             'interaction_distance': 1.0},
            {'type': 'harmonic_repulsion', 'species1': 'A',
             'species2': 'B', 'force_constant': 5.,
             'interaction_distance': 1.0},
        ],
        'initial_particles': {
            'A': _random_positions(30, 6.0, seed=100),
            'B': _random_positions(10, 6.0, seed=200),
        },
        'timestep': 0.005,
        'observe_stride': 50,
    }

    t0 = time.perf_counter()
    proc = ReaDDyProcess(config=config, core=core)
    proc.initial_state()

    n_steps = 10000
    interval = n_steps * config['timestep']
    proc.update({}, interval=interval)
    runtime = time.perf_counter() - t0

    traj_data = proc.get_trajectory_data()
    pos_snapshots = proc.get_position_snapshots()
    for snap in pos_snapshots:
        snap['bonds'] = []
    return traj_data, pos_snapshots, runtime


# ── Simulation 3: Living Polymer Equilibrium (topology) ───────────

def run_simulation_living_polymers():
    """Run living polymer equilibrium using ReaDDy topologies directly."""
    import readdy

    np.random.seed(300)
    t0 = time.perf_counter()

    system = readdy.ReactionDiffusionSystem(box_size=[40., 40., 40.], unit_system=None)
    system.topologies.add_type('Polymer')
    system.add_topology_species('Head', 0.05)
    system.add_topology_species('Tail', 0.05)

    # Bonds for all combos
    for t1, t2 in [('Head', 'Tail'), ('Tail', 'Tail'), ('Head', 'Head')]:
        system.topologies.configure_harmonic_bond(t1, t2, force_constant=50, length=1.)

    # Angles for all combos
    for t1 in ['Head', 'Tail']:
        for t2 in ['Head', 'Tail']:
            for t3 in ['Head', 'Tail']:
                system.topologies.configure_harmonic_angle(
                    t1, t2, t3, force_constant=10, equilibrium_angle=np.pi)

    # Association: two polymer heads merge
    system.topologies.add_spatial_reaction(
        'associate: Polymer(Head) + Polymer(Head) -> Polymer(Tail--Tail)',
        rate=0.5, radius=1.0)

    # Dissociation structural reaction (break random internal bond)
    def dissociation_rate(topology):
        vertices = topology.get_graph().get_vertices()
        n_edges = 0
        for v in vertices:
            n_edges += len(v.neighbors())
        n_edges //= 2  # each edge counted twice
        if n_edges > 2:
            return 0.00005 * n_edges
        return 0.

    def dissociation_reaction(topology):
        recipe = readdy.StructuralReactionRecipe(topology)
        graph = topology.get_graph()
        edges = graph.get_edges()

        if len(edges) <= 2:
            return recipe

        # Collect endpoint vertex indices (degree 1)
        vertices = graph.get_vertices()
        endpoint_indices = set()
        for vi, v in enumerate(vertices):
            if len(v.neighbors()) == 1:
                endpoint_indices.add(vi)

        # Collect internal edges (not touching endpoints)
        internal_edges = []
        for e in edges:
            vi1 = e[0].get().particle_index
            vi2 = e[1].get().particle_index
            # Find vertex indices
            idx1 = idx2 = None
            for vi, v in enumerate(vertices):
                if v.particle_index == vi1:
                    idx1 = vi
                if v.particle_index == vi2:
                    idx2 = vi
            if idx1 is not None and idx2 is not None:
                if idx1 not in endpoint_indices and idx2 not in endpoint_indices:
                    internal_edges.append((e, idx1, idx2))

        if not internal_edges:
            return recipe

        # Pick a random internal edge to break
        choice = internal_edges[np.random.randint(len(internal_edges))]
        edge, vi1, vi2 = choice

        recipe.remove_edge(edge[0], edge[1])
        recipe.change_particle_type(vi1, 'Head')
        recipe.change_particle_type(vi2, 'Head')

        return recipe

    system.topologies.add_structural_reaction(
        'dissociate', 'Polymer', dissociation_reaction, dissociation_rate)

    # Create simulation
    sim = system.simulation(kernel='CPU')
    sim.show_progress = False

    # Initialize 200 short polymers (4 particles: Head-Tail-Tail-Head)
    for i in range(200):
        cx, cy, cz = (np.random.random(3) - 0.5) * 36
        # Random orientation
        direction = np.random.randn(3)
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        positions_top = np.array([
            [cx - 1.5 * direction[0], cy - 1.5 * direction[1], cz - 1.5 * direction[2]],
            [cx - 0.5 * direction[0], cy - 0.5 * direction[1], cz - 0.5 * direction[2]],
            [cx + 0.5 * direction[0], cy + 0.5 * direction[1], cz + 0.5 * direction[2]],
            [cx + 1.5 * direction[0], cy + 1.5 * direction[1], cz + 1.5 * direction[2]],
        ])
        top = sim.add_topology('Polymer',
                               ['Head', 'Tail', 'Tail', 'Head'],
                               positions_top)
        g = top.get_graph()
        g.add_edge(0, 1)
        g.add_edge(1, 2)
        g.add_edge(2, 3)

    # Observables
    n_steps = 30000
    timestep = 0.5
    observe_stride = 100

    count_data = []
    energy_data = []

    species_order = ['Head', 'Tail']

    sim.observe.number_of_particles(
        stride=observe_stride, types=species_order,
        callback=lambda x: count_data.append(np.array(x)))
    sim.observe.energy(
        stride=observe_stride,
        callback=lambda x: energy_data.append(float(x)))

    # File-based trajectory + topology recording for bond extraction
    tmpf = tempfile.mktemp(suffix='.h5')
    sim.output_file = tmpf
    sim.record_trajectory(stride=observe_stride)
    sim.observe.topologies(stride=observe_stride)

    sim.run(n_steps, timestep=timestep)
    runtime = time.perf_counter() - t0

    # Build trajectory data
    n_points = len(energy_data)
    times = [i * observe_stride * timestep for i in range(n_points)]
    counts = {}
    for i, sp in enumerate(species_order):
        counts[sp] = [int(c[i]) for c in count_data]

    # Derive average chain length: total_particles / num_chains
    # num_chains = Head_count / 2 (each chain has 2 heads)
    avg_chain_length = []
    for j in range(n_points):
        total = counts['Head'][j] + counts['Tail'][j]
        n_chains = max(counts['Head'][j] / 2, 1)
        avg_chain_length.append(round(total / n_chains, 1))
    counts['avg_chain_length'] = avg_chain_length

    traj_data = {
        'times': times,
        'counts': counts,
        'energy': list(energy_data),
    }

    # Read positions and topology bonds from HDF5 file
    import readdy as readdy_traj
    traj = readdy_traj.Trajectory(tmpf)
    time_top, topology_records = traj.read_observable_topologies()
    frames = list(traj.read())

    pos_snapshots = []
    for frame_idx, frame in enumerate(frames):
        positions = []
        id_to_idx = {}
        for fi, p in enumerate(frame):
            positions.append([float(p.position[0]), float(p.position[1]), float(p.position[2])])
            id_to_idx[p.id] = fi

        bonds = []
        if frame_idx < len(topology_records):
            for t in topology_records[frame_idx]:
                for e in t.edges:
                    pid1 = t.particles[e[0]]
                    pid2 = t.particles[e[1]]
                    if pid1 in id_to_idx and pid2 in id_to_idx:
                        bonds.append([id_to_idx[pid1], id_to_idx[pid2]])

        real_time = frame_idx * observe_stride * timestep
        pos_snapshots.append({
            'time': round(real_time, 6),
            'positions': positions,
            'bonds': bonds,
        })

    os.unlink(tmpf)

    return traj_data, pos_snapshots, runtime


# ── Simulation 4: Crowded Diffusion with Spherical Confinement ────

def run_simulation_crowded_sphere():
    """Run crowded diffusion with spherical confinement using ReaDDy directly."""
    import readdy

    np.random.seed(400)
    t0 = time.perf_counter()

    system = readdy.ReactionDiffusionSystem(box_size=[20., 20., 20.], unit_system=None)
    system.periodic_boundary_conditions = [False, False, False]
    system.add_species('P', 0.5)
    system.potentials.add_harmonic_repulsion(
        'P', 'P', force_constant=20., interaction_distance=2.0)
    system.potentials.add_sphere(
        'P', force_constant=50., origin=[0, 0, 0], radius=7., inclusion=True)

    sim = system.simulation(kernel='CPU')
    sim.show_progress = False

    # 80 particles inside sphere of radius 6
    sphere_pos = _random_positions_sphere(80, radius=6.0, seed=400)
    for p in sphere_pos:
        sim.add_particle('P', p)

    n_steps = 10000
    timestep = 0.002
    observe_stride = 50

    count_data = []
    energy_data = []
    position_data = []

    species_order = ['P']

    sim.observe.number_of_particles(
        stride=observe_stride, types=species_order,
        callback=lambda x: count_data.append(np.array(x)))
    sim.observe.energy(
        stride=observe_stride,
        callback=lambda x: energy_data.append(float(x)))
    sim.observe.particle_positions(
        stride=observe_stride,
        callback=lambda x: position_data.append(
            [[p[0], p[1], p[2]] for p in x]))

    sim.run(n_steps, timestep=timestep)
    runtime = time.perf_counter() - t0

    n_points = len(energy_data)
    times = [i * observe_stride * timestep for i in range(n_points)]
    counts = {}
    for i, sp in enumerate(species_order):
        counts[sp] = [int(c[i]) for c in count_data]

    traj_data = {
        'times': times,
        'counts': counts,
        'energy': list(energy_data),
    }

    pos_snapshots = []
    for idx, positions_list in enumerate(position_data):
        real_time = idx * observe_stride * timestep
        pos_snapshots.append({
            'time': round(real_time, 6),
            'positions': positions_list,
            'bonds': [],
        })

    return traj_data, pos_snapshots, runtime


# ── Simulation Configs (metadata for report) ──────────────────────

SIM_CONFIGS = [
    {
        'id': 'actin',
        'title': 'Actin Filament Treadmilling',
        'subtitle': 'Topology-based polymerization and structural depolymerization',
        'description': (
            'An actin-like filament grows at its barbed (head) end by recruiting '
            'diffusing substrate monomers via a spatial topology reaction, while '
            'simultaneously losing monomers at the pointed (tail) end through a '
            'structural reaction. This treadmilling produces a steady-state '
            'filament length that balances polymerization and depolymerization rates. '
            'Topological bonds enforce linear chain geometry with angular rigidity.'
        ),
        'box_size': [20., 20., 20.],
        'species_list': ['head', 'core', 'tail', 'substrate'],
        'species_colors': {
            'head': '#818cf8', 'core': '#6366f1',
            'tail': '#a78bfa', 'substrate': '#c7d2fe',
        },
        'chart_species': ['head', 'core', 'tail', 'substrate', 'filament_length'],
        'chart_colors': {
            'head': '#818cf8', 'core': '#6366f1', 'tail': '#a78bfa',
            'substrate': '#c7d2fe', 'filament_length': '#4f46e5',
        },
        'n_steps': 30000,
        'timestep': 0.005,
        'observe_stride': 100,
        'camera': [24, 16, 24],
        'color_scheme': 'indigo',
        'uses_topologies': True,
        'reactions_display': [
            'attach: filament(head) + substrate -> filament(core--head)',
            'depolymerize: structural (tail end release)',
        ],
        'runner': run_simulation_actin,
    },
    {
        'id': 'lotka_volterra',
        'title': 'Lotka-Volterra Predator-Prey',
        'subtitle': 'Predator-prey dynamics with spatial stochasticity',
        'description': (
            'A spatial Lotka-Volterra predator-prey model: prey (A) reproduce '
            'by fission, predators (B) consume prey upon contact via enzymatic '
            'reaction (A + B -> B + B), and predators spontaneously decay. '
            'Spatial diffusion and stochastic reactions introduce noise and '
            'fluctuations around the classical oscillatory dynamics. This uses '
            'the ReaDDyProcess PBG wrapper for configuration-driven setup.'
        ),
        'box_size': [15., 15., 15.],
        'species_list': ['A', 'B'],
        'species_colors': {'A': '#10b981', 'B': '#f59e0b'},
        'chart_species': ['A', 'B'],
        'chart_colors': {'A': '#10b981', 'B': '#f59e0b'},
        'n_steps': 10000,
        'timestep': 0.005,
        'observe_stride': 50,
        'camera': [18, 12, 18],
        'color_scheme': 'emerald',
        'uses_topologies': False,
        'reactions_display': [
            'reproduce: A -> A + A',
            'eat: A + B -> B + B (enzymatic)',
            'death: B ->',
        ],
        'config': {
            'box_size': (15., 15., 15.),
            'species': {'A': 1.5, 'B': 1.5},
            'reactions': [
                {'descriptor': 'reproduce: A -> A +(2) A', 'rate': 0.3},
                {'method': 'enzymatic', 'name': 'eat', 'catalyst': 'B',
                 'type_from': 'A', 'type_to': 'B', 'rate': 0.8,
                 'educt_distance': 3.0},
                {'descriptor': 'death: B ->', 'rate': 0.4},
            ],
            'potentials': [
                {'type': 'harmonic_repulsion', 'species1': 'A',
                 'species2': 'A', 'force_constant': 5.,
                 'interaction_distance': 1.0},
                {'type': 'harmonic_repulsion', 'species1': 'B',
                 'species2': 'B', 'force_constant': 5.,
                 'interaction_distance': 1.0},
                {'type': 'harmonic_repulsion', 'species1': 'A',
                 'species2': 'B', 'force_constant': 5.,
                 'interaction_distance': 1.0},
            ],
            'initial_particles': {
                'A': _random_positions(30, 6.0, seed=100),
                'B': _random_positions(10, 6.0, seed=200),
            },
            'timestep': 0.005,
            'observe_stride': 50,
        },
        'runner': run_simulation_lotka_volterra,
    },
    {
        'id': 'living_polymers',
        'title': 'Living Polymer Equilibrium',
        'subtitle': 'Reversible polymerization with association and dissociation',
        'description': (
            'Two hundred short polymer chains (Head-Tail-Tail-Head) undergo '
            'reversible association: chain heads merge via spatial topology '
            'reactions, while a structural reaction randomly breaks internal '
            'bonds. Over time the system reaches a dynamic equilibrium between '
            'chain growth and fragmentation, producing an exponential-like '
            'chain length distribution characteristic of living polymers.'
        ),
        'box_size': [40., 40., 40.],
        'species_list': ['Head', 'Tail'],
        'species_colors': {'Head': '#f59e0b', 'Tail': '#6366f1'},
        'chart_species': ['Head', 'Tail', 'avg_chain_length'],
        'chart_colors': {
            'Head': '#f59e0b', 'Tail': '#6366f1',
            'avg_chain_length': '#d97706',
        },
        'n_steps': 30000,
        'timestep': 0.5,
        'observe_stride': 100,
        'camera': [50, 35, 50],
        'color_scheme': 'amber',
        'uses_topologies': True,
        'reactions_display': [
            'associate: Polymer(Head) + Polymer(Head) -> Polymer(Tail--Tail)',
            'dissociate: structural (random bond break)',
        ],
        'runner': run_simulation_living_polymers,
    },
    {
        'id': 'crowded_sphere',
        'title': 'Crowded Diffusion with Spherical Confinement',
        'subtitle': 'Dense particle packing inside a confining sphere',
        'description': (
            'Eighty particles with strong excluded-volume repulsion diffuse '
            'inside a spherical confining potential (radius 7), reaching a '
            'disordered equilibrium packing. There are no reactions -- this '
            'tests pure Brownian dynamics with pair potentials and a confining '
            'sphere_in potential. The energy decreases as particles spread '
            'apart to minimize repulsive overlap within the spherical boundary.'
        ),
        'box_size': [20., 20., 20.],
        'species_list': ['P'],
        'species_colors': {'P': '#f43f5e'},
        'chart_species': ['P'],
        'chart_colors': {'P': '#f43f5e'},
        'n_steps': 10000,
        'timestep': 0.002,
        'observe_stride': 50,
        'camera': [12, 8, 12],
        'color_scheme': 'rose',
        'uses_topologies': False,
        'reactions_display': [],
        'runner': run_simulation_crowded_sphere,
    },
]


# ── Bigraph Image ───────────────────────────────────────────────────

def generate_bigraph_image(cfg):
    """Generate a colored bigraph-viz PNG for the simulation."""
    from bigraph_viz import plot_bigraph

    if cfg.get('uses_topologies'):
        # Simplified diagram for topology-based sims
        doc = {
            'readdy': {
                '_type': 'process',
                'address': 'local:ReaDDy (topologies)',
                'outputs': {
                    'particle_counts': ['stores', 'particle_counts'],
                    'positions': ['stores', 'positions'],
                    'energy': ['stores', 'energy'],
                },
            },
            'stores': {},
            'emitter': {
                '_type': 'step',
                'address': 'local:ram-emitter',
                'inputs': {
                    'energy': ['stores', 'energy'],
                    'time': ['global_time'],
                },
            },
        }
    else:
        doc = {
            'readdy': {
                '_type': 'process',
                'address': 'local:ReaDDyProcess',
                'outputs': {
                    'particle_counts': ['stores', 'particle_counts'],
                    'total_particles': ['stores', 'total_particles'],
                    'energy': ['stores', 'energy'],
                    'positions': ['stores', 'positions'],
                },
            },
            'stores': {},
            'emitter': {
                '_type': 'step',
                'address': 'local:ram-emitter',
                'inputs': {
                    'total_particles': ['stores', 'total_particles'],
                    'energy': ['stores', 'energy'],
                    'time': ['global_time'],
                },
            },
        }

    node_colors = {
        ('readdy',): '#6366f1',
        ('emitter',): '#8b5cf6',
        ('stores',): '#e0e7ff',
    }

    outdir = tempfile.mkdtemp()
    plot_bigraph(
        state=doc,
        out_dir=outdir,
        filename='bigraph',
        file_format='png',
        remove_process_place_edges=True,
        rankdir='LR',
        node_fill_colors=node_colors,
        node_label_size='16pt',
        port_labels=False,
        dpi='150',
    )
    png_path = os.path.join(outdir, 'bigraph.png')
    with open(png_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    return f'data:image/png;base64,{b64}'


def build_pbg_document(cfg):
    """Build the PBG composite document dict for display."""
    if cfg.get('uses_topologies'):
        # Build a representative document for topology sims
        species_dict = {sp: 0.0 for sp in cfg['species_list']}
        return {
            'readdy_topologies': {
                '_type': 'process',
                'address': 'local:ReaDDy (direct topology API)',
                'config': {
                    'box_size': cfg['box_size'],
                    'species': species_dict,
                    'topology_types': ['filament'] if cfg['id'] == 'actin' else ['Polymer'],
                    'n_steps': cfg['n_steps'],
                    'timestep': cfg['timestep'],
                    'observe_stride': cfg['observe_stride'],
                    'note': 'Structural reactions require Python callables; not PBG-configurable',
                },
                'outputs': {
                    'particle_counts': ['stores', 'particle_counts'],
                    'positions': ['stores', 'positions'],
                    'energy': ['stores', 'energy'],
                },
            },
            'stores': {},
        }
    else:
        wrapper_config = cfg.get('config', {})
        return make_readdy_document(
            box_size=list(wrapper_config.get('box_size', cfg['box_size'])),
            species=wrapper_config.get('species', {}),
            reactions=wrapper_config.get('reactions', []),
            potentials=wrapper_config.get('potentials', []),
            initial_particles={k: f'[{len(v)} positions]'
                               for k, v in wrapper_config.get(
                                   'initial_particles', {}).items()},
            timestep=wrapper_config.get('timestep', cfg['timestep']),
            observe_stride=wrapper_config.get('observe_stride',
                                              cfg['observe_stride']),
            interval=cfg['n_steps'] * cfg['timestep'],
        )


# ── Color Schemes ──────────────────────────────────────────────────

COLOR_SCHEMES = {
    'indigo': {'primary': '#6366f1', 'light': '#e0e7ff', 'dark': '#4338ca',
               'bg': '#eef2ff', 'accent': '#818cf8', 'text': '#312e81'},
    'emerald': {'primary': '#10b981', 'light': '#d1fae5', 'dark': '#059669',
                'bg': '#ecfdf5', 'accent': '#34d399', 'text': '#064e3b'},
    'amber': {'primary': '#f59e0b', 'light': '#fef3c7', 'dark': '#d97706',
              'bg': '#fffbeb', 'accent': '#fbbf24', 'text': '#78350f'},
    'rose': {'primary': '#f43f5e', 'light': '#ffe4e6', 'dark': '#e11d48',
             'bg': '#fff1f2', 'accent': '#fb7185', 'text': '#881337'},
}


# ── HTML Report Generator ──────────────────────────────────────────

def generate_html(sim_results, output_path):
    """Generate comprehensive HTML report with 4 simulation sections."""
    sections_html = []
    all_js_data = {}

    for idx, (cfg, (traj_data, pos_snapshots, runtime)) in enumerate(sim_results):
        sid = cfg['id']
        cs = COLOR_SCHEMES[cfg['color_scheme']]
        species_list = cfg['species_list']

        # Counts
        final_counts = {}
        for sp in species_list:
            series = traj_data['counts'].get(sp, [0])
            final_counts[sp] = series[-1] if series else 0
        final_total = sum(final_counts.values())

        # Initial total (approximate from first data point)
        initial_counts = {}
        for sp in species_list:
            series = traj_data['counts'].get(sp, [0])
            initial_counts[sp] = series[0] if series else 0
        initial_total = sum(initial_counts.values())

        # Energy
        energies = traj_data['energy']
        e_min = min(energies) if energies else 0
        e_max = max(energies) if energies else 0

        # Times
        times = traj_data['times']
        total_time = times[-1] if times else 0

        # Downsample position snapshots
        vis_snaps = _downsample_snapshots(pos_snapshots, max_snaps=50)

        # JS data
        all_js_data[sid] = {
            'snapshots': vis_snaps,
            'camera': cfg['camera'],
            'box_size': cfg['box_size'],
            'species_colors': cfg['species_colors'],
            'chart_species': cfg.get('chart_species', list(cfg['species_colors'].keys())),
            'chart_colors': cfg.get('chart_colors', cfg['species_colors']),
            'charts': {
                'times': times,
                'energy': energies,
                'counts': traj_data['counts'],
            },
        }

        # Bigraph image
        print(f'  Generating bigraph diagram for {sid}...')
        bigraph_img = generate_bigraph_image(cfg)

        # PBG document
        pbg_doc = build_pbg_document(cfg)

        # Build counts display
        counts_str = ', '.join(
            f'{sp}: {final_counts[sp]}' for sp in species_list)

        # Reaction descriptors for display
        rxn_strs = cfg.get('reactions_display', [])
        rxn_display = '; '.join(rxn_strs) if rxn_strs else 'None'

        # Topology badge
        topology_badge = ''
        if cfg.get('uses_topologies'):
            topology_badge = (
                f'<span style="display:inline-block; background:{cs["light"]}; '
                f'color:{cs["dark"]}; font-size:.65rem; font-weight:700; '
                f'padding:.15rem .5rem; border-radius:6px; margin-left:.5rem; '
                f'vertical-align:middle;">TOPOLOGIES</span>'
            )

        section = f"""
    <div class="sim-section" id="sim-{sid}">
      <div class="sim-header" style="border-left: 4px solid {cs['primary']};">
        <div class="sim-number" style="background:{cs['light']}; color:{cs['dark']};">{idx+1}</div>
        <div>
          <h2 class="sim-title">{cfg['title']}{topology_badge}</h2>
          <p class="sim-subtitle">{cfg['subtitle']}</p>
        </div>
      </div>
      <p class="sim-description">{cfg['description']}</p>

      <div class="metrics-row">
        <div class="metric"><span class="metric-label">Initial</span><span class="metric-value">{initial_total}</span><span class="metric-sub">particles</span></div>
        <div class="metric"><span class="metric-label">Final</span><span class="metric-value">{final_total}</span><span class="metric-sub">{counts_str}</span></div>
        <div class="metric"><span class="metric-label">Species</span><span class="metric-value">{len(species_list)}</span></div>
        <div class="metric"><span class="metric-label">Reactions</span><span class="metric-value">{len(rxn_strs)}</span><span class="metric-sub" title="{rxn_display}">{rxn_display[:40]}{'...' if len(rxn_display) > 40 else ''}</span></div>
        <div class="metric"><span class="metric-label">Time</span><span class="metric-value">{total_time:.1f}</span><span class="metric-sub">sim. units</span></div>
        <div class="metric"><span class="metric-label">Steps</span><span class="metric-value">{cfg['n_steps']:,}</span></div>
        <div class="metric"><span class="metric-label">Runtime</span><span class="metric-value">{runtime:.1f}s</span></div>
      </div>

      <h3 class="subsection-title">3D Particle Viewer</h3>
      <div class="viewer-wrap">
        <canvas id="canvas-{sid}" class="mesh-canvas"></canvas>
        <div class="viewer-info">
          <strong>{final_total}</strong> particles &middot; Box: {cfg['box_size'][0]}&times;{cfg['box_size'][1]}&times;{cfg['box_size'][2]}<br>
          Drag to rotate &middot; Scroll to zoom
        </div>
        <div class="legend-box" id="legend-{sid}"></div>
        <div class="slider-controls">
          <button class="play-btn" style="border-color:{cs['primary']}; color:{cs['primary']};" onclick="togglePlay('{sid}')">Play</button>
          <label>Time</label>
          <input type="range" class="time-slider" id="slider-{sid}" min="0" max="{len(vis_snaps)-1}" value="0" step="1"
                 style="accent-color:{cs['primary']};">
          <span class="time-val" id="tval-{sid}">t = 0</span>
        </div>
      </div>

      <h3 class="subsection-title">Population &amp; Energy Dynamics</h3>
      <div class="charts-row">
        <div class="chart-box"><div id="chart-counts-{sid}" class="chart"></div></div>
        <div class="chart-box"><div id="chart-energy-{sid}" class="chart"></div></div>
      </div>

      <div class="pbg-row">
        <div class="pbg-col">
          <h3 class="subsection-title">Bigraph Architecture</h3>
          <div class="bigraph-img-wrap">
            <img src="{bigraph_img}" alt="Bigraph architecture diagram">
          </div>
        </div>
        <div class="pbg-col">
          <h3 class="subsection-title">Composite Document</h3>
          <div class="json-tree" id="json-{sid}"></div>
        </div>
      </div>
    </div>
"""
        sections_html.append(section)

    # Navigation
    nav_items = ''.join(
        f'<a href="#sim-{c["id"]}" class="nav-link" '
        f'style="border-color:{COLOR_SCHEMES[c["color_scheme"]]["primary"]};">'
        f'{c["title"]}</a>'
        for c in [r[0] for r in sim_results])

    # PBG docs for JSON viewer
    pbg_docs = {r[0]['id']: build_pbg_document(r[0]) for r in sim_results}

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ReaDDy Reaction-Diffusion Simulation Report</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
       background:#fff; color:#1e293b; line-height:1.6; }}
.page-header {{
  background:linear-gradient(135deg,#f8fafc 0%,#eef2ff 30%,#fffbeb 60%,#fdf2f8 100%);
  border-bottom:1px solid #e2e8f0; padding:3rem;
}}
.page-header h1 {{ font-size:2.2rem; font-weight:800; color:#0f172a; margin-bottom:.3rem; }}
.page-header p {{ color:#64748b; font-size:.95rem; max-width:700px; }}
.nav {{ display:flex; gap:.8rem; padding:1rem 3rem; background:#f8fafc;
        border-bottom:1px solid #e2e8f0; position:sticky; top:0; z-index:100;
        flex-wrap:wrap; }}
.nav-link {{ padding:.4rem 1rem; border-radius:8px; border:1.5px solid;
             text-decoration:none; font-size:.85rem; font-weight:600;
             transition:all .15s; }}
.nav-link:hover {{ transform:translateY(-1px); box-shadow:0 2px 8px rgba(0,0,0,.08); }}
.sim-section {{ padding:2.5rem 3rem; border-bottom:1px solid #e2e8f0; }}
.sim-header {{ display:flex; align-items:center; gap:1rem; margin-bottom:.8rem;
               padding-left:1rem; }}
.sim-number {{ width:36px; height:36px; border-radius:10px; display:flex;
               align-items:center; justify-content:center; font-weight:800; font-size:1.1rem; }}
.sim-title {{ font-size:1.5rem; font-weight:700; color:#0f172a; }}
.sim-subtitle {{ font-size:.9rem; color:#64748b; }}
.sim-description {{ color:#475569; font-size:.9rem; margin-bottom:1.5rem; max-width:800px; }}
.subsection-title {{ font-size:1.05rem; font-weight:600; color:#334155;
                     margin:1.5rem 0 .8rem; }}
.metrics-row {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr));
                gap:.8rem; margin-bottom:1.5rem; }}
.metric {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
           padding:.8rem; text-align:center; }}
.metric-label {{ display:block; font-size:.7rem; text-transform:uppercase;
                 letter-spacing:.06em; color:#94a3b8; margin-bottom:.2rem; }}
.metric-value {{ display:block; font-size:1.3rem; font-weight:700; color:#1e293b; }}
.metric-sub {{ display:block; font-size:.65rem; color:#94a3b8; overflow:hidden;
               text-overflow:ellipsis; white-space:nowrap; }}
.viewer-wrap {{ position:relative; background:#0f172a; border:1px solid #e2e8f0;
                border-radius:14px; overflow:hidden; margin-bottom:1rem; }}
.mesh-canvas {{ width:100%; height:500px; display:block; cursor:grab; }}
.mesh-canvas:active {{ cursor:grabbing; }}
.viewer-info {{ position:absolute; top:.8rem; left:.8rem; background:rgba(15,23,42,.85);
                border:1px solid rgba(255,255,255,.1); border-radius:8px; padding:.5rem .8rem;
                font-size:.75rem; color:#94a3b8; backdrop-filter:blur(4px); }}
.viewer-info strong {{ color:#e2e8f0; }}
.legend-box {{ position:absolute; top:.8rem; right:.8rem; background:rgba(15,23,42,.85);
               border:1px solid rgba(255,255,255,.1); border-radius:8px; padding:.6rem .8rem;
               backdrop-filter:blur(4px); }}
.legend-item {{ display:flex; align-items:center; gap:.4rem; margin-bottom:.2rem; }}
.legend-dot {{ width:10px; height:10px; border-radius:50%; }}
.legend-label {{ font-size:.7rem; color:#e2e8f0; font-weight:500; }}
.slider-controls {{ position:absolute; bottom:0; left:0; right:0;
                    background:linear-gradient(transparent,rgba(15,23,42,.95));
                    padding:1.5rem 1.5rem 1rem; display:flex; align-items:center; gap:.8rem; }}
.slider-controls label {{ font-size:.8rem; color:#94a3b8; }}
.time-slider {{ flex:1; height:5px; }}
.time-val {{ font-size:.95rem; font-weight:600; color:#e2e8f0; min-width:100px; text-align:right; }}
.play-btn {{ background:rgba(15,23,42,.8); border:1.5px solid; padding:.3rem .8rem; border-radius:7px;
             cursor:pointer; font-size:.8rem; font-weight:600; transition:all .15s; }}
.play-btn:hover {{ transform:scale(1.05); }}
.charts-row {{ display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-bottom:1rem; }}
.chart-box {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; overflow:hidden; }}
.chart {{ height:300px; }}
.pbg-row {{ display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; margin-top:1rem; }}
.pbg-col {{ min-width:0; }}
.bigraph-img-wrap {{ background:#fafafa; border:1px solid #e2e8f0; border-radius:10px;
                     padding:1.5rem; text-align:center; }}
.bigraph-img-wrap img {{ max-width:100%; height:auto; }}
.json-tree {{ background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
              padding:1rem; max-height:500px; overflow-y:auto; font-family:'SF Mono',
              Menlo,Monaco,'Courier New',monospace; font-size:.78rem; line-height:1.5; }}
.jt-key {{ color:#7c3aed; font-weight:600; }}
.jt-str {{ color:#059669; }}
.jt-num {{ color:#2563eb; }}
.jt-bool {{ color:#d97706; }}
.jt-null {{ color:#94a3b8; }}
.jt-toggle {{ cursor:pointer; user-select:none; color:#94a3b8; margin-right:.3rem; }}
.jt-toggle:hover {{ color:#1e293b; }}
.jt-collapsed {{ display:none; }}
.jt-bracket {{ color:#64748b; }}
.footer {{ text-align:center; padding:2rem; color:#94a3b8; font-size:.8rem;
           border-top:1px solid #e2e8f0; }}
@media(max-width:900px) {{
  .charts-row,.pbg-row {{ grid-template-columns:1fr; }}
  .sim-section,.page-header {{ padding:1.5rem; }}
  .nav {{ padding:1rem 1.5rem; }}
}}
</style>
</head>
<body>

<div class="page-header">
  <h1>ReaDDy Reaction-Diffusion Simulation Report</h1>
  <p>Four particle-based reaction-diffusion simulations showcasing
  <strong>process-bigraph</strong> integration with ReaDDy's Brownian dynamics
  engine. Configurations range from topology-based actin treadmilling and
  living polymer equilibria to predator-prey oscillations and confined diffusion.</p>
</div>

<div class="nav">{nav_items}</div>

{''.join(sections_html)}

<div class="footer">
  Generated by <strong>pbg-readdy</strong> &mdash;
  ReaDDy + process-bigraph &mdash;
  Particle-Based Reaction-Diffusion Dynamics
</div>

<script>
const DATA = {json.dumps(all_js_data)};
const DOCS = {json.dumps(pbg_docs, indent=2)};

// --- JSON Tree Viewer ---
function renderJson(obj, depth) {{
  if (depth === undefined) depth = 0;
  if (obj === null) return '<span class="jt-null">null</span>';
  if (typeof obj === 'boolean') return '<span class="jt-bool">' + obj + '</span>';
  if (typeof obj === 'number') return '<span class="jt-num">' + obj + '</span>';
  if (typeof obj === 'string') return '<span class="jt-str">"' + obj.replace(/</g,'&lt;') + '"</span>';
  if (Array.isArray(obj)) {{
    if (obj.length === 0) return '<span class="jt-bracket">[]</span>';
    if (obj.length <= 5 && obj.every(x => typeof x !== 'object' || x === null)) {{
      const items = obj.map(x => renderJson(x, depth+1)).join(', ');
      return '<span class="jt-bracket">[</span>' + items + '<span class="jt-bracket">]</span>';
    }}
    const id = 'jt' + Math.random().toString(36).slice(2,9);
    let html = '<span class="jt-toggle" onclick="toggleJt(\\'' + id + '\\')">&blacktriangledown;</span>';
    html += '<span class="jt-bracket">[</span> <span style="color:#94a3b8;font-size:.7rem;">' + obj.length + ' items</span>';
    html += '<div id="' + id + '" style="margin-left:1.2rem;">';
    obj.forEach((v, i) => {{ html += '<div>' + renderJson(v, depth+1) + (i < obj.length-1 ? ',' : '') + '</div>'; }});
    html += '</div><span class="jt-bracket">]</span>';
    return html;
  }}
  if (typeof obj === 'object') {{
    const keys = Object.keys(obj);
    if (keys.length === 0) return '<span class="jt-bracket">{{}}</span>';
    const id = 'jt' + Math.random().toString(36).slice(2,9);
    const collapsed = depth >= 2;
    let html = '<span class="jt-toggle" onclick="toggleJt(\\'' + id + '\\')">' +
               (collapsed ? '&blacktriangleright;' : '&blacktriangledown;') + '</span>';
    html += '<span class="jt-bracket">{{</span>';
    html += '<div id="' + id + '"' + (collapsed ? ' class="jt-collapsed"' : '') + ' style="margin-left:1.2rem;">';
    keys.forEach((k, i) => {{
      html += '<div><span class="jt-key">' + k + '</span>: ' +
              renderJson(obj[k], depth+1) + (i < keys.length-1 ? ',' : '') + '</div>';
    }});
    html += '</div><span class="jt-bracket">}}</span>';
    return html;
  }}
  return String(obj);
}}
function toggleJt(id) {{
  const el = document.getElementById(id);
  if (el.classList.contains('jt-collapsed')) {{
    el.classList.remove('jt-collapsed');
    const prev = el.previousElementSibling;
    if (prev && prev.previousElementSibling && prev.previousElementSibling.classList.contains('jt-toggle'))
      prev.previousElementSibling.innerHTML = '&blacktriangledown;';
  }} else {{
    el.classList.add('jt-collapsed');
    const prev = el.previousElementSibling;
    if (prev && prev.previousElementSibling && prev.previousElementSibling.classList.contains('jt-toggle'))
      prev.previousElementSibling.innerHTML = '&blacktriangleright;';
  }}
}}
Object.keys(DOCS).forEach(sid => {{
  const el = document.getElementById('json-' + sid);
  if (el) el.innerHTML = renderJson(DOCS[sid], 0);
}});

// --- Three.js Particle Viewers ---
const viewers = {{}};
const playStates = {{}};

function hexToRgb(hex) {{
  const r = parseInt(hex.slice(1,3), 16) / 255;
  const g = parseInt(hex.slice(3,5), 16) / 255;
  const b = parseInt(hex.slice(5,7), 16) / 255;
  return [r, g, b];
}}

function initViewer(sid) {{
  const d = DATA[sid];
  const canvas = document.getElementById('canvas-' + sid);
  const W = canvas.parentElement.clientWidth;
  const H = 500;
  canvas.width = W * window.devicePixelRatio;
  canvas.height = H * window.devicePixelRatio;
  canvas.style.width = W + 'px';
  canvas.style.height = H + 'px';

  const renderer = new THREE.WebGLRenderer({{canvas, antialias:true}});
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(W, H);
  renderer.setClearColor(0x0f172a);

  const scene = new THREE.Scene();
  const cam = new THREE.PerspectiveCamera(45, W/H, 0.1, 500);
  cam.position.set(...d.camera);

  const controls = new THREE.OrbitControls(cam, canvas);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 0.5;

  scene.add(new THREE.AmbientLight(0xffffff, 0.4));
  const dl1 = new THREE.DirectionalLight(0xffffff, 0.8);
  dl1.position.set(5,8,5); scene.add(dl1);
  const dl2 = new THREE.DirectionalLight(0x94a3b8, 0.3);
  dl2.position.set(-5,-3,-5); scene.add(dl2);

  // Box wireframe
  const bs = d.box_size;
  const boxGeo = new THREE.BoxGeometry(bs[0], bs[1], bs[2]);
  const boxEdges = new THREE.EdgesGeometry(boxGeo);
  const boxLine = new THREE.LineSegments(boxEdges,
    new THREE.LineBasicMaterial({{color:0x334155, transparent:true, opacity:0.4}}));
  scene.add(boxLine);

  // Species color map (only physical species, not derived quantities)
  const speciesColors = d.species_colors;
  const speciesNames = Object.keys(speciesColors);

  // Create sphere geometry (shared)
  const sphereGeo = new THREE.SphereGeometry(0.35, 12, 8);

  // Instanced meshes per species
  const meshes = {{}};
  const maxParticles = 500;
  speciesNames.forEach(sp => {{
    const color = new THREE.Color(speciesColors[sp]);
    const mat = new THREE.MeshPhongMaterial({{
      color, shininess:60, specular:0x444444
    }});
    const im = new THREE.InstancedMesh(sphereGeo, mat, maxParticles);
    im.count = 0;
    scene.add(im);
    meshes[sp] = im;
  }});

  // Bond lines
  const maxBonds = 2000;
  const bondPositions = new Float32Array(maxBonds * 6);
  const bondGeo = new THREE.BufferGeometry();
  bondGeo.setAttribute('position', new THREE.BufferAttribute(bondPositions, 3));
  const bondMat = new THREE.LineBasicMaterial({{
    color: 0xffffff, transparent: true, opacity: 0.6, linewidth: 1
  }});
  const bondLines = new THREE.LineSegments(bondGeo, bondMat);
  scene.add(bondLines);

  // Build legend
  const legendEl = document.getElementById('legend-' + sid);
  speciesNames.forEach(sp => {{
    const item = document.createElement('div');
    item.className = 'legend-item';
    item.innerHTML = '<div class="legend-dot" style="background:' + speciesColors[sp] + ';"></div>' +
                     '<span class="legend-label">' + sp + '</span>';
    legendEl.appendChild(item);
  }});

  const dummy = new THREE.Object3D();

  function updateParticles(snapIdx) {{
    const snap = d.snapshots[snapIdx];
    if (!snap) return;
    const positions = snap.positions;

    const chartCounts = d.charts.counts;
    const chartTimes = d.charts.times;

    // Find closest chart timepoint to this snapshot
    const snapTime = snap.time;
    let closestIdx = 0;
    let minDiff = Math.abs(chartTimes[0] - snapTime);
    for (let i = 1; i < chartTimes.length; i++) {{
      const diff = Math.abs(chartTimes[i] - snapTime);
      if (diff < minDiff) {{ minDiff = diff; closestIdx = i; }}
    }}

    // Get counts at this time and assign positions to species
    let offset = 0;
    speciesNames.forEach(sp => {{
      const count = chartCounts[sp] ? chartCounts[sp][closestIdx] : 0;
      const mesh = meshes[sp];
      mesh.count = Math.min(count, maxParticles);
      for (let i = 0; i < mesh.count; i++) {{
        const pi = offset + i;
        if (pi < positions.length) {{
          dummy.position.set(positions[pi][0], positions[pi][1], positions[pi][2]);
          dummy.updateMatrix();
          mesh.setMatrixAt(i, dummy.matrix);
        }}
      }}
      mesh.instanceMatrix.needsUpdate = true;
      offset += count;
    }});

    // Handle remaining particles (assign to last species)
    if (offset < positions.length && speciesNames.length > 0) {{
      const lastSp = speciesNames[speciesNames.length - 1];
      const mesh = meshes[lastSp];
      const extra = positions.length - offset;
      const newCount = Math.min(mesh.count + extra, maxParticles);
      for (let i = mesh.count; i < newCount; i++) {{
        const pi = offset + (i - mesh.count);
        if (pi < positions.length) {{
          dummy.position.set(positions[pi][0], positions[pi][1], positions[pi][2]);
          dummy.updateMatrix();
          mesh.setMatrixAt(i, dummy.matrix);
        }}
      }}
      mesh.count = newCount;
      mesh.instanceMatrix.needsUpdate = true;
    }}

    // Update bonds
    const bonds = snap.bonds || [];
    const nBonds = Math.min(bonds.length, maxBonds);
    for (let i = 0; i < nBonds; i++) {{
      const [a, b] = bonds[i];
      if (a < positions.length && b < positions.length) {{
        bondPositions[i*6]   = positions[a][0];
        bondPositions[i*6+1] = positions[a][1];
        bondPositions[i*6+2] = positions[a][2];
        bondPositions[i*6+3] = positions[b][0];
        bondPositions[i*6+4] = positions[b][1];
        bondPositions[i*6+5] = positions[b][2];
      }}
    }}
    for (let i = nBonds * 6; i < maxBonds * 6; i++) bondPositions[i] = 0;
    bondGeo.attributes.position.needsUpdate = true;
    bondGeo.setDrawRange(0, nBonds * 2);
  }}

  updateParticles(0);

  const slider = document.getElementById('slider-' + sid);
  const tval = document.getElementById('tval-' + sid);
  slider.addEventListener('input', () => {{
    const idx = parseInt(slider.value);
    updateParticles(idx);
    const snap = d.snapshots[idx];
    tval.textContent = snap ? 't = ' + snap.time.toFixed(2) : 't = 0';
  }});

  viewers[sid] = {{ renderer, scene, cam, controls, updateParticles, slider, tval }};
  playStates[sid] = {{ playing: false, interval: null }};

  function animate() {{
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, cam);
  }}
  animate();
}}

function togglePlay(sid) {{
  const ps = playStates[sid];
  const v = viewers[sid];
  const d = DATA[sid];
  const btn = event.target;
  ps.playing = !ps.playing;
  if (ps.playing) {{
    btn.textContent = 'Pause';
    v.controls.autoRotate = false;
    ps.interval = setInterval(() => {{
      let idx = parseInt(v.slider.value) + 1;
      if (idx >= d.snapshots.length) idx = 0;
      v.slider.value = idx;
      v.updateParticles(idx);
      const snap = d.snapshots[idx];
      v.tval.textContent = snap ? 't = ' + snap.time.toFixed(2) : 't = 0';
    }}, 200);
  }} else {{
    btn.textContent = 'Play';
    v.controls.autoRotate = true;
    clearInterval(ps.interval);
  }}
}}

// Init all viewers
Object.keys(DATA).forEach(sid => initViewer(sid));

// --- Plotly Charts ---
const pLayout = {{
  paper_bgcolor:'#f8fafc', plot_bgcolor:'#f8fafc',
  font:{{ color:'#64748b', family:'-apple-system,sans-serif', size:11 }},
  margin:{{ l:55, r:15, t:35, b:45 }},
  xaxis:{{ gridcolor:'#e2e8f0', zerolinecolor:'#e2e8f0',
           title:{{ text:'Time', font:{{ size:10 }} }} }},
  yaxis:{{ gridcolor:'#e2e8f0', zerolinecolor:'#e2e8f0' }},
}};
const pCfg = {{ responsive:true, displayModeBar:false }};

const chartColorsFallback = ['#6366f1','#10b981','#f43f5e','#f59e0b','#8b5cf6','#06b6d4'];

Object.keys(DATA).forEach(sid => {{
  const c = DATA[sid].charts;
  const chartSpecies = DATA[sid].chart_species;
  const chartColors = DATA[sid].chart_colors;

  // Population / dynamics chart
  const countTraces = chartSpecies.map((sp, i) => ({{
    x: c.times, y: c.counts[sp], type:'scatter', mode:'lines',
    line:{{ color: chartColors[sp] || chartColorsFallback[i % chartColorsFallback.length], width:2,
            dash: (sp === 'filament_length' || sp === 'avg_chain_length') ? 'dash' : 'solid' }},
    name: sp,
  }}));

  const countTitle = chartSpecies.some(s => s === 'filament_length')
    ? 'Particle Counts & Filament Length'
    : chartSpecies.some(s => s === 'avg_chain_length')
      ? 'Particle Counts & Avg Chain Length'
      : 'Particle Counts';

  Plotly.newPlot('chart-counts-'+sid, countTraces, {{
    ...pLayout,
    title:{{ text: countTitle, font:{{ size:12, color:'#334155' }} }},
    yaxis:{{...pLayout.yaxis, title:{{ text:'Count', font:{{ size:10 }} }} }},
    legend:{{ font:{{ size:10 }}, bgcolor:'rgba(0,0,0,0)' }},
    showlegend: true,
  }}, pCfg);

  // Energy chart
  Plotly.newPlot('chart-energy-'+sid, [{{
    x:c.times, y:c.energy, type:'scatter', mode:'lines',
    line:{{ color:'#f43f5e', width:2 }},
    fill:'tozeroy', fillcolor:'rgba(244,63,94,0.06)',
  }}], {{
    ...pLayout,
    title:{{ text:'Potential Energy', font:{{ size:12, color:'#334155' }} }},
    yaxis:{{...pLayout.yaxis, title:{{ text:'Energy', font:{{ size:10 }} }} }},
    showlegend: false,
  }}, pCfg);
}});

</script>
</body>
</html>"""

    with open(output_path, 'w') as f:
        f.write(html)
    print(f'Report saved to {output_path}')


# ── Main ────────────────────────────────────────────────────────────

def run_demo():
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(demo_dir, 'report.html')

    sim_results = []
    for cfg in SIM_CONFIGS:
        print(f'Running: {cfg["title"]}...')
        runner = cfg['runner']
        traj_data, pos_snapshots, runtime = runner()
        sim_results.append((cfg, (traj_data, pos_snapshots, runtime)))
        print(f'  Runtime: {runtime:.2f}s')
        print(f'  {len(traj_data["times"])} time points, '
              f'{len(pos_snapshots)} position snapshots')

    print('Generating HTML report...')
    generate_html(sim_results, output_path)

    # Open in Safari
    import subprocess
    subprocess.run(['open', '-a', 'Safari', output_path])


if __name__ == '__main__':
    run_demo()
