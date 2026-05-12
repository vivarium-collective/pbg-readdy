"""ReaDDy Process wrapper for process-bigraph.

Wraps the ReaDDy particle-based reaction-diffusion simulator as a
time-driven Process using the bridge pattern. The internal ReaDDy
system and simulation are lazily initialized on first update() call.
Uses callback-based observables so update() can be called multiple times.
"""

import numpy as np
from process_bigraph import Process


_WALL_HYSTERESIS = 1e-2


def _wall_z_equal(a, b):
    """Treat None as a distinct value (no wall). Float comparisons use a
    1e-2 hysteresis tolerance: a coupled bigraph composite typically
    publishes a slightly-different wall_z every step (upstream signal is
    noisy from membrane vertex jitter), and a tighter threshold would
    rebuild the entire ReaDDy system on every PBG step. With each rebuild
    costing O(100ms) for a small system + topologies, that turns a
    12-second sim into multi-minute wall time. 1e-2 (~1% of a typical
    barrier position) preserves real coupling dynamics while filtering out
    micro-jitter."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) < _WALL_HYSTERESIS


def _wall_radius_equal(a, b):
    """Same hysteresis semantics as `_wall_z_equal`, for the spherical-
    barrier input port (`wall_radius`)."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) < _WALL_HYSTERESIS


class ReaDDyProcess(Process):
    """Bridge Process wrapping ReaDDy particle-based reaction-diffusion.

    Simulates particles diffusing in a box, interacting via potentials,
    and undergoing stochastic reactions. On each update(), advances the
    ReaDDy Brownian dynamics integrator by the requested time interval,
    then returns updated particle counts, positions, and energy.

    Config:
        box_size: Simulation box dimensions [x, y, z]
        periodic: Periodic boundary conditions [bool, bool, bool]
        kbt: Thermal energy (kBT). If 0, uses readdy default.
        timestep: Integration timestep for Brownian dynamics
        species: Dict mapping species name -> diffusion_constant
        reactions: List of reaction descriptor dicts:
            {'descriptor': 'A +(2) A -> B', 'rate': 5.0}
        potentials: List of potential dicts:
            {'type': 'harmonic_repulsion', 'species1': 'A', 'species2': 'A',
             'force_constant': 10.0, 'interaction_distance': 1.0}
        initial_particles: Dict mapping species name -> list of [x,y,z] positions
        observe_stride: Stride for recording observables (in integration steps)
        reaction_handler: 'Gillespie' or 'UncontrolledApproximation'
    """

    config_schema = {
        'box_size': {'_type': 'tuple[float,float,float]',
                     '_default': (10.0, 10.0, 10.0)},
        'periodic': {'_type': 'tuple[boolean,boolean,boolean]',
                     '_default': (True, True, True)},
        'kbt': {'_type': 'float', '_default': 0.0},
        'timestep': {'_type': 'float', '_default': 0.01},
        'observe_stride': {'_type': 'integer', '_default': 1},
        'reaction_handler': {'_type': 'string', '_default': 'Gillespie'},
    }

    _COMPLEX_DEFAULTS = {
        'species': {'A': 1.0},
        'reactions': [],
        'potentials': [],
        'initial_particles': {},
        # ---- Topology config (opt-in; bonded chains / filaments) ----
        # Topology species are particles that participate in bonded
        # complexes — distinct from free species above. Same {name:
        # diffusion_constant} shape.
        'topology_species': {},
        # Topology type names. Each instance below references one of these.
        'topology_types': [],
        # Harmonic bond definitions between topology species, applied to
        # every bonded edge between matching particle types.
        # [{type1, type2, force_constant, length}, ...]
        'topology_bonds': [],
        # Optional harmonic angle potentials for chain bending stiffness.
        # [{type1, type2, type3, force_constant, equilibrium_angle}, ...]
        'topology_angles': [],
        # Initial topology instances. Each entry creates one topology in
        # the simulation:
        # {
        #   'type': 'filament',
        #   'particle_types': ['F', 'F', ...],   # one per particle
        #   'positions': [[x, y, z], ...],       # matching length
        #   'edges': [[i, j], ...],              # bonds within the topology;
        #                                        # default = sequential chain
        # }
        'initial_topologies': [],
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        for key, default in self._COMPLEX_DEFAULTS.items():
            if key not in self.config:
                self.config[key] = default
        self._system = None
        self._simulation = None
        self._species_list = None
        self._cumulative_steps = 0
        # Callback-accumulated data
        self._count_data = []
        self._energy_data = []
        self._position_data = []
        # Track the wall-z barrier currently baked into the box potentials.
        # None = no wall (default — existing demos behave identically). A
        # numeric value triggers add_box potentials confining every species
        # to z <= wall_z. When the input port reports a new value, the
        # simulation is rebuilt with the box potential repositioned.
        self._current_wall_z = None
        # Spherical barrier — when set, every species gets an add_sphere
        # inclusion potential of this radius centered at the origin. Used
        # by composites where actin is confined inside a vesicle and
        # pushes radially outward against the membrane.
        self._current_wall_radius = None
        # Cached per-species position snapshot used to restore particles
        # across a rebuild. Populated each update() from the most recent
        # callback-recorded positions.
        self._particle_snapshot = None

    def inputs(self):
        # `wall_z`: planar upper barrier — confines particles to z <= wall_z
        # via add_box. Used for actin pushing UP against a membrane patch.
        # `wall_radius`: spherical barrier — confines particles to a sphere
        # of that radius centered at the origin via add_sphere(inclusion).
        # Used for actin INSIDE a vesicle pushing radially outward.
        # Both are maybe[float] so None unambiguously means "no barrier".
        return {
            'wall_z': 'maybe[float]',
            'wall_radius': 'maybe[float]',
        }

    def outputs(self):
        return {
            'particle_counts': 'overwrite[map[integer]]',
            'total_particles': 'overwrite[integer]',
            'positions': 'overwrite[list]',
            'energy': 'overwrite[float]',
            'time': 'overwrite[float]',
        }

    def _build_system(self, override_initial_particles=None,
                      override_initial_topologies=None,
                      wall_z=None, wall_radius=None):
        """Initialize (or rebuild) the ReaDDy system and simulation.

        First call: builds System with cfg['initial_particles'] and
        cfg['initial_topologies'].

        Subsequent calls (rebuild path triggered by wall_z OR wall_radius
        input changes): the caller supplies the current per-species
        particle snapshot AND the current topology snapshot, and the
        appropriate barrier potentials are added on every species before
        run() is called. Discards the prior simulation, system, and
        accumulated observable buffers.
        """
        if (self._system is not None and override_initial_particles is None
                and override_initial_topologies is None):
            return

        import readdy

        cfg = self.config
        box = list(cfg['box_size'])

        kwargs = {
            'box_size': box,
            'periodic_boundary_conditions': list(cfg['periodic']),
        }
        if cfg['kbt'] > 0:
            kwargs['unit_system'] = None
            kwargs['temperature'] = cfg['kbt']

        self._system = readdy.ReactionDiffusionSystem(**kwargs)

        # Add species
        self._species_list = sorted(cfg['species'].keys())
        for name, diff_const in cfg['species'].items():
            self._system.add_species(name, diffusion_constant=diff_const)

        # Add topology species — distinct from regular species; only these
        # can participate in bonded topologies (chains/filaments).
        self._topology_species_list = sorted(cfg['topology_species'].keys())
        for name, diff_const in cfg['topology_species'].items():
            self._system.add_topology_species(name, diffusion_constant=diff_const)

        # Register topology types and their bond/angle templates.
        for type_name in cfg['topology_types']:
            self._system.topologies.add_type(type_name)
        for bond in cfg['topology_bonds']:
            self._system.topologies.configure_harmonic_bond(
                bond['type1'], bond['type2'],
                force_constant=bond['force_constant'],
                length=bond['length'],
            )
        for angle in cfg['topology_angles']:
            self._system.topologies.configure_harmonic_angle(
                angle['type1'], angle['type2'], angle['type3'],
                force_constant=angle['force_constant'],
                equilibrium_angle=angle['equilibrium_angle'],
            )

        # Add reactions
        for rxn in cfg['reactions']:
            if rxn.get('method') == 'enzymatic':
                self._system.reactions.add_enzymatic(
                    rxn['name'], rxn['catalyst'], rxn['type_from'],
                    rxn['type_to'], rate=rxn['rate'],
                    educt_distance=rxn.get('educt_distance', 2.0))
            else:
                self._system.reactions.add(
                    rxn['descriptor'], rate=rxn['rate'])

        # Add potentials
        for pot in cfg['potentials']:
            self._add_potential(pot)

        # Membrane-imposed wall barriers. Two flavors:
        #   wall_z: planar — confine to z <= wall_z (add_box).
        #   wall_radius: spherical — confine inside a sphere of that radius
        #                centered at origin (add_sphere with inclusion=True).
        # Both apply to every species (regular AND topology). A sibling
        # membrane simulator can drive either via the corresponding input.
        all_species = self._species_list + self._topology_species_list
        if wall_z is not None:
            origin = [-box[0] / 2.0, -box[1] / 2.0, -box[2] / 2.0]
            extent_z = max(1e-6, wall_z - origin[2])
            extent = [box[0], box[1], extent_z]
            for species_name in all_species:
                self._system.potentials.add_box(
                    species_name, force_constant=50.0,
                    origin=origin, extent=extent)
        if wall_radius is not None:
            r = max(1e-6, float(wall_radius))
            for species_name in all_species:
                # Stiff wall — bonded filaments + their angle potentials
                # generate substantial inward drift that a soft barrier
                # can't contain. 500 is high enough to keep particles
                # confined to within ~10% of the radius even under load.
                self._system.potentials.add_sphere(
                    species_name, force_constant=500.0,
                    origin=[0.0, 0.0, 0.0], radius=r, inclusion=True)
        self._current_wall_z = wall_z
        self._current_wall_radius = wall_radius

        # Create simulation (no output file — use callbacks for multi-run support)
        self._simulation = self._system.simulation(kernel='CPU')
        self._simulation.reaction_handler = cfg['reaction_handler']
        self._simulation.show_progress = False

        # Register observable callbacks
        stride = cfg['observe_stride']
        self._simulation.observe.number_of_particles(
            stride=stride, types=self._species_list,
            callback=lambda x: self._count_data.append(np.array(x)))
        self._simulation.observe.energy(
            stride=stride,
            callback=lambda x: self._energy_data.append(float(x)))
        self._simulation.observe.particle_positions(
            stride=stride,
            callback=lambda x: self._position_data.append(
                [[p[0], p[1], p[2]] for p in x]))

        # Add initial particles. On rebuild, the caller passes the
        # most-recent per-species snapshot via override_initial_particles
        # so the new simulation continues from where the old left off.
        initial = (override_initial_particles
                   if override_initial_particles is not None
                   else cfg['initial_particles'])
        for species_name, positions in initial.items():
            for pos in positions:
                self._simulation.add_particle(species_name, pos)

        # Add initial topologies (bonded filaments / chains). On rebuild,
        # the caller passes a snapshot with current positions and edges so
        # the chains' bond structure survives the rebuild intact.
        initial_tops = (override_initial_topologies
                        if override_initial_topologies is not None
                        else cfg['initial_topologies'])
        for topo_def in initial_tops:
            positions_arr = np.asarray(topo_def['positions'], dtype=np.float64)
            top = self._simulation.add_topology(
                topo_def['type'],
                list(topo_def['particle_types']),
                positions_arr,
            )
            graph = top.get_graph()
            edges = topo_def.get('edges')
            if edges is None:
                # Default: sequential chain — bond consecutive particles.
                edges = [[i, i + 1] for i in range(len(positions_arr) - 1)]
            for i, j in edges:
                graph.add_edge(int(i), int(j))

    def _add_potential(self, pot):
        """Add a potential to the system from a config dict."""
        ptype = pot['type']
        if ptype == 'harmonic_repulsion':
            self._system.potentials.add_harmonic_repulsion(
                pot['species1'], pot['species2'],
                force_constant=pot['force_constant'],
                interaction_distance=pot['interaction_distance'])
        elif ptype == 'lennard_jones':
            self._system.potentials.add_lennard_jones(
                pot['species1'], pot['species2'],
                m=pot.get('m', 12), n=pot.get('n', 6),
                cutoff=pot['cutoff'], shift=pot.get('shift', True),
                epsilon=pot['epsilon'], sigma=pot['sigma'])
        elif ptype == 'weak_interaction':
            self._system.potentials.add_weak_interaction_piecewise_harmonic(
                pot['species1'], pot['species2'],
                force_constant=pot['force_constant'],
                desired_distance=pot['desired_distance'],
                depth=pot['depth'], cutoff=pot['cutoff'])
        elif ptype == 'box':
            self._system.potentials.add_box(
                pot['species'], force_constant=pot['force_constant'],
                origin=pot['origin'], extent=pot['extent'])
        elif ptype == 'screened_electrostatics':
            self._system.potentials.add_screened_electrostatics(
                pot['species1'], pot['species2'],
                electrostatic_strength=pot['electrostatic_strength'],
                inverse_screening_depth=pot['inverse_screening_depth'],
                repulsion_strength=pot['repulsion_strength'],
                repulsion_distance=pot['repulsion_distance'],
                exponent=pot.get('exponent', 6),
                cutoff=pot['cutoff'])

    def initial_state(self):
        self._build_system()
        count_dict = {}
        total = 0
        positions = []
        for species_name, pos_list in self.config['initial_particles'].items():
            count_dict[species_name] = len(pos_list)
            total += len(pos_list)
            positions.extend(pos_list)
        return {
            'particle_counts': count_dict,
            'total_particles': total,
            'positions': positions,
            'energy': 0.0,
            'time': 0.0,
        }

    def update(self, state, interval):
        self._build_system()

        # Runtime barrier-rebuild path. When a sibling process publishes a
        # new wall_z OR a new wall_radius, snapshot every live particle AND
        # every live topology, drop the current simulation, and rebuild with
        # the appropriate barrier potential. Costly (full ReaDDy rebuild +
        # state restoration), so the change-detection guards matter.
        new_wall_z = state.get('wall_z')
        new_wall_radius = state.get('wall_radius')
        if (not _wall_z_equal(new_wall_z, self._current_wall_z)
                or not _wall_radius_equal(new_wall_radius, self._current_wall_radius)):
            particle_snapshot = self._snapshot_particles_by_species()
            topology_snapshot = self._snapshot_topologies()
            self._system = None
            self._simulation = None
            self._species_list = None
            self._count_data = []
            self._energy_data = []
            self._position_data = []
            self._build_system(
                override_initial_particles=particle_snapshot,
                override_initial_topologies=topology_snapshot,
                wall_z=new_wall_z, wall_radius=new_wall_radius,
            )

        dt = self.config['timestep']
        n_steps = max(1, int(round(interval / dt)))

        self._simulation.run(n_steps, timestep=dt)
        self._cumulative_steps += n_steps

        # Read latest state from accumulated callback data
        count_dict = {}
        if self._count_data:
            latest_counts = self._count_data[-1]
            for i, sp in enumerate(self._species_list):
                count_dict[sp] = int(latest_counts[i])

        total = sum(count_dict.values()) if count_dict else 0
        e = self._energy_data[-1] if self._energy_data else 0.0
        positions = self._position_data[-1] if self._position_data else []
        t = self._cumulative_steps * dt

        return {
            'particle_counts': count_dict,
            'total_particles': total,
            'positions': positions,
            'energy': e,
            'time': round(t, 6),
        }

    def _snapshot_particles_by_species(self):
        """Return per-species [[x,y,z], ...] positions for every live FREE
        particle (not part of a topology).

        Used by the rebuild path to seed the new simulation with the same
        particles as the old one. Reads from `simulation.current_particles`
        and excludes any particle that's part of a topology — those are
        handled by `_snapshot_topologies`.
        """
        snapshot = {sp: [] for sp in (self._species_list or [])}
        if self._simulation is None:
            return snapshot
        # Build a set of topology-particle IDs to exclude.
        topology_particle_ids = set()
        for top in self._simulation.current_topologies:
            for p in top.particles:
                topology_particle_ids.add(p.id)
        for particle in self._simulation.current_particles:
            if particle.id in topology_particle_ids:
                continue
            sp = particle.type
            pos = particle.pos
            snapshot.setdefault(sp, []).append([float(pos[0]), float(pos[1]), float(pos[2])])
        return snapshot

    def _snapshot_topologies(self):
        """Return a list of topology snapshots suitable for rebuild.

        Each entry: {type, particle_types, positions, edges}. Edges are
        local-within-topology indices (0..N-1) matching the order of
        `particle_types` and `positions`, so the new simulation can rebond
        them via add_edge().

        ReaDDy quirk: `vertex.particle_index` is an opaque per-simulation
        index that does not equal `particle.id` (the global ID counter
        across simulation rebuilds) and does not equal the local
        topology-relative index. The graph's vertex list is in 1-to-1
        positional correspondence with `top.particles`, so we build a
        `vertex.particle_index → local_index` map from that enumeration.
        """
        if self._simulation is None:
            return []
        snapshots = []
        for top in self._simulation.current_topologies:
            particles = list(top.particles)
            particle_types = [p.type for p in particles]
            positions = [[float(p.pos[0]), float(p.pos[1]), float(p.pos[2])] for p in particles]
            graph = top.get_graph()
            # Build the opaque-particle-index → local-index map by
            # enumerating the graph's vertex list (same order as particles).
            pi_to_local = {
                v.particle_index: i for i, v in enumerate(graph.get_vertices())
            }
            edges = []
            for e in graph.get_edges():
                a = e[0].get().particle_index
                b = e[1].get().particle_index
                if a in pi_to_local and b in pi_to_local:
                    edges.append([pi_to_local[a], pi_to_local[b]])
                # Else: malformed edge — silently drop.
            snapshots.append({
                'type': top.type,
                'particle_types': particle_types,
                'positions': positions,
                'edges': edges,
            })
        return snapshots

    def get_trajectory_data(self):
        """Return full time-series data from callback-accumulated observables.

        Returns dict with 'times', 'counts' (dict per species), 'energy'.
        Must be called after update().
        """
        dt = self.config['timestep']
        stride = self.config['observe_stride']

        n_points = len(self._energy_data)
        times = [i * stride * dt for i in range(n_points)]

        count_series = {}
        if self._count_data:
            for i, sp in enumerate(self._species_list):
                count_series[sp] = [int(c[i]) for c in self._count_data]

        return {
            'times': times,
            'counts': count_series,
            'energy': list(self._energy_data),
        }

    def get_position_snapshots(self, stride=None):
        """Return particle position snapshots from callback data.

        Args:
            stride: Only return every Nth snapshot. Default: return all.

        Returns list of dicts with 'time' and 'positions' (list of [x,y,z]).
        """
        dt = self.config['timestep']
        obs_stride = self.config['observe_stride']

        snapshots = []
        for idx, positions in enumerate(self._position_data):
            if stride and idx % stride != 0:
                continue
            real_time = idx * obs_stride * dt
            snapshots.append({
                'time': round(real_time, 6),
                'positions': positions,
            })
        return snapshots
