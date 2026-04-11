"""ReaDDy Process wrapper for process-bigraph.

Wraps the ReaDDy particle-based reaction-diffusion simulator as a
time-driven Process using the bridge pattern. The internal ReaDDy
system and simulation are lazily initialized on first update() call.
Uses callback-based observables so update() can be called multiple times.
"""

import numpy as np
from process_bigraph import Process


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

    def inputs(self):
        return {}

    def outputs(self):
        return {
            'particle_counts': 'overwrite[map[integer]]',
            'total_particles': 'overwrite[integer]',
            'positions': 'overwrite[list]',
            'energy': 'overwrite[float]',
            'time': 'overwrite[float]',
        }

    def _build_system(self):
        """Lazily initialize the ReaDDy system and simulation."""
        if self._system is not None:
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

        # Add initial particles
        for species_name, positions in cfg['initial_particles'].items():
            for pos in positions:
                self._simulation.add_particle(species_name, pos)

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
