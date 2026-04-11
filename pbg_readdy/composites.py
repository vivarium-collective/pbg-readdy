"""Pre-built composite document factories for ReaDDy simulations."""


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
