"""Demo: ReaDDy multi-configuration reaction-diffusion report with 3D viewers.

Runs three distinct particle-based reaction-diffusion simulations
(annihilation, Lotka-Volterra, crowded diffusion), generates interactive
3D particle viewers with Three.js, Plotly charts, bigraph-viz diagrams,
and navigatable PBG document trees — all in a single self-contained HTML.
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


# ── Simulation Configs ──────────────────────────────────────────────

def _random_positions(n, box_half=4.0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    return (np.random.random((n, 3)) * 2 * box_half - box_half).tolist()


CONFIGS = [
    {
        'id': 'annihilation',
        'title': 'Annihilation Kinetics',
        'subtitle': 'A + A → B fusion with excluded-volume interactions',
        'description': (
            'Fifty A particles diffuse in a periodic box and undergo '
            'bimolecular fusion (A + A → B) with strong excluded-volume '
            'repulsion. As A particles are consumed, the slower-diffusing '
            'B products accumulate. This demonstrates second-order reaction '
            'kinetics in a crowded environment where spatial correlations '
            'affect the effective reaction rate.'
        ),
        'config': {
            'box_size': (12., 12., 12.),
            'species': {'A': 1.0, 'B': 0.3},
            'reactions': [
                {'descriptor': 'fusion: A +(2) A -> B', 'rate': 5.0},
            ],
            'potentials': [
                {'type': 'harmonic_repulsion', 'species1': 'A',
                 'species2': 'A', 'force_constant': 10.,
                 'interaction_distance': 1.5},
                {'type': 'harmonic_repulsion', 'species1': 'A',
                 'species2': 'B', 'force_constant': 10.,
                 'interaction_distance': 1.5},
                {'type': 'harmonic_repulsion', 'species1': 'B',
                 'species2': 'B', 'force_constant': 10.,
                 'interaction_distance': 1.5},
            ],
            'initial_particles': {'A': _random_positions(50, 5.0, seed=42)},
            'timestep': 0.005,
            'observe_stride': 50,
        },
        'n_steps': 10000,
        'camera': [14, 10, 14],
        'color_scheme': 'indigo',
        'species_colors': {'A': '#6366f1', 'B': '#f43f5e'},
    },
    {
        'id': 'lotka_volterra',
        'title': 'Lotka-Volterra Oscillations',
        'subtitle': 'Predator-prey dynamics with spatial stochasticity',
        'description': (
            'A spatial Lotka-Volterra predator-prey model: prey (A) reproduce '
            'by fission, predators (B) consume prey upon contact via enzymatic '
            'reaction (A + B → B + B), and predators spontaneously decay. '
            'Spatial diffusion and stochastic reactions introduce noise and '
            'fluctuations around the classical oscillatory dynamics.'
        ),
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
        'n_steps': 10000,
        'camera': [18, 12, 18],
        'color_scheme': 'emerald',
        'species_colors': {'A': '#10b981', 'B': '#f59e0b'},
    },
    {
        'id': 'crowding',
        'title': 'Crowded Diffusion',
        'subtitle': 'Dense particle packing with excluded-volume repulsion',
        'description': (
            'One hundred particles with strong excluded-volume repulsion '
            'diffuse in a periodic box, reaching a disordered equilibrium '
            'packing. There are no reactions — this tests pure Brownian '
            'dynamics with pair potentials. The energy measures the total '
            'repulsive interaction, which decreases as particles spread '
            'apart to avoid overlap.'
        ),
        'config': {
            'box_size': (10., 10., 10.),
            'species': {'P': 0.8},
            'reactions': [],
            'potentials': [
                {'type': 'harmonic_repulsion', 'species1': 'P',
                 'species2': 'P', 'force_constant': 20.,
                 'interaction_distance': 2.0},
            ],
            'initial_particles': {'P': _random_positions(100, 4.5, seed=300)},
            'timestep': 0.002,
            'observe_stride': 50,
        },
        'n_steps': 10000,
        'camera': [12, 8, 12],
        'color_scheme': 'rose',
        'species_colors': {'P': '#f43f5e'},
    },
]


# ── Simulation Runner ───────────────────────────────────────────────

def run_simulation(cfg_entry):
    """Run a simulation, returning snapshots and trajectory data."""
    core = allocate_core()
    core.register_link('ReaDDyProcess', ReaDDyProcess)

    t0 = time.perf_counter()
    proc = ReaDDyProcess(config=cfg_entry['config'], core=core)
    proc.initial_state()

    dt = cfg_entry['config']['timestep']
    n_steps = cfg_entry['n_steps']
    interval = n_steps * dt
    proc.update({}, interval=interval)

    runtime = time.perf_counter() - t0

    traj_data = proc.get_trajectory_data()
    pos_snapshots = proc.get_position_snapshots()

    return traj_data, pos_snapshots, runtime


# ── Bigraph Image ───────────────────────────────────────────────────

def generate_bigraph_image(cfg_entry):
    """Generate a colored bigraph-viz PNG for the composite document."""
    from bigraph_viz import plot_bigraph

    species_list = sorted(cfg_entry['config']['species'].keys())
    emit_ports = {sp: f'overwrite[integer]' for sp in species_list}

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


def build_pbg_document(cfg_entry):
    """Build the PBG composite document dict for display."""
    cfg = cfg_entry['config']
    return make_readdy_document(
        box_size=list(cfg['box_size']),
        species=cfg['species'],
        reactions=cfg.get('reactions', []),
        potentials=cfg.get('potentials', []),
        initial_particles={k: f'[{len(v)} positions]'
                           for k, v in cfg.get('initial_particles', {}).items()},
        timestep=cfg['timestep'],
        observe_stride=cfg['observe_stride'],
        interval=cfg_entry['n_steps'] * cfg['timestep'],
    )


# ── Color Schemes ──────────────────────────────────────────────────

COLOR_SCHEMES = {
    'indigo': {'primary': '#6366f1', 'light': '#e0e7ff', 'dark': '#4338ca',
               'bg': '#eef2ff', 'accent': '#818cf8', 'text': '#312e81'},
    'emerald': {'primary': '#10b981', 'light': '#d1fae5', 'dark': '#059669',
                'bg': '#ecfdf5', 'accent': '#34d399', 'text': '#064e3b'},
    'rose': {'primary': '#f43f5e', 'light': '#ffe4e6', 'dark': '#e11d48',
             'bg': '#fff1f2', 'accent': '#fb7185', 'text': '#881337'},
}


# ── HTML Report Generator ──────────────────────────────────────────

def generate_html(sim_results, output_path):
    """Generate comprehensive HTML report."""
    sections_html = []
    all_js_data = {}

    for idx, (cfg, (traj_data, pos_snapshots, runtime)) in enumerate(sim_results):
        sid = cfg['id']
        cs = COLOR_SCHEMES[cfg['color_scheme']]
        species_list = sorted(cfg['config']['species'].keys())

        # Counts
        initial_total = sum(
            len(v) for v in cfg['config'].get('initial_particles', {}).values())
        final_counts = {sp: traj_data['counts'].get(sp, [0])[-1]
                        for sp in species_list}
        final_total = sum(final_counts.values())

        # Energy
        energies = traj_data['energy']
        e_min = min(energies) if energies else 0
        e_max = max(energies) if energies else 0

        # Times
        times = traj_data['times']
        total_time = times[-1] if times else 0

        # JS data for charts and 3D viewer
        # Downsample position snapshots if too many
        max_snaps = 50
        if len(pos_snapshots) > max_snaps:
            step = len(pos_snapshots) // max_snaps
            vis_snaps = pos_snapshots[::step]
        else:
            vis_snaps = pos_snapshots

        all_js_data[sid] = {
            'snapshots': vis_snaps,
            'camera': cfg['camera'],
            'box_size': list(cfg['config']['box_size']),
            'species_colors': cfg['species_colors'],
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
        rxn_strs = []
        for r in cfg['config'].get('reactions', []):
            if 'descriptor' in r:
                rxn_strs.append(r['descriptor'])
            elif r.get('method') == 'enzymatic':
                rxn_strs.append(
                    f"{r['name']}: {r['type_from']} + {r['catalyst']} "
                    f"-> {r['type_to']} + {r['catalyst']}")
        rxn_display = '; '.join(rxn_strs) if rxn_strs else 'None'

        section = f"""
    <div class="sim-section" id="sim-{sid}">
      <div class="sim-header" style="border-left: 4px solid {cs['primary']};">
        <div class="sim-number" style="background:{cs['light']}; color:{cs['dark']};">{idx+1}</div>
        <div>
          <h2 class="sim-title">{cfg['title']}</h2>
          <p class="sim-subtitle">{cfg['subtitle']}</p>
        </div>
      </div>
      <p class="sim-description">{cfg['description']}</p>

      <div class="metrics-row">
        <div class="metric"><span class="metric-label">Initial</span><span class="metric-value">{initial_total}</span><span class="metric-sub">particles</span></div>
        <div class="metric"><span class="metric-label">Final</span><span class="metric-value">{final_total}</span><span class="metric-sub">{counts_str}</span></div>
        <div class="metric"><span class="metric-label">Species</span><span class="metric-value">{len(species_list)}</span></div>
        <div class="metric"><span class="metric-label">Reactions</span><span class="metric-value">{len(rxn_strs)}</span><span class="metric-sub" title="{rxn_display}">{rxn_display[:30]}{'...' if len(rxn_display) > 30 else ''}</span></div>
        <div class="metric"><span class="metric-label">Time</span><span class="metric-value">{total_time:.1f}</span><span class="metric-sub">sim. units</span></div>
        <div class="metric"><span class="metric-label">Steps</span><span class="metric-value">{cfg['n_steps']:,}</span></div>
        <div class="metric"><span class="metric-label">Runtime</span><span class="metric-value">{runtime:.1f}s</span></div>
      </div>

      <h3 class="subsection-title">3D Particle Viewer</h3>
      <div class="viewer-wrap">
        <canvas id="canvas-{sid}" class="mesh-canvas"></canvas>
        <div class="viewer-info">
          <strong>{final_total}</strong> particles &middot; Box: {cfg['config']['box_size'][0]}&times;{cfg['config']['box_size'][1]}&times;{cfg['config']['box_size'][2]}<br>
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
  background:linear-gradient(135deg,#f8fafc 0%,#eef2ff 50%,#fdf2f8 100%);
  border-bottom:1px solid #e2e8f0; padding:3rem;
}}
.page-header h1 {{ font-size:2.2rem; font-weight:800; color:#0f172a; margin-bottom:.3rem; }}
.page-header p {{ color:#64748b; font-size:.95rem; max-width:700px; }}
.nav {{ display:flex; gap:.8rem; padding:1rem 3rem; background:#f8fafc;
        border-bottom:1px solid #e2e8f0; position:sticky; top:0; z-index:100; }}
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
}}
</style>
</head>
<body>

<div class="page-header">
  <h1>ReaDDy Reaction-Diffusion Simulation Report</h1>
  <p>Three particle-based reaction-diffusion simulations wrapped as
  <strong>process-bigraph</strong> Processes using ReaDDy's Brownian dynamics
  engine. Each configuration demonstrates a distinct biophysical scenario
  with interactive 3D visualization and population dynamics.</p>
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

// ─── JSON Tree Viewer ───
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

// ─── Three.js Particle Viewers ───
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
  const cam = new THREE.PerspectiveCamera(45, W/H, 0.1, 200);
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

  // Species color map
  const speciesColors = d.species_colors;
  const speciesNames = Object.keys(speciesColors);

  // Create sphere geometry (shared)
  const sphereGeo = new THREE.SphereGeometry(0.35, 12, 8);

  // Instanced meshes per species
  const meshes = {{}};
  const maxParticles = 200;  // max per species
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

    // Group positions by species based on order (ReaDDy preserves type order)
    // Since we don't have type info per particle in callback data,
    // we color all particles with the first species color, or if multiple species,
    // we need a different approach.
    // For single-species configs, all particles get one color.
    // For multi-species, we use a simple heuristic: position-based coloring.

    // Actually, for the demo, let's assign colors by index ranges based on
    // initial counts and the known number of each species from chart data.
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

    // Get counts at this time
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

    // Handle any remaining particles (assign to last species)
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

// ─── Plotly Charts ───
const pLayout = {{
  paper_bgcolor:'#f8fafc', plot_bgcolor:'#f8fafc',
  font:{{ color:'#64748b', family:'-apple-system,sans-serif', size:11 }},
  margin:{{ l:55, r:15, t:35, b:45 }},
  xaxis:{{ gridcolor:'#e2e8f0', zerolinecolor:'#e2e8f0',
           title:{{ text:'Time', font:{{ size:10 }} }} }},
  yaxis:{{ gridcolor:'#e2e8f0', zerolinecolor:'#e2e8f0' }},
}};
const pCfg = {{ responsive:true, displayModeBar:false }};

const chartColors = ['#6366f1','#10b981','#f43f5e','#f59e0b','#8b5cf6','#06b6d4'];

Object.keys(DATA).forEach(sid => {{
  const c = DATA[sid].charts;
  const speciesNames = Object.keys(c.counts);

  // Population chart
  const countTraces = speciesNames.map((sp, i) => ({{
    x: c.times, y: c.counts[sp], type:'scatter', mode:'lines',
    line:{{ color: DATA[sid].species_colors[sp] || chartColors[i % chartColors.length], width:2 }},
    name: sp,
  }}));

  Plotly.newPlot('chart-counts-'+sid, countTraces, {{
    ...pLayout,
    title:{{ text:'Particle Counts', font:{{ size:12, color:'#334155' }} }},
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
    for cfg in CONFIGS:
        print(f'Running: {cfg["title"]}...')
        traj_data, pos_snapshots, runtime = run_simulation(cfg)
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
