"""Interactive Plotly visualizations for the six validation rigs.

Each `viz_rigN(record, out_html)` consumes a "viz record" dict produced by
the corresponding rig (when `run_one_trial(..., keep_record=True)` is
used) and writes a standalone HTML file.

The HTML files are pure browser artefacts — open with `open
output/rig1/viz.html` (macOS) or just double-click. Pan / rotate / zoom
in 3D; scrub through time with the slider.

Functions
---------
    viz_rig1(record, out_html)
        3D obstacle voxels + FIRI polytopes + MINCO trajectory animated.
    viz_rig2(record, out_html)
        Multi-drone trajectories animated with near-miss connectors.
    viz_rig3(record, out_html)
        Top-down hex perimeter — true vs estimated positions, drift arrow.
    viz_rig4(record, out_html)
        Top-down threat-response timeline with phase-coded inspector path.
    viz_rig5(record, out_html)
        Multi-panel: top-down + coverage timeline + battery bars.
    viz_rig6(record, out_html)
        Corridor box + MINCO trajectory + wind force + RTL marker.

    emit_viz(rig_id, record, out_html)
        Dispatcher that picks the right `viz_rigN` for `rig_id`.

Design choices
--------------
- Pure plotly.graph_objects (no plotly.express) so the HTML is fully
  self-contained and we control every trace explicitly.
- Animation uses `frames` + `sliders` so users can scrub time without
  the autoplay treadmill.
- Records are pure JSON-able dicts (no numpy arrays in the public
  surface) so the same plumbing can write a `.viz.json` cache, replay
  rigs offline, etc.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import plotly.graph_objects as go
except ImportError as e:  # pragma: no cover — surfaced via test_visualize
    raise ImportError(
        "plotly is required for src.validation.visualize. Install with "
        "`pip install plotly` (or `.venv/bin/pip install plotly` if you "
        "use the project venv)."
    ) from e


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


_DRONE_PALETTE = (
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
    "#aec7e8",
    "#ffbb78",
)


def _drone_color(i: int) -> str:
    return _DRONE_PALETTE[i % len(_DRONE_PALETTE)]


def _make_slider(
    n_frames: int,
    times: Sequence[float],
    prefix: str = "t = ",
    unit: str = " s",
) -> List[dict]:
    """Build a single time-slider config for a plotly Figure."""
    return [
        {
            "active": 0,
            "steps": [
                {
                    "method": "animate",
                    "label": f"{times[i]:.1f}",
                    "args": [
                        [str(i)],
                        {
                            "mode": "immediate",
                            "frame": {"duration": 0, "redraw": True},
                            "transition": {"duration": 0},
                        },
                    ],
                }
                for i in range(n_frames)
            ],
            "x": 0.1,
            "len": 0.85,
            "currentvalue": {
                "prefix": prefix,
                "suffix": unit,
                "visible": True,
                "xanchor": "right",
            },
            "transition": {"duration": 0},
        }
    ]


def _make_play_pause_buttons() -> List[dict]:
    """Standard play/pause controls for animated figures."""
    return [
        {
            "type": "buttons",
            "direction": "left",
            "x": 0.1,
            "y": -0.05,
            "buttons": [
                {
                    "label": "▶ Play",
                    "method": "animate",
                    "args": [
                        None,
                        {
                            "frame": {"duration": 80, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 0},
                        },
                    ],
                },
                {
                    "label": "⏸ Pause",
                    "method": "animate",
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                },
            ],
        }
    ]


def _save_html(fig: "go.Figure", out_html: str) -> None:
    """Write a standalone HTML file (Plotly JS embedded inline-cdn)."""
    fig.write_html(
        out_html,
        include_plotlyjs="cdn",
        full_html=True,
        config={"displayModeBar": True, "displaylogo": False},
    )


# ---------------------------------------------------------------------------
# Dispatcher (kept skeletal; each viz_rigN is filled in below)
# ---------------------------------------------------------------------------


def emit_viz(rig_id: str, record: Dict, out_html: str) -> None:
    """Dispatch to the matching `viz_rigN`."""
    fn = _VIZ_FNS.get(rig_id)
    if fn is None:
        raise ValueError(
            f"unknown rig_id {rig_id!r}; expected one of {sorted(_VIZ_FNS)}"
        )
    fn(record, out_html)


# Per-rig viz functions are defined below this point; the dispatcher table
# is populated at the bottom of the module.


# ---------------------------------------------------------------------------
# Rig 1: 3D corridor + animated MINCO trajectory
# ---------------------------------------------------------------------------


def viz_rig1(record: Dict, out_html: str) -> None:
    """Render Rig 1's pipeline output as an interactive 3D scene.

    Expected `record` keys:
        density               : float
        seed                  : int
        success               : bool
        obstacle_points       : list of [x, y, z]   (surface voxels)
        rrt_route             : list of [x, y, z]   (raw RRT waypoints)
        shortcut_route        : list of [x, y, z]   (post-shortcut)
        polytope_boxes        : list of {min: [x, y, z], max: [x, y, z]}
                                 axis-aligned bounding box approximations of
                                 each FIRI polytope (the AABBs are good
                                 enough for visual intuition; the true
                                 polytopes are general half-space sets)
        trajectory_samples    : list of {t: float, p: [x, y, z], v: float}
        start, goal           : [x, y, z]
    """
    obstacles = np.asarray(record.get("obstacle_points", []), dtype=np.float64)
    rrt_route = np.asarray(record.get("rrt_route", []), dtype=np.float64)
    shortcut = np.asarray(record.get("shortcut_route", []), dtype=np.float64)
    polys = record.get("polytope_boxes", [])
    samples = record.get("trajectory_samples", [])
    start = np.asarray(record.get("start", [0, 0, 0]), dtype=np.float64)
    goal = np.asarray(record.get("goal", [0, 0, 0]), dtype=np.float64)

    fig = go.Figure()

    # obstacle voxel cloud
    if obstacles.size:
        fig.add_trace(
            go.Scatter3d(
                x=obstacles[:, 0],
                y=obstacles[:, 1],
                z=obstacles[:, 2],
                mode="markers",
                marker=dict(size=2.5, color="#666", opacity=0.45),
                name="obstacles",
            )
        )

    # FIRI polytope AABBs as semi-transparent boxes
    for i, p in enumerate(polys):
        lo = p["min"]
        hi = p["max"]
        # 8 corners
        xs = [lo[0], hi[0], hi[0], lo[0], lo[0], hi[0], hi[0], lo[0]]
        ys = [lo[1], lo[1], hi[1], hi[1], lo[1], lo[1], hi[1], hi[1]]
        zs = [lo[2], lo[2], lo[2], lo[2], hi[2], hi[2], hi[2], hi[2]]
        # 12 triangle indices (2 per face)
        I = [0, 0, 0, 0, 4, 4, 1, 1, 2, 2, 3, 3]
        J = [1, 3, 4, 1, 5, 7, 2, 5, 3, 6, 0, 7]
        K = [2, 2, 5, 5, 6, 6, 5, 6, 6, 7, 7, 4]
        fig.add_trace(
            go.Mesh3d(
                x=xs, y=ys, z=zs, i=I, j=J, k=K,
                color="#2ca02c",
                opacity=0.10,
                showlegend=(i == 0),
                name="FIRI polytope" if i == 0 else None,
                hoverinfo="skip",
            )
        )

    # RRT route + shortcut
    if rrt_route.size:
        fig.add_trace(
            go.Scatter3d(
                x=rrt_route[:, 0], y=rrt_route[:, 1], z=rrt_route[:, 2],
                mode="lines+markers",
                line=dict(color="#9467bd", width=3, dash="dot"),
                marker=dict(size=4, color="#9467bd"),
                name="RRT route",
                opacity=0.5,
            )
        )
    if shortcut.size:
        fig.add_trace(
            go.Scatter3d(
                x=shortcut[:, 0], y=shortcut[:, 1], z=shortcut[:, 2],
                mode="lines+markers",
                line=dict(color="#d62728", width=5),
                marker=dict(size=6, color="#d62728"),
                name="shortcut route",
            )
        )

    # MINCO trajectory line (colour-coded by velocity magnitude)
    if samples:
        xs = [s["p"][0] for s in samples]
        ys = [s["p"][1] for s in samples]
        zs = [s["p"][2] for s in samples]
        vs = [s["v"] for s in samples]
        fig.add_trace(
            go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines",
                line=dict(color=vs, colorscale="Viridis", width=6,
                          colorbar=dict(title="‖v‖ m/s", x=1.02, len=0.75)),
                name="MINCO trajectory",
            )
        )

    # start & goal markers
    fig.add_trace(
        go.Scatter3d(
            x=[start[0]], y=[start[1]], z=[start[2]],
            mode="markers+text",
            marker=dict(size=10, color="#2ca02c", symbol="diamond"),
            text=["start"], textposition="top center",
            name="start",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[goal[0]], y=[goal[1]], z=[goal[2]],
            mode="markers+text",
            marker=dict(size=10, color="#d62728", symbol="x"),
            text=["goal"], textposition="top center",
            name="goal",
        )
    )

    # animated position marker
    if samples:
        frames = []
        times = [s["t"] for s in samples]
        for i, s in enumerate(samples):
            frames.append(
                go.Frame(
                    name=str(i),
                    data=[
                        go.Scatter3d(
                            x=[s["p"][0]], y=[s["p"][1]], z=[s["p"][2]],
                            mode="markers",
                            marker=dict(size=8, color="#ff7f0e",
                                        symbol="circle"),
                            name="drone",
                        )
                    ],
                    traces=[len(fig.data)],
                )
            )
        # initial marker (frame 0)
        fig.add_trace(
            go.Scatter3d(
                x=[samples[0]["p"][0]],
                y=[samples[0]["p"][1]],
                z=[samples[0]["p"][2]],
                mode="markers",
                marker=dict(size=8, color="#ff7f0e", symbol="circle"),
                name="drone",
            )
        )
        fig.frames = frames
        fig.update_layout(
            sliders=_make_slider(len(frames), times),
            updatemenus=_make_play_pause_buttons(),
        )

    fig.update_layout(
        title=(
            f"Rig 1 — corridor / RRT / MINCO  "
            f"(density={record.get('density', '?')}, "
            f"success={record.get('success', '?')})"
        ),
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
            aspectmode="data",
        ),
        height=800,
        margin=dict(l=0, r=0, b=80, t=50),
    )
    _save_html(fig, out_html)


# ---------------------------------------------------------------------------
# Rig 2: multi-drone animated swarm
# ---------------------------------------------------------------------------


def viz_rig2(record: Dict, out_html: str) -> None:
    """Render Rig 2's swarm scenario as an interactive 3D animation.

    Expected `record` keys:
        n_drones       : int
        scenario       : str
        d_min_inter_m  : float
        near_miss_radius_m : float
        sample_dt_s    : float
        positions_per_drone : list of list of [x, y, z]
            positions_per_drone[i][j] = drone i's position at time j*sample_dt.
        endpoints      : list of {start: [x,y,z], goal: [x,y,z]}
    """
    pos_lists = record.get("positions_per_drone", [])
    if not pos_lists:
        # Empty scene fallback
        fig = go.Figure()
        _save_html(fig, out_html)
        return

    positions = np.asarray(pos_lists, dtype=np.float64)  # (N, T, 3)
    N, T, _ = positions.shape
    dt = float(record.get("sample_dt_s", 0.1))
    times = [j * dt for j in range(T)]
    nm_radius = float(record.get("near_miss_radius_m", 1.5))

    fig = go.Figure()

    # static trajectory lines (one per drone)
    for i in range(N):
        fig.add_trace(
            go.Scatter3d(
                x=positions[i, :, 0],
                y=positions[i, :, 1],
                z=positions[i, :, 2],
                mode="lines",
                line=dict(color=_drone_color(i), width=3),
                name=f"drone {i} path",
                opacity=0.35,
                hoverinfo="skip",
            )
        )

    # endpoints
    eps = record.get("endpoints", [])
    if eps:
        ex = [e["start"][0] for e in eps] + [e["goal"][0] for e in eps]
        ey = [e["start"][1] for e in eps] + [e["goal"][1] for e in eps]
        ez = [e["start"][2] for e in eps] + [e["goal"][2] for e in eps]
        labels = (
            [f"S{i}" for i in range(len(eps))]
            + [f"G{i}" for i in range(len(eps))]
        )
        fig.add_trace(
            go.Scatter3d(
                x=ex, y=ey, z=ez,
                mode="markers+text",
                marker=dict(
                    size=5,
                    color=(
                        ["#2ca02c"] * len(eps) + ["#d62728"] * len(eps)
                    ),
                    symbol="diamond",
                ),
                text=labels, textposition="top center",
                name="start/goal",
                opacity=0.7,
            )
        )

    # current-position markers per drone (placeholder for animation)
    drone_marker_trace_indices = []
    for i in range(N):
        idx = len(fig.data)
        drone_marker_trace_indices.append(idx)
        fig.add_trace(
            go.Scatter3d(
                x=[positions[i, 0, 0]],
                y=[positions[i, 0, 1]],
                z=[positions[i, 0, 2]],
                mode="markers+text",
                marker=dict(size=8, color=_drone_color(i)),
                text=[f"D{i}"], textposition="top center",
                name=f"drone {i}",
            )
        )

    # near-miss connector trace (a single trace per frame, may be empty)
    nm_trace_idx = len(fig.data)
    fig.add_trace(
        go.Scatter3d(
            x=[], y=[], z=[],
            mode="lines",
            line=dict(color="#d62728", width=4),
            name=f"near-miss < {nm_radius:.1f} m",
        )
    )

    # build frames
    frames = []
    for j in range(T):
        frame_data = []
        for i in range(N):
            frame_data.append(
                go.Scatter3d(
                    x=[positions[i, j, 0]],
                    y=[positions[i, j, 1]],
                    z=[positions[i, j, 2]],
                    mode="markers+text",
                    marker=dict(size=8, color=_drone_color(i)),
                    text=[f"D{i}"], textposition="top center",
                )
            )
        # near-miss connectors
        nm_x: List[float] = []
        nm_y: List[float] = []
        nm_z: List[float] = []
        for a in range(N):
            for b in range(a + 1, N):
                d = float(
                    np.linalg.norm(positions[a, j] - positions[b, j])
                )
                if d < nm_radius:
                    nm_x.extend(
                        [positions[a, j, 0], positions[b, j, 0], None]
                    )
                    nm_y.extend(
                        [positions[a, j, 1], positions[b, j, 1], None]
                    )
                    nm_z.extend(
                        [positions[a, j, 2], positions[b, j, 2], None]
                    )
        frame_data.append(
            go.Scatter3d(
                x=nm_x, y=nm_y, z=nm_z,
                mode="lines",
                line=dict(color="#d62728", width=4),
            )
        )
        frames.append(
            go.Frame(
                name=str(j),
                data=frame_data,
                traces=drone_marker_trace_indices + [nm_trace_idx],
            )
        )

    fig.frames = frames
    fig.update_layout(
        title=(
            f"Rig 2 — swarm avoidance  "
            f"(N={N}, scenario={record.get('scenario', '?')}, "
            f"d_min={record.get('d_min_inter_m', float('nan')):.2f} m, "
            f"collisions={int(record.get('collisions', 0))})"
        ),
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
            aspectmode="data",
        ),
        height=800,
        margin=dict(l=0, r=0, b=80, t=50),
        sliders=_make_slider(len(frames), times),
        updatemenus=_make_play_pause_buttons(),
    )
    _save_html(fig, out_html)


# ---------------------------------------------------------------------------
# Rig 3: top-down hex perimeter with true vs estimated
# ---------------------------------------------------------------------------


def viz_rig3(record: Dict, out_html: str) -> None:
    """Top-down 2D view of hex perimeter with true / estimated drone twin
    markers and drift arrows over time.

    Expected `record` keys:
        n_drones                : int
        correction              : str ("on"/"off")
        perimeter_radius        : float
        altitude                : float
        perimeter_tolerance_m   : float
        sample_dt_s             : float
        truth_per_drone         : list of list of [x, y]   (T frames)
        estimated_per_drone     : list of list of [x, y]
    """
    truth = np.asarray(record.get("truth_per_drone", []), dtype=np.float64)
    est = np.asarray(record.get("estimated_per_drone", []), dtype=np.float64)
    if truth.size == 0:
        fig = go.Figure()
        _save_html(fig, out_html)
        return

    N, T, _ = truth.shape
    dt = float(record.get("sample_dt_s", 0.1))
    times = [j * dt for j in range(T)]
    R = float(record.get("perimeter_radius", 30.0))
    tol = float(record.get("perimeter_tolerance_m", 2.0))

    fig = go.Figure()

    # perimeter circle (intended path)
    theta = np.linspace(0.0, 2 * np.pi, 100)
    fig.add_trace(
        go.Scatter(
            x=R * np.cos(theta), y=R * np.sin(theta),
            mode="lines",
            line=dict(color="#888", width=2, dash="dot"),
            name=f"perimeter R={R:.1f} m",
        )
    )
    # tolerance band
    for r_band in (R - tol, R + tol):
        fig.add_trace(
            go.Scatter(
                x=r_band * np.cos(theta), y=r_band * np.sin(theta),
                mode="lines",
                line=dict(color="#d62728", width=1, dash="dash"),
                name=f"tolerance ±{tol:.1f} m" if r_band == R + tol else None,
                showlegend=(r_band == R + tol),
            )
        )

    # sector boundaries
    for i in range(N):
        ang = 2 * np.pi * i / N + np.pi / N
        fig.add_trace(
            go.Scatter(
                x=[0, (R + tol) * np.cos(ang)],
                y=[0, (R + tol) * np.sin(ang)],
                mode="lines",
                line=dict(color="#bbb", width=1, dash="dot"),
                showlegend=(i == 0),
                name="sector boundary" if i == 0 else None,
                hoverinfo="skip",
            )
        )

    # static historical paths (truth)
    for i in range(N):
        fig.add_trace(
            go.Scatter(
                x=truth[i, :, 0], y=truth[i, :, 1],
                mode="lines",
                line=dict(color=_drone_color(i), width=1),
                opacity=0.25,
                name=f"drone {i} truth", showlegend=False,
                hoverinfo="skip",
            )
        )

    # animated markers (truth + estimated) + drift connector
    truth_idx = []
    est_idx = []
    drift_idx = []
    for i in range(N):
        truth_idx.append(len(fig.data))
        fig.add_trace(
            go.Scatter(
                x=[truth[i, 0, 0]], y=[truth[i, 0, 1]],
                mode="markers",
                marker=dict(size=10, color=_drone_color(i), symbol="circle"),
                name=f"D{i} truth",
            )
        )
        est_idx.append(len(fig.data))
        fig.add_trace(
            go.Scatter(
                x=[est[i, 0, 0]], y=[est[i, 0, 1]],
                mode="markers",
                marker=dict(size=8, color=_drone_color(i), symbol="x",
                            line=dict(width=2, color="#000")),
                name=f"D{i} est",
            )
        )
        drift_idx.append(len(fig.data))
        fig.add_trace(
            go.Scatter(
                x=[truth[i, 0, 0], est[i, 0, 0]],
                y=[truth[i, 0, 1], est[i, 0, 1]],
                mode="lines",
                line=dict(color=_drone_color(i), width=1),
                opacity=0.6,
                showlegend=False,
            )
        )

    # frames
    frames = []
    for j in range(T):
        fdata = []
        for i in range(N):
            fdata.append(
                go.Scatter(
                    x=[truth[i, j, 0]], y=[truth[i, j, 1]],
                    mode="markers",
                    marker=dict(size=10, color=_drone_color(i)),
                )
            )
            fdata.append(
                go.Scatter(
                    x=[est[i, j, 0]], y=[est[i, j, 1]],
                    mode="markers",
                    marker=dict(size=8, color=_drone_color(i), symbol="x",
                                line=dict(width=2, color="#000")),
                )
            )
            fdata.append(
                go.Scatter(
                    x=[truth[i, j, 0], est[i, j, 0]],
                    y=[truth[i, j, 1], est[i, j, 1]],
                    mode="lines",
                    line=dict(color=_drone_color(i), width=1),
                    opacity=0.6,
                )
            )
        idx = [v for triple in zip(truth_idx, est_idx, drift_idx) for v in triple]
        frames.append(go.Frame(name=str(j), data=fdata, traces=idx))

    fig.frames = frames
    fig.update_layout(
        title=(
            f"Rig 3 — VIO drift  "
            f"(correction={record.get('correction', '?')}, "
            f"drift_max={record.get('drift_magnitude_max_m', 0):.3f} m, "
            f"perim_max={record.get('perimeter_deviation_max_m', 0):.3f} m)"
        ),
        xaxis_title="X (m)", yaxis_title="Y (m)",
        height=800, width=900,
        margin=dict(l=40, r=40, b=80, t=60),
        sliders=_make_slider(len(frames), times),
        updatemenus=_make_play_pause_buttons(),
        showlegend=True,
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    _save_html(fig, out_html)


# ---------------------------------------------------------------------------
# Rig 4: threat response timeline
# ---------------------------------------------------------------------------


def viz_rig4(record: Dict, out_html: str) -> None:
    """Top-down 2D view of mission response.

    Expected `record` keys:
        n_drones, perimeter_radius, altitude, threat_position, threat_time_s
        sample_dt_s
        positions_per_drone : list of list of [x, y, z]
        inspector_id        : int (-1 if no inspector chosen)
        inspector_phases    : optional list of {t, phase}
            phase ∈ {patrol, outbound, dwell, inbound}
        coverage_pct_over_time : optional list of float (one per sample)
    """
    pos_lists = record.get("positions_per_drone", [])
    if not pos_lists:
        fig = go.Figure()
        _save_html(fig, out_html)
        return

    positions = np.asarray(pos_lists, dtype=np.float64)  # (N, T, 3)
    N, T, _ = positions.shape
    dt = float(record.get("sample_dt_s", 0.1))
    times = [j * dt for j in range(T)]
    R = float(record.get("perimeter_radius", 30.0))
    threat = record.get("threat_position", [0, 0, 0])
    inspector_id = int(record.get("inspector_id", -1))

    fig = go.Figure()

    # perimeter circle
    theta = np.linspace(0.0, 2 * np.pi, 100)
    fig.add_trace(
        go.Scatter(
            x=R * np.cos(theta), y=R * np.sin(theta),
            mode="lines",
            line=dict(color="#888", width=2, dash="dot"),
            name="perimeter",
        )
    )

    # threat marker
    fig.add_trace(
        go.Scatter(
            x=[threat[0]], y=[threat[1]],
            mode="markers+text",
            marker=dict(size=18, color="#d62728", symbol="star",
                        line=dict(width=2, color="#000")),
            text=[f"threat @ t={record.get('threat_time_s', '?')}s"],
            textposition="top center",
            name="threat",
        )
    )

    # static historical paths
    for i in range(N):
        color = _drone_color(i)
        width = 4 if i == inspector_id else 2
        fig.add_trace(
            go.Scatter(
                x=positions[i, :, 0], y=positions[i, :, 1],
                mode="lines",
                line=dict(color=color, width=width),
                opacity=0.35,
                name=f"drone {i}" + (" (inspector)" if i == inspector_id else ""),
                hoverinfo="skip",
            )
        )

    # animated current-position markers
    marker_idx = []
    for i in range(N):
        marker_idx.append(len(fig.data))
        fig.add_trace(
            go.Scatter(
                x=[positions[i, 0, 0]], y=[positions[i, 0, 1]],
                mode="markers+text",
                marker=dict(
                    size=14 if i == inspector_id else 10,
                    color=_drone_color(i),
                    symbol="diamond" if i == inspector_id else "circle",
                ),
                text=[f"D{i}"], textposition="top center",
                name=f"D{i}",
            )
        )

    frames = []
    for j in range(T):
        fdata = []
        for i in range(N):
            fdata.append(
                go.Scatter(
                    x=[positions[i, j, 0]], y=[positions[i, j, 1]],
                    mode="markers+text",
                    marker=dict(
                        size=14 if i == inspector_id else 10,
                        color=_drone_color(i),
                        symbol="diamond" if i == inspector_id else "circle",
                    ),
                    text=[f"D{i}"], textposition="top center",
                )
            )
        frames.append(go.Frame(name=str(j), data=fdata, traces=marker_idx))

    fig.frames = frames
    fig.update_layout(
        title=(
            f"Rig 4 — mission response  "
            f"(inspector=D{inspector_id}, "
            f"t_replan={record.get('t_detect_to_replan_ms', '?')} ms, "
            f"cov={record.get('coverage_pct_during', 0):.1f}%)"
        ),
        xaxis_title="X (m)", yaxis_title="Y (m)",
        height=800, width=900,
        margin=dict(l=40, r=40, b=80, t=60),
        sliders=_make_slider(len(frames), times),
        updatemenus=_make_play_pause_buttons(),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    _save_html(fig, out_html)


# ---------------------------------------------------------------------------
# Rig 5: endurance multi-panel
# ---------------------------------------------------------------------------


def viz_rig5(record: Dict, out_html: str) -> None:
    """Multi-panel: top-down + coverage timeline + battery bars.

    Expected `record` keys:
        scenario, n_active, n_standby
        perimeter_radius, altitude
        sample_dt_s
        positions_per_drone : list of list of [x, y, z]
        status_per_drone    : list of list of str ("active"|"standby"|"returning"|"failed")
        battery_per_drone   : list of list of float  (SoC %)
        coverage_timeline   : list of [t, pct]
    """
    from plotly.subplots import make_subplots

    pos_lists = record.get("positions_per_drone", [])
    statuses = record.get("status_per_drone", [])
    batteries = record.get("battery_per_drone", [])
    cov_timeline = record.get("coverage_timeline", [])
    if not pos_lists:
        fig = go.Figure()
        _save_html(fig, out_html)
        return

    positions = np.asarray(pos_lists, dtype=np.float64)
    N, T, _ = positions.shape
    dt = float(record.get("sample_dt_s", 0.5))
    times = [j * dt for j in range(T)]
    R = float(record.get("perimeter_radius", 30.0))

    fig = make_subplots(
        rows=2, cols=2,
        column_widths=[0.6, 0.4],
        row_heights=[0.6, 0.4],
        specs=[
            [{"type": "xy", "rowspan": 2}, {"type": "xy"}],
            [None, {"type": "xy"}],
        ],
        subplot_titles=(
            "Top-down patrol",
            "Coverage %",
            "Battery SoC % per drone",
        ),
    )

    # ---- top-down ----
    theta = np.linspace(0.0, 2 * np.pi, 100)
    fig.add_trace(
        go.Scatter(
            x=R * np.cos(theta), y=R * np.sin(theta),
            mode="lines",
            line=dict(color="#888", width=2, dash="dot"),
            name="perimeter", showlegend=False,
        ),
        row=1, col=1,
    )

    pos_idx = []
    for i in range(N):
        pos_idx.append(len(fig.data))
        fig.add_trace(
            go.Scatter(
                x=[positions[i, 0, 0]], y=[positions[i, 0, 1]],
                mode="markers+text",
                marker=dict(size=12, color=_drone_color(i)),
                text=[f"D{i}"], textposition="top center",
                name=f"D{i}",
            ),
            row=1, col=1,
        )

    # ---- coverage timeline ----
    if cov_timeline:
        cov_t = [e[0] for e in cov_timeline]
        cov_v = [e[1] for e in cov_timeline]
        fig.add_trace(
            go.Scatter(
                x=cov_t, y=cov_v, mode="lines",
                line=dict(color="#2ca02c", width=2),
                name="coverage", showlegend=False,
            ),
            row=1, col=2,
        )
    # time cursor on coverage panel (placeholder, updated per frame)
    cov_cursor_idx = len(fig.data)
    fig.add_trace(
        go.Scatter(
            x=[times[0], times[0]], y=[0, 100],
            mode="lines",
            line=dict(color="#d62728", width=2, dash="dash"),
            name="t now", showlegend=False,
        ),
        row=1, col=2,
    )

    # ---- battery bars ----
    bar_idx = len(fig.data)
    if batteries:
        socs_t0 = [float(b[0]) if len(b) > 0 else 0.0 for b in batteries]
        fig.add_trace(
            go.Bar(
                x=[f"D{i}" for i in range(N)],
                y=socs_t0,
                marker=dict(color=[_drone_color(i) for i in range(N)]),
                name="SoC", showlegend=False,
            ),
            row=2, col=2,
        )
    else:
        bar_idx = -1

    # ---- frames ----
    frames = []
    for j in range(T):
        fdata = []
        # drone positions
        for i in range(N):
            status = (
                statuses[i][j]
                if statuses and i < len(statuses) and j < len(statuses[i])
                else "active"
            )
            color = _drone_color(i)
            marker = dict(size=12, color=color)
            if status == "failed":
                marker = dict(size=12, color="#333", symbol="x",
                              line=dict(width=2, color="#d62728"))
            elif status == "standby":
                marker = dict(size=10, color=color, symbol="circle-open",
                              line=dict(width=2, color=color))
            elif status == "returning":
                marker = dict(size=12, color=color, symbol="triangle-down")
            fdata.append(
                go.Scatter(
                    x=[positions[i, j, 0]], y=[positions[i, j, 1]],
                    mode="markers+text",
                    marker=marker,
                    text=[f"D{i}"], textposition="top center",
                )
            )
        traces_to_update = list(pos_idx)
        # cov cursor
        fdata.append(
            go.Scatter(
                x=[times[j], times[j]], y=[0, 100],
                mode="lines",
                line=dict(color="#d62728", width=2, dash="dash"),
            )
        )
        traces_to_update.append(cov_cursor_idx)
        # battery bars
        if bar_idx >= 0 and batteries:
            socs = [
                float(batteries[i][j])
                if i < len(batteries) and j < len(batteries[i])
                else 0.0
                for i in range(N)
            ]
            fdata.append(
                go.Bar(
                    x=[f"D{i}" for i in range(N)], y=socs,
                    marker=dict(color=[_drone_color(i) for i in range(N)]),
                )
            )
            traces_to_update.append(bar_idx)
        frames.append(
            go.Frame(name=str(j), data=fdata, traces=traces_to_update)
        )

    fig.frames = frames
    fig.update_layout(
        title=(
            f"Rig 5 — endurance / {record.get('scenario', '?')} "
            f"(cov_mean={record.get('coverage_pct_timeline_mean', 0):.1f}%, "
            f"gap_max={record.get('coverage_gap_max_s', 0):.1f}s)"
        ),
        height=900,
        margin=dict(l=40, r=40, b=80, t=80),
        sliders=_make_slider(len(frames), times),
        updatemenus=_make_play_pause_buttons(),
    )
    fig.update_xaxes(title_text="X (m)", row=1, col=1)
    fig.update_yaxes(title_text="Y (m)", scaleanchor="x", scaleratio=1, row=1, col=1)
    fig.update_xaxes(title_text="time (s)", row=1, col=2)
    fig.update_yaxes(title_text="coverage (%)", range=[0, 105], row=1, col=2)
    fig.update_yaxes(title_text="SoC (%)", range=[0, 100], row=2, col=2)
    _save_html(fig, out_html)


# ---------------------------------------------------------------------------
# Rig 6: corridor + wind + RTL
# ---------------------------------------------------------------------------


def viz_rig6(record: Dict, out_html: str) -> None:
    """3D corridor + animated drone trajectory with wind force arrow and
    RTL trigger annotation.

    Expected `record` keys:
        scenario, sample_dt_s
        start, goal
        corridor_min, corridor_max  (AABB)
        trajectory_samples : list of {t, p:[x,y,z], desired_p:[x,y,z], wind:[x,y,z]}
        rtl_trigger_time_s : float | None
    """
    samples = record.get("trajectory_samples", [])
    if not samples:
        fig = go.Figure()
        _save_html(fig, out_html)
        return

    times = [s["t"] for s in samples]
    start = record.get("start", [0, 0, 0])
    goal = record.get("goal", [0, 0, 0])
    corr_lo = record.get("corridor_min", start)
    corr_hi = record.get("corridor_max", goal)
    rtl_t = record.get("rtl_trigger_time_s", None)

    fig = go.Figure()

    # corridor AABB as a transparent mesh
    lo = corr_lo
    hi = corr_hi
    xs = [lo[0], hi[0], hi[0], lo[0], lo[0], hi[0], hi[0], lo[0]]
    ys = [lo[1], lo[1], hi[1], hi[1], lo[1], lo[1], hi[1], hi[1]]
    zs = [lo[2], lo[2], lo[2], lo[2], hi[2], hi[2], hi[2], hi[2]]
    I = [0, 0, 0, 0, 4, 4, 1, 1, 2, 2, 3, 3]
    J = [1, 3, 4, 1, 5, 7, 2, 5, 3, 6, 0, 7]
    K = [2, 2, 5, 5, 6, 6, 5, 6, 6, 7, 7, 4]
    fig.add_trace(
        go.Mesh3d(
            x=xs, y=ys, z=zs, i=I, j=J, k=K,
            color="#1f77b4", opacity=0.08,
            name="corridor", hoverinfo="skip",
        )
    )

    # desired trajectory
    dp_x = [s.get("desired_p", s["p"])[0] for s in samples]
    dp_y = [s.get("desired_p", s["p"])[1] for s in samples]
    dp_z = [s.get("desired_p", s["p"])[2] for s in samples]
    fig.add_trace(
        go.Scatter3d(
            x=dp_x, y=dp_y, z=dp_z,
            mode="lines",
            line=dict(color="#888", width=4, dash="dot"),
            name="MINCO commanded",
            opacity=0.6,
        )
    )

    # actual trajectory
    p_x = [s["p"][0] for s in samples]
    p_y = [s["p"][1] for s in samples]
    p_z = [s["p"][2] for s in samples]
    fig.add_trace(
        go.Scatter3d(
            x=p_x, y=p_y, z=p_z,
            mode="lines",
            line=dict(color="#d62728", width=5),
            name="actual path",
        )
    )

    # start/goal markers
    fig.add_trace(
        go.Scatter3d(
            x=[start[0]], y=[start[1]], z=[start[2]],
            mode="markers+text",
            marker=dict(size=10, color="#2ca02c", symbol="diamond"),
            text=["start"], textposition="top center",
            name="start",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[goal[0]], y=[goal[1]], z=[goal[2]],
            mode="markers+text",
            marker=dict(size=10, color="#d62728", symbol="x"),
            text=["goal"], textposition="top center",
            name="goal",
        )
    )

    # animated marker + wind vector
    drone_idx = len(fig.data)
    fig.add_trace(
        go.Scatter3d(
            x=[p_x[0]], y=[p_y[0]], z=[p_z[0]],
            mode="markers",
            marker=dict(size=8, color="#ff7f0e"),
            name="drone",
        )
    )
    wind_idx = len(fig.data)
    fig.add_trace(
        go.Scatter3d(
            x=[], y=[], z=[],
            mode="lines",
            line=dict(color="#9467bd", width=4),
            name="wind force",
        )
    )

    # RTL trigger marker (static — appears at the trigger position if any)
    if rtl_t is not None and np.isfinite(float(rtl_t)):
        # find sample closest to rtl_t
        idx = int(np.argmin([abs(s["t"] - float(rtl_t)) for s in samples]))
        fig.add_trace(
            go.Scatter3d(
                x=[samples[idx]["p"][0]],
                y=[samples[idx]["p"][1]],
                z=[samples[idx]["p"][2]],
                mode="markers+text",
                marker=dict(size=12, color="#9467bd", symbol="diamond-open",
                            line=dict(width=3, color="#9467bd")),
                text=[f"RTL @ t={float(rtl_t):.1f}s"],
                textposition="bottom center",
                name="RTL trigger",
            )
        )

    # frames — show drone + wind vector
    frames = []
    wind_scale = 0.5  # m per (m/s²); arbitrary visual scale
    for j, s in enumerate(samples):
        p = s["p"]
        w = s.get("wind", [0, 0, 0])
        wx = [p[0], p[0] + w[0] * wind_scale, None]
        wy = [p[1], p[1] + w[1] * wind_scale, None]
        wz = [p[2], p[2] + w[2] * wind_scale, None]
        frames.append(
            go.Frame(
                name=str(j),
                data=[
                    go.Scatter3d(
                        x=[p[0]], y=[p[1]], z=[p[2]],
                        mode="markers",
                        marker=dict(size=8, color="#ff7f0e"),
                    ),
                    go.Scatter3d(
                        x=wx, y=wy, z=wz,
                        mode="lines",
                        line=dict(color="#9467bd", width=4),
                    ),
                ],
                traces=[drone_idx, wind_idx],
            )
        )

    fig.frames = frames
    rtl_str = (
        f"RTL @ {float(rtl_t):.1f}s"
        if rtl_t is not None and np.isfinite(float(rtl_t))
        else "no RTL"
    )
    fig.update_layout(
        title=(
            f"Rig 6 — disturbance / {record.get('scenario', '?')}  "
            f"(track_max={record.get('tracking_error_max_m', 0):.3f} m, "
            f"vf_med={record.get('depth_valid_fraction_mean', 0):.3f}, {rtl_str})"
        ),
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
            aspectmode="data",
        ),
        height=800,
        margin=dict(l=0, r=0, b=80, t=50),
        sliders=_make_slider(len(frames), times),
        updatemenus=_make_play_pause_buttons(),
    )
    _save_html(fig, out_html)


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------


_VIZ_FNS = {
    "rig1": viz_rig1,
    "rig2": viz_rig2,
    "rig3": viz_rig3,
    "rig4": viz_rig4,
    "rig5": viz_rig5,
    "rig6": viz_rig6,
}
