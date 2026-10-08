"""Numerical experiments and plots for Figures 4-7 of the paper."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from control import (
    build_mesh_case,
    build_mesh_lqr_model,
    closed_loop_tail_cost,
    compute_sls_residual,
    dense_lqr_responses,
    fast_block_column_l1_norm,
    fast_block_row_l1_norm,
    fast_conic_truncate_response,
    fast_truncation_screening_metrics,
    h2_cost,
    implementation_complexity,
    lqr_fir_true_response_prefix_costs,
    residual_schur_bound,
    solve_lqr,
    weighted_error_h2,
)


@dataclass(frozen=True)
class MeshArchitectureScreenConfig:
    output_dir: str | Path = "results/screen"
    side: int = 10
    edge_weight: float = 0.05
    eta_power: int = -4
    kappas: tuple[int, ...] = tuple(range(0, 19))
    kappa_bars: tuple[int, ...] = tuple(range(0, 19))
    time_horizons: tuple[int, ...] = (
        5,
        10,
        15,
        20,
        25,
        30,
        40,
        50,
        60,
        80,
        100,
        120,
        160,
        200,
        240,
        280,
        320,
        360,
        400,
    )
    response_horizon: int = 560
    dt: float = 1.0
    r_dyn: int = 1


@dataclass(frozen=True)
class MeshArchitectureScreenSummary:
    rows: list[dict[str, float | int | str]]
    result_paths: list[Path]


def run_mesh_architecture_screen(
    config: MeshArchitectureScreenConfig,
) -> MeshArchitectureScreenSummary:
    output_dir = Path(config.output_dir)
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    case = build_mesh_case(side=config.side, edge_weight=config.edge_weight)
    eta = 2.0**config.eta_power
    model = build_mesh_lqr_model(case, eta=eta, dt=config.dt)
    solution = solve_lqr(model)
    closed_loop = model.A - model.B @ solution.K
    dense = dense_lqr_responses(
        model.A,
        model.B,
        solution.K,
        horizon=config.response_horizon,
    )
    central_cost = float(np.trace(solution.P) / model.bus_count)
    distance_profile = _weighted_distance_profile(
        dense.phi_x,
        dense.phi_u,
        distances=model.distances,
        q_weights=np.diag(model.Q),
        r_weights=np.diag(model.R),
        bus_count=model.bus_count,
        max_time_horizon=max(config.time_horizons),
    )
    temporal_tail_by_T = {
        time_horizon: closed_loop_tail_cost(
            closed_loop,
            solution.P,
            start_power=time_horizon + 1,
            normalize_by=model.bus_count,
        )
        for time_horizon in config.time_horizons
    }

    rows: list[dict[str, float | int | str]] = []
    for time_horizon in sorted(config.time_horizons):
        for kappa in sorted(config.kappas):
            for kappa_bar in sorted(config.kappa_bars):
                if kappa_bar < kappa:
                    continue
                spatial_tail = _saturated_spatial_tail(
                    distance_profile,
                    kappa=kappa,
                    kappa_bar=kappa_bar,
                    time_horizon=time_horizon,
                    r_dyn=config.r_dyn,
                )
                temporal_tail = temporal_tail_by_T[time_horizon]
                total_tail = spatial_tail + temporal_tail
                complexity = implementation_complexity(
                    model.distances,
                    kappa=kappa,
                    kappa_bar=kappa_bar,
                    time_horizon=time_horizon,
                    r_dyn=config.r_dyn,
                )
                rows.append(
                    {
                        "case_name": "2d-mesh",
                        "side": config.side,
                        "node_count": case.node_count,
                        "edge_count": case.edge_count,
                        "diameter": case.diameter,
                        "edge_weight": config.edge_weight,
                        "eta_power": config.eta_power,
                        "eta": eta,
                        "kappa": kappa,
                        "kappa_bar": kappa_bar,
                        "time_horizon": time_horizon,
                        "response_horizon": config.response_horizon,
                        "spatial_tail_cost": spatial_tail,
                        "temporal_tail_cost_infinite": temporal_tail,
                        "tail_cost_proxy": total_tail,
                        "relative_spatial_tail_proxy": spatial_tail / central_cost,
                        "relative_temporal_tail_proxy": temporal_tail / central_cost,
                        "relative_tail_proxy": total_tail / central_cost,
                        "squared_relative_tail_proxy": (total_tail / central_cost) ** 2,
                        "central_cost": central_cost,
                        "central_spectral_radius": solution.spectral_radius,
                        "zero_delay_fan_in": complexity.average_fan_in,
                        "max_communication_footprint": complexity.maximum_footprint,
                        "average_communication_footprint": complexity.average_footprint,
                        "implementation_taps_per_controller": (
                            complexity.average_taps_per_controller
                        ),
                        "maximum_taps_per_controller": (
                            complexity.maximum_taps_per_controller
                        ),
                        "architecture_cost": complexity.total_tap_blocks,
                    }
                )

    front = _pareto_front(rows, y_key="relative_tail_proxy")
    front_keys = {
        (
            int(row["kappa"]),
            int(row["kappa_bar"]),
            int(row["time_horizon"]),
        )
        for row in front
    }
    annotated_rows = [
        {
            **row,
            "proxy_pareto": int(
                (
                    int(row["kappa"]),
                    int(row["kappa_bar"]),
                    int(row["time_horizon"]),
                )
                in front_keys
            ),
        }
        for row in rows
    ]

    screen_path = results_dir / "mesh-architecture-screen.csv"
    front_path = results_dir / "mesh-architecture-proxy-pareto.csv"
    _write_rows(screen_path, annotated_rows)
    _write_rows(front_path, front)
    return MeshArchitectureScreenSummary(
        rows=annotated_rows,
        result_paths=[screen_path, front_path],
    )


def _weighted_distance_profile(
    phi_x: Sequence[np.ndarray],
    phi_u: Sequence[np.ndarray],
    *,
    distances: np.ndarray,
    q_weights: np.ndarray,
    r_weights: np.ndarray,
    bus_count: int,
    max_time_horizon: int,
    state_block_size: int = 2,
    input_block_size: int = 1,
) -> np.ndarray:
    max_tau = min(len(phi_x) - 1, max_time_horizon + 1)
    diameter = int(np.max(distances))
    profile = np.zeros((max_tau + 1, diameter + 1), dtype=float)
    q_blocks = np.asarray(q_weights, dtype=float).reshape(bus_count, state_block_size)
    r_blocks = np.asarray(r_weights, dtype=float).reshape(bus_count, input_block_size)
    distance_masks = [distances == distance for distance in range(diameter + 1)]

    for tau in range(1, max_tau + 1):
        state_blocks = phi_x[tau].reshape(
            bus_count,
            state_block_size,
            bus_count,
            state_block_size,
        )
        input_blocks = phi_u[tau].reshape(
            bus_count,
            input_block_size,
            bus_count,
            state_block_size,
        )
        state_weighted = (np.square(state_blocks) * q_blocks[:, :, None, None]).sum(
            axis=(1, 3)
        )
        input_weighted = (np.square(input_blocks) * r_blocks[:, :, None, None]).sum(
            axis=(1, 3)
        )
        weighted = (state_weighted + input_weighted) / bus_count
        for distance, mask in enumerate(distance_masks):
            profile[tau, distance] = float(weighted[mask].sum())
    return profile


def _saturated_spatial_tail(
    distance_profile: np.ndarray,
    *,
    kappa: int,
    kappa_bar: int,
    time_horizon: int,
    r_dyn: int,
) -> float:
    tail = 0.0
    max_tau = min(distance_profile.shape[0] - 1, time_horizon + 1)
    diameter = distance_profile.shape[1] - 1
    for tau in range(1, max_tau + 1):
        physical_time = tau - 1
        radius = min(kappa + r_dyn * physical_time, kappa_bar, diameter)
        if radius < diameter:
            tail += float(distance_profile[tau, radius + 1 :].sum())
    return tail


def _pareto_front(
    rows: Sequence[dict[str, float | int | str]],
    *,
    y_key: str,
) -> list[dict[str, float | int | str]]:
    finite_rows = [
        row
        for row in rows
        if _is_positive(row.get("architecture_cost")) and _is_positive(row.get(y_key))
    ]
    ordered = sorted(
        finite_rows,
        key=lambda row: (float(row["architecture_cost"]), float(row[y_key])),
    )
    front: list[dict[str, float | int | str]] = []
    best_y = math.inf
    for row in ordered:
        value = float(row[y_key])
        if value < best_y:
            front.append(row)
            best_y = value
    return front


def _write_rows(path: Path, rows: Sequence[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _is_positive(value: object) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and numeric > 0.0


LABEL_SIZE = 8.8

TICK_SIZE = 8.0

LEGEND_SIZE = 8.0

MARKERS = ("o", "s", "^", "D", "v")

LINESTYLES = ("-", "--", "-.", ":")

GRID_COLOR = "#DCE6E8"

SPINE_COLOR = "#8DA0A6"

PANEL_PALETTES = {
    "halo": ("#8BC9C0", "#3F9E91", "#176B61", "#0F4D47"),
    "memory": ("#B6C4E3", "#667DB8", "#374F9A", "#223066"),
    "footprint": ("#E0AE92", "#B86F52", "#7C3E29", "#522719"),
}


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Times",
                "Nimbus Roman No9 L",
                "STIXGeneral",
            ],
            "font.size": 8.0,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _series_groups(
    rows: Sequence[dict[str, str]],
    *,
    slice_name: str,
) -> list[tuple[str, list[dict[str, str]]]]:
    grouped: dict[tuple[int, str], list[dict[str, str]]] = {}
    for row in rows:
        if row.get("slice") != slice_name:
            continue
        key = (int(float(row["series_order"])), row.get("series_label", ""))
        grouped.setdefault(key, []).append(row)
    output: list[tuple[str, list[dict[str, str]]]] = []
    for (_order, label), series in sorted(grouped.items(), key=lambda item: item[0]):
        output.append((label, sorted(series, key=lambda row: float(row["value"]))))
    return output


def _positive_array(values: np.ndarray) -> np.ndarray:
    finite_positive = values[np.isfinite(values) & (values > 0.0)]
    floor = (
        max(float(np.min(finite_positive)) * 0.5, 1.0e-16)
        if finite_positive.size
        else 1.0e-16
    )
    return np.where(np.isfinite(values) & (values > 0.0), values, floor)


def _add_log_linear_fit(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    start_index: int,
    color: str,
) -> None:
    if x.size <= start_index + 1:
        return
    fit_x = x[start_index:]
    fit_y = y[start_index:]
    valid = np.isfinite(fit_x) & np.isfinite(fit_y) & (fit_y > 0.0)
    if np.count_nonzero(valid) < 2:
        return
    slope, intercept = np.polyfit(fit_x[valid], np.log(fit_y[valid]), 1)
    ax.semilogy(
        fit_x[valid],
        np.exp(intercept + slope * fit_x[valid]),
        "--",
        color=color,
        linewidth=0.82,
        alpha=0.55,
        label="_nolegend_",
    )


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(True, which="major", color=GRID_COLOR, linewidth=0.56, alpha=0.82)
    ax.grid(True, which="minor", color=GRID_COLOR, linewidth=0.34, alpha=0.45)
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)
        spine.set_linewidth(0.66)
    ax.tick_params(labelsize=TICK_SIZE, width=0.62, colors="#25323A")
    ax.xaxis.label.set_size(LABEL_SIZE)
    ax.yaxis.label.set_size(LABEL_SIZE)


def plot_parameter_sweep_matrix(
    tail_rows: Sequence[dict[str, str]],
    residual_rows: Sequence[dict[str, str]],
    exact_rows: Sequence[dict[str, str]],
    output_stem: Path,
) -> list[Path]:
    """Plot the three resource sweeps for the three stages of the analysis."""

    _configure_matplotlib()
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(7.15, 5.65),
        dpi=360,
        constrained_layout=True,
    )
    row_specs = (
        (tail_rows, "tail_for_plot", "response tail\nabove minimum"),
        (residual_rows, "residual_excess", "residual bound\nabove minimum"),
        (exact_rows, "excess_gap", "exact SLS gap\nabove minimum"),
    )
    column_specs = (
        ("halo", r"varying immediate radius $\kappa$", r"$\kappa$"),
        ("memory", r"varying memory horizon $T$", r"$T$"),
        ("footprint", r"varying maximum footprint $\bar\kappa$", r"$\bar\kappa$"),
    )

    for row_index, (rows, metric, ylabel) in enumerate(row_specs):
        for column_index, (slice_name, title, xlabel) in enumerate(column_specs):
            ax = axes[row_index, column_index]
            groups = _series_groups(rows, slice_name=slice_name)
            palette = PANEL_PALETTES[slice_name]
            for series_index, (label, series) in enumerate(groups):
                x = np.asarray([float(row["value"]) for row in series], dtype=float)
                raw_y = np.asarray([float(row[metric]) for row in series], dtype=float)
                valid = np.isfinite(raw_y) & (raw_y > 0.0)
                if not np.any(valid):
                    continue
                y = _positive_array(raw_y[valid])
                color = palette[series_index % len(palette)]
                if row_index == 2 and slice_name == "memory":
                    source_count = int(float(series[0].get("source_count", "0")))
                    label = rf"all sources, $n_{{\rm src}}={source_count}$"
                ax.semilogy(
                    x[valid],
                    y,
                    marker=MARKERS[series_index % len(MARKERS)],
                    linestyle=LINESTYLES[series_index % len(LINESTYLES)],
                    color=color,
                    linewidth=1.38,
                    markersize=3.1,
                    label=label,
                )
                start_index = 1 if slice_name == "halo" and row_index < 2 else 0
                _add_log_linear_fit(
                    ax,
                    x[valid],
                    y,
                    start_index=min(start_index, max(0, y.size - 2)),
                    color=color,
                )

            if row_index == 0:
                ax.set_title(title, fontsize=LABEL_SIZE, pad=3.0)
            if column_index == 0:
                ax.set_ylabel(ylabel)
            if row_index == 2:
                ax.set_xlabel(xlabel)

            show_legend = row_index == 0 or row_index == 2
            if show_legend:
                legend_title = (
                    r"$(\kappa,\bar\kappa)$"
                    if row_index == 0 and slice_name == "memory"
                    else None
                )
                legend = ax.legend(
                    frameon=False,
                    fontsize=LEGEND_SIZE,
                    loc="best",
                    handlelength=1.55,
                    title=legend_title,
                )
                if legend_title is not None:
                    legend.get_title().set_fontsize(LEGEND_SIZE)
            _style_axes(ax)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [output_stem.with_suffix(".pdf"), output_stem.with_suffix(".png")]
    for path in paths:
        fig.savefig(path, dpi=360)
    plt.close(fig)
    return paths


FIT_COLOR = "#25323A"


SINGLE_COLUMN_FIGSIZE = (3.36, 2.18)


@dataclass(frozen=True)
class DecayFigureConfig:
    side: int = 10
    edge_weight: float = 0.05
    eta_power: int = -4
    dt: float = 1.0
    source_bus: int | None = 55
    max_time: int = 80
    response_horizon: int = 180
    r_dyn: int = 1


def _compute_single_source_raw_and_shell_rows(
    config: DecayFigureConfig,
) -> tuple[list[dict[str, float | int | str]], list[dict[str, float | int | str]]]:
    case = build_mesh_case(side=config.side, edge_weight=config.edge_weight)
    source_bus = _resolve_source_bus(config.source_bus, side=config.side)
    model = build_mesh_lqr_model(
        case,
        eta=2.0**config.eta_power,
        dt=config.dt,
    )
    solution = solve_lqr(model)
    response = dense_lqr_responses(
        model.A,
        model.B,
        solution.K,
        horizon=config.response_horizon,
    )

    bus_count = int(model.bus_count)
    state_block_size = int(model.state_dim // bus_count)
    input_block_size = int(model.input_dim // bus_count)
    source_x = _decay_block_slice(source_bus, state_block_size)
    max_time = min(config.max_time, response.horizon - 1)
    raw_rows: list[dict[str, float | int | str]] = []
    shell_accumulator: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for tau in range(1, max_time + 2):
        physical_time = tau - 1
        phi_x = response.phi_x[tau]
        phi_u = response.phi_u[tau]
        for target_bus in range(bus_count):
            target_x = _decay_block_slice(target_bus, state_block_size)
            target_u = _decay_block_slice(target_bus, input_block_size)
            distance = int(model.distances[target_bus, source_bus])
            state_norm = float(np.linalg.norm(phi_x[target_x, source_x], ord="fro"))
            input_norm = float(np.linalg.norm(phi_u[target_u, source_x], ord="fro"))
            row_index, col_index = divmod(target_bus, config.side)
            raw_rows.append(
                {
                    "source_bus": source_bus,
                    "target_bus": target_bus,
                    "target_row": row_index,
                    "target_col": col_index,
                    "physical_time": physical_time,
                    "distance": distance,
                    "state_norm": state_norm,
                    "input_norm": input_norm,
                    "sum_norm": state_norm + input_norm,
                    "max_norm": max(state_norm, input_norm),
                }
            )
            shell_accumulator.setdefault((physical_time, distance), []).append(
                (state_norm, input_norm)
            )

    shell_rows: list[dict[str, float | int | str]] = []
    for (physical_time, distance), values in sorted(shell_accumulator.items()):
        state = np.asarray([value[0] for value in values], dtype=float)
        control = np.asarray([value[1] for value in values], dtype=float)
        shell_rows.append(
            {
                "source_bus": source_bus,
                "physical_time": physical_time,
                "distance": distance,
                "sample_count": int(state.size),
                "state_max": float(np.max(state)),
                "state_mean": float(np.mean(state)),
                "input_max": float(np.max(control)),
                "input_mean": float(np.mean(control)),
                "combined_max": float(max(np.max(state), np.max(control))),
            }
        )
    return raw_rows, shell_rows


def _plot_pointwise_offcone_decay(
    rows: Sequence[dict[str, float | int | str]],
    output_stem: Path,
    *,
    r_dyn: int,
) -> list[Path]:
    point_excess: list[float] = []
    point_sum: list[float] = []
    envelope: dict[int, dict[str, float]] = {}
    for row in rows:
        time = _int(row, "physical_time")
        distance = _int(row, "distance")
        excess = max(distance - r_dyn * time, 0)
        if excess <= 0:
            continue
        state = _float(row, "state_norm")
        control = _float(row, "input_norm")
        response_sum = state + control
        if not math.isfinite(response_sum) or response_sum <= 0.0:
            continue
        point_excess.append(float(excess))
        point_sum.append(response_sum)
        entry = envelope.setdefault(excess, {"state": 0.0, "input": 0.0, "sum": 0.0})
        entry["state"] = max(entry["state"], state)
        entry["input"] = max(entry["input"], control)
        entry["sum"] = max(entry["sum"], response_sum)
    if not point_excess:
        return []

    x = np.asarray(point_excess, dtype=float)
    y = np.asarray(point_sum, dtype=float)
    envelope_x = np.asarray(sorted(envelope), dtype=float)
    envelope_sum = _decay_positive_array(
        [envelope[int(value)]["sum"] for value in envelope_x]
    )

    fig, ax = plt.subplots(
        figsize=SINGLE_COLUMN_FIGSIZE,
        dpi=360,
        constrained_layout=True,
    )
    ax.scatter(
        x,
        y,
        color="#6F96A6",
        s=14,
        alpha=0.42,
        edgecolor="none",
        label="individual response blocks",
    )
    ax.semilogy(
        envelope_x,
        envelope_sum,
        "-o",
        color=FIT_COLOR,
        linewidth=1.35,
        markersize=3.6,
        label="upper envelope",
    )
    _add_log_fit(
        ax,
        envelope_x,
        envelope_sum,
        start_index=0,
        label="linear fit on log scale",
    )
    ax.set_yscale("log")
    ax.set_xlabel(r"distance beyond cone $[d_{\mathcal{G}}(i,j)-t]_+$")
    ax.set_ylabel(r"$\|\Phi_x^\star[t]_{ij}\|+\|\Phi_u^\star[t]_{ij}\|$")
    ax.legend(frameon=False, fontsize=LEGEND_SIZE, loc="best")
    _style_axes(ax)
    return _save_figure(fig, output_stem)


def _direct_truncation_slice_rows(
    config: DecayFigureConfig,
) -> list[dict[str, float | int | str]]:
    templates = _direct_truncation_slice_templates()
    if not templates:
        return []

    case = build_mesh_case(side=config.side, edge_weight=config.edge_weight)
    model = build_mesh_lqr_model(
        case,
        eta=2.0**config.eta_power,
        dt=config.dt,
    )
    solution = solve_lqr(model)
    max_time_horizon = max(_int(row, "time_horizon") for row in templates)
    response_horizon = max(config.response_horizon, max_time_horizon + 160)
    dense = dense_lqr_responses(
        model.A,
        model.B,
        solution.K,
        horizon=response_horizon,
    )
    central_cost = float(np.trace(solution.P) / model.bus_count)
    dense_prefix_cost = h2_cost(
        dense,
        model.Q,
        model.R,
        normalize_by=model.bus_count,
    )

    metric_cache: dict[tuple[int, int, int], dict[str, float | int | str]] = {}
    output: list[dict[str, float | int | str]] = []
    for template in templates:
        kappa = _int(template, "kappa")
        kappa_bar = _int(template, "kappa_bar")
        time_horizon = _int(template, "time_horizon")
        key = (kappa, kappa_bar, time_horizon)
        if key not in metric_cache:
            metrics = fast_truncation_screening_metrics(
                dense,
                A=model.A,
                B=model.B,
                Q=model.Q,
                R=model.R,
                distances=model.distances,
                kappa=kappa,
                kappa_bar=kappa_bar,
                time_horizon=time_horizon,
                bus_count=model.bus_count,
                normalize_by=model.bus_count,
                state_block_size=2,
                input_block_size=1,
                r_dyn=config.r_dyn,
            )
            gamma = float(metrics.residual_schur_bound)
            metric_cache[key] = {
                "case_name": "2d-mesh",
                "side": config.side,
                "node_count": case.node_count,
                "edge_count": case.edge_count,
                "diameter": case.diameter,
                "edge_weight": config.edge_weight,
                "eta_power": config.eta_power,
                "eta": 2.0**config.eta_power,
                "dt": config.dt,
                "response_horizon": response_horizon,
                "r_dyn": config.r_dyn,
                "central_cost": central_cost,
                "dense_prefix_cost": dense_prefix_cost,
                "dense_relative_prefix_cost": dense_prefix_cost / central_cost,
                "residual_l1_row": float(metrics.residual_l1_row),
                "residual_l1_col": float(metrics.residual_l1_col),
                "residual_schur_bound": gamma,
                "direct_truncation_gamma": gamma,
                "direct_truncation_certified": int(
                    math.isfinite(gamma) and gamma < 1.0
                ),
                "direct_truncation_tau_min_0p1_certified": int(
                    math.isfinite(gamma) and gamma <= 0.9
                ),
                "tail_l1": float(metrics.tail_l1),
                "response_l1": float(metrics.response_l1),
                "direct_truncation_nominal_prefix_cost": (
                    float(metrics.nominal_prefix_cost)
                ),
                "direct_truncation_nominal_relative_cost": (
                    float(metrics.nominal_prefix_cost) / central_cost
                ),
            }
        output.append({**template, **metric_cache[key]})
    return _with_direct_residual_excess(output)


def _direct_truncation_slice_templates() -> list[dict[str, float | int | str]]:
    fixed_time = 160
    memory_times = (40, 60, 80, 100, 120, 160, 200, 240, 320, 400)
    output: list[dict[str, float | int | str]] = []
    for order, kappa_bar in enumerate((4, 8, 18)):
        for kappa in range(0, kappa_bar + 1):
            output.append(
                {
                    "slice": "halo",
                    "vary": "kappa",
                    "value": kappa,
                    "kappa": kappa,
                    "kappa_bar": kappa_bar,
                    "time_horizon": fixed_time,
                    "series_label": rf"$\bar\kappa={kappa_bar}$",
                    "series_order": order,
                }
            )
    for order, (kappa, kappa_bar) in enumerate(((0, 4), (2, 8), (4, 18))):
        for time_horizon in memory_times:
            output.append(
                {
                    "slice": "memory",
                    "vary": "time_horizon",
                    "value": time_horizon,
                    "kappa": kappa,
                    "kappa_bar": kappa_bar,
                    "time_horizon": time_horizon,
                    "series_label": rf"$({kappa},{kappa_bar})$",
                    "series_order": order,
                }
            )
    for order, kappa in enumerate((0, 2, 4)):
        for kappa_bar in range(kappa, 19):
            output.append(
                {
                    "slice": "footprint",
                    "vary": "kappa_bar",
                    "value": kappa_bar,
                    "kappa": kappa,
                    "kappa_bar": kappa_bar,
                    "time_horizon": fixed_time,
                    "series_label": rf"$\kappa={kappa}$",
                    "series_order": order,
                }
            )
    return output


def _with_direct_residual_excess(
    rows: Sequence[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    grouped: dict[tuple[str, int, str], list[dict[str, float | int | str]]] = {}
    for row in rows:
        key = (
            str(row.get("slice", "")),
            _int(row, "series_order"),
            str(row.get("series_label", "")),
        )
        grouped.setdefault(key, []).append(row)

    floors: dict[tuple[str, int, str], float] = {}
    for key, series in grouped.items():
        values = [
            _float(row, "direct_truncation_gamma")
            for row in series
            if math.isfinite(_float(row, "direct_truncation_gamma"))
        ]
        floors[key] = min(values) if values else 0.0

    output: list[dict[str, float | int | str]] = []
    for row in rows:
        key = (
            str(row.get("slice", "")),
            _int(row, "series_order"),
            str(row.get("series_label", "")),
        )
        gamma = _float(row, "direct_truncation_gamma")
        floor = floors[key]
        output.append(
            {
                **row,
                "residual_floor": floor,
                "residual_excess": max(gamma - floor, 0.0)
                if math.isfinite(gamma)
                else math.nan,
            }
        )
    return output


def _resolve_source_bus(source_bus: int | None, *, side: int) -> int:
    if source_bus is not None:
        return int(source_bus)
    center = side // 2
    return center * side + center


def _decay_block_slice(index: int, block_size: int) -> slice:
    start = int(index) * int(block_size)
    return slice(start, start + int(block_size))


def _add_log_fit(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    start_index: int,
    label: str = "log-linear fit",
) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size <= start_index + 1:
        return math.nan, math.nan
    fit_x = x[start_index:]
    fit_y = y[start_index:]
    valid = np.isfinite(fit_x) & np.isfinite(fit_y) & (fit_y > 0.0)
    if np.count_nonzero(valid) < 2:
        return math.nan, math.nan
    log_y = np.log(fit_y[valid])
    slope, intercept = np.polyfit(fit_x[valid], log_y, 1)
    pred = intercept + slope * fit_x[valid]
    ss_res = float(np.sum((log_y - pred) ** 2))
    ss_tot = float(np.sum((log_y - np.mean(log_y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else math.nan
    ax.semilogy(
        fit_x[valid],
        np.exp(pred),
        "--",
        color=FIT_COLOR,
        linewidth=0.95,
        alpha=0.72,
        label=label,
    )
    return float(slope), float(r2)


def _decay_positive_array(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    finite_positive = array[np.isfinite(array) & (array > 0.0)]
    floor = (
        max(float(np.min(finite_positive)) * 0.5, 1e-16)
        if finite_positive.size
        else 1e-16
    )
    return np.where(np.isfinite(array) & (array > 0.0), array, floor)


def _float(row: dict[str, float | int | str], key: str) -> float:
    try:
        return float(row.get(key, float("nan")))
    except (TypeError, ValueError):
        return float("nan")


def _int(row: dict[str, float | int | str], key: str) -> int:
    value = row.get(key, 0)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _save_figure(fig: plt.Figure, output_stem: Path) -> list[Path]:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = [output_stem.with_suffix(".pdf"), output_stem.with_suffix(".png")]
    for path in paths:
        fig.savefig(path, dpi=360, bbox_inches="tight")
    plt.close(fig)
    return paths


def _tail_slice_rows(
    rows: Sequence[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    output: list[dict[str, float | int | str]] = []
    specs = [
        ("halo", "kappa", {"kappa_bar": 4, "time_horizon": 80}, r"$\bar\kappa=4$", 0),
        ("halo", "kappa", {"kappa_bar": 8, "time_horizon": 80}, r"$\bar\kappa=8$", 1),
        ("halo", "kappa", {"kappa_bar": 18, "time_horizon": 80}, r"$\bar\kappa=18$", 2),
        ("memory", "time_horizon", {"kappa": 0, "kappa_bar": 4}, r"$(0,4)$", 0),
        ("memory", "time_horizon", {"kappa": 2, "kappa_bar": 8}, r"$(2,8)$", 1),
        ("memory", "time_horizon", {"kappa": 4, "kappa_bar": 18}, r"$(4,18)$", 2),
        ("footprint", "kappa_bar", {"kappa": 0, "time_horizon": 80}, r"$\kappa=0$", 0),
        ("footprint", "kappa_bar", {"kappa": 2, "time_horizon": 80}, r"$\kappa=2$", 1),
        ("footprint", "kappa_bar", {"kappa": 4, "time_horizon": 80}, r"$\kappa=4$", 2),
    ]
    for slice_name, vary, fixed, label, order in specs:
        output.extend(
            _parameter_slice_rows(
                rows,
                slice_name=slice_name,
                vary=vary,
                fixed=fixed,
                metric="relative_tail_proxy",
                floor_mode="subtract_last",
                series_label=label,
                series_order=order,
            )
        )
    return output


def _parameter_slice_rows(
    rows: Sequence[dict[str, float | int | str]],
    *,
    slice_name: str,
    vary: str,
    fixed: dict[str, int],
    metric: str,
    floor_mode: str,
    series_label: str,
    series_order: int,
) -> list[dict[str, float | int | str]]:
    selected = [
        row
        for row in rows
        if all(_int(row, key) == value for key, value in fixed.items())
    ]
    selected = sorted(selected, key=lambda row: _int(row, vary))
    if not selected:
        return []
    raw_values = np.asarray([_float(row, metric) for row in selected], dtype=float)
    floor = 0.0
    if floor_mode == "subtract_last":
        finite = raw_values[np.isfinite(raw_values)]
        floor = float(finite[-1]) if finite.size else 0.0
    output: list[dict[str, float | int | str]] = []
    for row, raw_value in zip(selected, raw_values):
        excess = max(float(raw_value) - floor, 0.0)
        output.append(
            {
                "slice": slice_name,
                "vary": vary,
                "value": _int(row, vary),
                "raw_tail": float(raw_value),
                "tail_for_plot": excess if floor_mode == "subtract_last" else raw_value,
                "floor": floor,
                "series_label": series_label,
                "series_order": series_order,
                "kappa": _int(row, "kappa"),
                "kappa_bar": _int(row, "kappa_bar"),
                "time_horizon": _int(row, "time_horizon"),
            }
        )
    return output


SPATIAL_COLOR = "#3F9E91"

FOOTPRINT_COLOR = "#B86F52"

FRONTIER_COLOR = "#25323A"


SINGLE_COLUMN_WIDTH = 3.36

PARETO_STACK_FIGSIZE = (SINGLE_COLUMN_WIDTH, 3.70)


def _plot_architecture_pareto_all_points(
    rows: Sequence[dict[str, float | int | str]],
    figures_dir: Path,
) -> list[Path]:
    finite_rows = _finite_gap_rows(rows)
    front = _screening_pareto_front(finite_rows)
    if not finite_rows or not front:
        return []

    fig, ax = plt.subplots(figsize=(4.05, 2.8), dpi=360)
    time_values = np.asarray([_int(row, "time_horizon") for row in finite_rows])
    norm = plt.Normalize(float(time_values.min()), float(time_values.max()))
    cmap = plt.colormaps["viridis"]

    for family, marker, label, size in [
        ("pure_cone", "o", r"$\bar\kappa=\kappa+v_{\rm cone}T$", 34),
        ("saturated", "s", r"saturated $\bar\kappa$", 42),
    ]:
        family_rows = [
            row for row in finite_rows if str(row.get("support_family")) == family
        ]
        if not family_rows:
            continue
        ax.scatter(
            [_tap_count_proxy(row) for row in family_rows],
            _screening_positive_array(
                [_float(row, "normalized_cost_gap") for row in family_rows]
            ),
            c=[_int(row, "time_horizon") for row in family_rows],
            cmap=cmap,
            norm=norm,
            marker=marker,
            s=size,
            edgecolor="#25323A",
            linewidth=0.35,
            alpha=0.72,
            label=label,
            zorder=3,
        )

    fx = np.asarray([_tap_count_proxy(row) for row in front], dtype=float)
    fy = _screening_positive_array(
        [_float(row, "normalized_cost_gap") for row in front]
    )
    order = np.argsort(fx)
    ax.plot(
        fx[order],
        fy[order],
        "-o",
        color=FRONTIER_COLOR,
        markerfacecolor=SPATIAL_COLOR,
        markeredgecolor="white",
        markeredgewidth=0.35,
        linewidth=1.55,
        markersize=3.8,
        label="Pareto frontier",
        zorder=5,
    )

    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    cbar = fig.colorbar(scalar, ax=ax, pad=0.018, fraction=0.055)
    cbar.set_label(r"FIR horizon $T$")
    cbar.ax.tick_params(labelsize=TICK_SIZE)
    cbar.ax.yaxis.label.set_size(LABEL_SIZE)

    ax.set_yscale("log")
    ax.set_xlabel(r"average tap count $C_{\rm tap}$")
    ax.set_ylabel(r"exact SLS performance loss")
    ax.legend(frameon=False, fontsize=LEGEND_SIZE, loc="best")
    _screening_style_axes(ax)
    return _save_figure(fig, figures_dir / "paper-mesh-architecture-pareto-all-points")


def _plot_architecture_screening_validation(
    proxy_rows: Sequence[dict[str, float | int | str]],
    exact_rows: Sequence[dict[str, float | int | str]],
    validation_rows: Sequence[dict[str, float | int | str]],
    figures_dir: Path,
) -> list[Path]:
    finite_proxy = _finite_proxy_rows(proxy_rows)
    proxy_front = _pareto_front_by_metric(finite_proxy, "relative_tail_proxy")
    finite_exact = _finite_gap_rows(exact_rows)
    infeasible_exact = _infeasible_exact_rows(exact_rows)
    exact_front = _screening_pareto_front(finite_exact)
    finite_validation = _finite_column_gap_rows(validation_rows)
    if not finite_proxy or not proxy_front or not finite_exact or not exact_front:
        return _plot_architecture_pareto_all_points(exact_rows, figures_dir)
    cost_values = np.asarray(
        [
            _tap_count_proxy(row)
            for row in finite_proxy + finite_exact + finite_validation
        ],
        dtype=float,
    )
    cost_values = cost_values[np.isfinite(cost_values) & (cost_values > 0)]
    cost_log_min = float(np.log10(cost_values.min()))
    cost_log_max = float(np.log10(cost_values.max()))
    cost_pad = 0.04 * (cost_log_max - cost_log_min)
    shared_cost_xlim = (
        10 ** (cost_log_min - cost_pad),
        10 ** (cost_log_max + cost_pad),
    )

    proxy_by_arch = {_proxy_architecture_key(row): row for row in finite_proxy}
    matched_exact = [
        (row, proxy_by_arch[_exact_proxy_key(row)])
        for row in finite_exact
        if _exact_proxy_key(row) in proxy_by_arch
    ]
    matched_validation = [
        (row, proxy_by_arch[_exact_proxy_key(row)])
        for row in finite_validation
        if _exact_proxy_key(row) in proxy_by_arch
    ]
    matched_infeasible = [
        (row, proxy_by_arch[_exact_proxy_key(row)])
        for row in infeasible_exact
        if _exact_proxy_key(row) in proxy_by_arch
    ]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=PARETO_STACK_FIGSIZE,
        dpi=360,
        constrained_layout=True,
        sharex=True,
    )
    ax = axes[0]
    ax.scatter(
        [_tap_count_proxy(row) for row in finite_proxy],
        _screening_positive_array(
            [_float(row, "relative_tail_proxy") for row in finite_proxy]
        ),
        s=5.5,
        color="#AAB9BF",
        alpha=0.22,
        linewidth=0.0,
        label="candidate control architectures",
    )
    px = np.asarray([_tap_count_proxy(row) for row in proxy_front])
    py = _screening_positive_array(
        [_float(row, "relative_tail_proxy") for row in proxy_front]
    )
    order = np.argsort(px)
    ax.plot(
        px[order],
        py[order],
        color=FRONTIER_COLOR,
        linewidth=1.55,
        label="resource versus tail frontier",
        zorder=4,
    )
    exact_screen_points = matched_exact + matched_validation
    if exact_screen_points:
        ax.scatter(
            [_tap_count_proxy(proxy) for _, proxy in exact_screen_points],
            _screening_positive_array(
                [
                    _float(proxy, "relative_tail_proxy")
                    for _, proxy in exact_screen_points
                ]
            ),
            s=22,
            facecolor="none",
            edgecolor=SPATIAL_COLOR,
            linewidth=0.9,
            label="tested by exact QP",
            zorder=5,
        )
    if matched_infeasible:
        ax.scatter(
            [_tap_count_proxy(proxy) for _, proxy in matched_infeasible],
            _screening_positive_array(
                [
                    _float(proxy, "relative_tail_proxy")
                    for _, proxy in matched_infeasible
                ]
            ),
            s=25,
            marker="x",
            color="#A94D4A",
            linewidth=0.95,
            label="infeasible exact QP",
            zorder=6,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*shared_cost_xlim)
    ax.set_xlabel("")
    ax.set_ylabel(r"normalized response tail $p_\theta$")
    ax.tick_params(labelbottom=False)
    ax.legend(frameon=False, fontsize=LEGEND_SIZE, loc="best")
    _screening_style_axes(ax)

    ax = axes[1]
    for family, marker, label, size, color in [
        (
            "pure_cone",
            "o",
            r"full response, $\bar\kappa=\kappa+v_{\rm cone}T$",
            26,
            "#6E5A8A",
        ),
        (
            "saturated",
            "s",
            r"full response, independent $\bar\kappa$",
            31,
            FOOTPRINT_COLOR,
        ),
    ]:
        family_rows = [
            row for row in finite_exact if str(row.get("support_family")) == family
        ]
        if not family_rows:
            continue
        ax.scatter(
            [_tap_count_proxy(row) for row in family_rows],
            _screening_positive_array(
                [_float(row, "normalized_cost_gap") for row in family_rows]
            ),
            color=color,
            marker=marker,
            s=size,
            edgecolor="#25323A",
            linewidth=0.42,
            alpha=0.8,
            label=label,
            zorder=3,
        )
    fx = np.asarray([_tap_count_proxy(row) for row in exact_front])
    fy = _screening_positive_array(
        [_float(row, "normalized_cost_gap") for row in exact_front]
    )
    order = np.argsort(fx)
    ax.plot(
        fx[order],
        fy[order],
        "-o",
        color=FRONTIER_COLOR,
        markerfacecolor=SPATIAL_COLOR,
        markeredgecolor="white",
        markeredgewidth=0.35,
        linewidth=1.65,
        markersize=3.8,
        label="exact empirical frontier",
        zorder=5,
    )
    if finite_validation:
        ax.scatter(
            [_tap_count_proxy(row) for row in finite_validation],
            _screening_positive_array(
                [_float(row, "feasible_column_cost_gap") for row in finite_validation]
            ),
            color=SPATIAL_COLOR,
            marker="D",
            s=35,
            edgecolor="#25323A",
            linewidth=0.45,
            alpha=0.92,
            label="selected disturbance columns",
            zorder=4,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*shared_cost_xlim)
    ax.set_xlabel(r"average tap count $C_{\rm tap}$")
    ax.set_ylabel(r"exact SLS performance loss")
    ax.legend(frameon=False, fontsize=LEGEND_SIZE, loc="best")
    _screening_style_axes(ax)

    return _save_figure(fig, figures_dir / "paper-mesh-architecture-pareto-all-points")


def _finite_gap_rows(
    rows: Sequence[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    finite = []
    for row in rows:
        gap = _float(row, "normalized_cost_gap")
        cost = _float(row, "architecture_cost")
        feasible = _float(row, "qp_feasible_fraction")
        if math.isfinite(gap) and gap > 0.0 and math.isfinite(cost) and feasible >= 1.0:
            finite.append(row)
    return finite


def _finite_proxy_rows(
    rows: Sequence[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    finite = []
    for row in rows:
        cost = _float(row, "architecture_cost")
        tail = _float(row, "relative_tail_proxy")
        if math.isfinite(cost) and cost > 0.0 and math.isfinite(tail) and tail > 0.0:
            finite.append(row)
    return finite


def _infeasible_exact_rows(
    rows: Sequence[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    infeasible = []
    for row in rows:
        cost = _float(row, "architecture_cost")
        status = str(row.get("qp_status", "")).lower()
        if not (math.isfinite(cost) and cost > 0.0):
            continue
        # A failed solve does not establish infeasibility; require an explicit
        # feasibility classification or an infeasibility certificate.
        classification = str(row.get("qp_feasibility", ""))
        if classification in {"infeasible_certificate", "infeasible_structural"} or (
            not classification and status in {"infeasible", "infeasible_inaccurate"}
        ):
            infeasible.append(row)
    return infeasible


def _finite_column_gap_rows(
    rows: Sequence[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    finite = []
    for row in rows:
        gap = _float(row, "feasible_column_cost_gap")
        cost = _float(row, "architecture_cost")
        feasible = _float(row, "qp_feasible_fraction")
        if math.isfinite(gap) and gap > 0.0 and math.isfinite(cost) and feasible >= 1.0:
            finite.append(row)
    return finite


def _screening_pareto_front(
    rows: Sequence[dict[str, float | int | str]],
) -> list[dict[str, float | int | str]]:
    ordered = sorted(
        _finite_gap_rows(rows),
        key=lambda row: (
            _float(row, "architecture_cost"),
            _float(row, "normalized_cost_gap"),
        ),
    )
    front: list[dict[str, float | int | str]] = []
    best_gap = math.inf
    for row in ordered:
        gap = _float(row, "normalized_cost_gap")
        if gap < best_gap:
            front.append(row)
            best_gap = gap
    return front


def _pareto_front_by_metric(
    rows: Sequence[dict[str, float | int | str]],
    metric: str,
) -> list[dict[str, float | int | str]]:
    ordered = sorted(
        rows,
        key=lambda row: (_float(row, "architecture_cost"), _float(row, metric)),
    )
    front: list[dict[str, float | int | str]] = []
    best_value = math.inf
    for row in ordered:
        value = _float(row, metric)
        if value < best_value:
            front.append(row)
            best_value = value
    return front


def _proxy_architecture_key(row: dict[str, float | int | str]) -> tuple[int, int, int]:
    return (_int(row, "kappa"), _int(row, "kappa_bar"), _int(row, "time_horizon"))


def _exact_proxy_key(row: dict[str, float | int | str]) -> tuple[int, int, int]:
    kappa_bar = str(row.get("kappa_bar", "none"))
    if kappa_bar.lower() == "none":
        kappa_bar_value = _int(row, "diameter")
        if kappa_bar_value <= 0:
            kappa_bar_value = 18
    else:
        kappa_bar_value = _int(row, "kappa_bar")
    return (_int(row, "kappa"), kappa_bar_value, _int(row, "time_horizon"))


def _screening_positive_array(values: Sequence[float]) -> np.ndarray:
    finite = np.asarray(
        [value for value in values if math.isfinite(value) and value > 0.0]
    )
    floor = 1e-16 if finite.size == 0 else max(float(finite.min()) * 0.5, 1e-16)
    return np.asarray(
        [value if math.isfinite(value) and value > 0.0 else floor for value in values],
        dtype=float,
    )


def _screening_style_axes(ax: plt.Axes) -> None:
    ax.grid(True, which="major", color=GRID_COLOR, linewidth=0.6, alpha=0.9)
    ax.grid(True, which="minor", color=GRID_COLOR, linewidth=0.38, alpha=0.5)
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)
        spine.set_linewidth(0.68)
    ax.tick_params(labelsize=TICK_SIZE, width=0.65, colors="#25323A")
    ax.xaxis.label.set_size(LABEL_SIZE)
    ax.yaxis.label.set_size(LABEL_SIZE)


def _tap_count_proxy(row: dict[str, float | int | str]) -> float:
    """Return the per-controller average tap count used by Eq. (C-tap-def)."""
    average = _float(row, "implementation_taps_per_controller")
    if math.isfinite(average) and average > 0.0:
        return average
    total = _float(row, "architecture_cost")
    node_count = _float(row, "node_count")
    if (
        math.isfinite(total)
        and total > 0.0
        and math.isfinite(node_count)
        and node_count > 0.0
    ):
        return total / node_count
    return total


@dataclass(frozen=True)
class MeshSizeScalingConfig:
    output_dir: str | Path = "results/scaling"
    sides: tuple[int, ...] = (5, 7, 10, 12)
    edge_weight: float = 0.05
    eta_power: int = -4
    dt: float = 1.0
    kappa: int = 4
    kappa_bar: int = 8
    time_horizon: int = 160
    response_horizon: int = 320
    actual_cost_horizon: int = 880
    actual_cost_check_horizon: int = 720
    r_dyn: int = 1


def run_mesh_size_scaling(
    config: MeshSizeScalingConfig,
) -> list[dict[str, float | int | str]]:
    """Evaluate one fixed local/FIR architecture across square-mesh sizes.

    The architecture parameters and all physical/model parameters remain fixed;
    only the side length changes.  The weighted omitted-response energy includes
    the analytically evaluated centralized tail beyond ``response_horizon``.
    """

    if config.kappa_bar < config.kappa:
        raise ValueError("kappa_bar must be at least kappa")
    if config.response_horizon <= config.time_horizon:
        raise ValueError("response_horizon must exceed time_horizon")
    if config.actual_cost_horizon <= config.response_horizon:
        raise ValueError("actual_cost_horizon must exceed response_horizon")
    if not (
        config.response_horizon
        < config.actual_cost_check_horizon
        < config.actual_cost_horizon
    ):
        raise ValueError(
            "actual_cost_check_horizon must lie between response_horizon "
            "and actual_cost_horizon"
        )

    output_dir = Path(config.output_dir)
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | int | str]] = []

    for side in config.sides:
        print(
            f"mesh size scaling side={side} N={side * side} "
            f"theta=({config.kappa},{config.kappa_bar},{config.time_horizon})",
            flush=True,
        )
        case = build_mesh_case(side=side, edge_weight=config.edge_weight)
        eta = 2.0**config.eta_power
        model = build_mesh_lqr_model(case, eta=eta, dt=config.dt)
        solution = solve_lqr(model)
        closed_loop = model.A - model.B @ solution.K
        dense = dense_lqr_responses(
            model.A,
            model.B,
            solution.K,
            horizon=config.response_horizon,
        )
        truncated = fast_conic_truncate_response(
            dense,
            distances=model.distances,
            kappa=config.kappa,
            kappa_bar=config.kappa_bar,
            time_horizon=config.time_horizon,
            state_block_size=2,
            input_block_size=1,
            r_dyn=config.r_dyn,
        )
        residual = compute_sls_residual(model.A, model.B, truncated)
        residual_row = fast_block_row_l1_norm(
            residual.delta,
            bus_count=model.bus_count,
            row_block_size=2,
            source_block_size=2,
        )
        residual_col = fast_block_column_l1_norm(
            residual.delta,
            bus_count=model.bus_count,
            column_block_size=2,
            row_block_size=2,
        )
        gamma = residual_schur_bound(residual_row, residual_col)
        central_cost = float(np.trace(solution.P) / model.bus_count)
        dense_prefix_cost = h2_cost(
            dense,
            model.Q,
            model.R,
            normalize_by=model.bus_count,
        )
        omitted_prefix_cost = weighted_error_h2(
            dense,
            truncated,
            model.Q,
            model.R,
            normalize_by=model.bus_count,
        )
        omitted_beyond_prefix = closed_loop_tail_cost(
            closed_loop,
            solution.P,
            start_power=config.response_horizon,
            normalize_by=model.bus_count,
        )
        omitted_cost = omitted_prefix_cost + omitted_beyond_prefix
        complexity = implementation_complexity(
            model.distances,
            kappa=config.kappa,
            kappa_bar=config.kappa_bar,
            time_horizon=config.time_horizon,
            r_dyn=config.r_dyn,
        )
        footprint_fraction = float(np.mean(model.distances <= config.kappa_bar))
        actual_prefix_costs = lqr_fir_true_response_prefix_costs(
            truncated,
            residual,
            q=model.Q,
            r=model.R,
            horizons=(
                config.actual_cost_check_horizon,
                config.actual_cost_horizon,
            ),
            normalize_by=model.bus_count,
            closed_loop=closed_loop,
            feedback_gain=solution.K,
        )
        actual_prefix_cost = actual_prefix_costs[config.actual_cost_horizon]
        actual_relative_cost = actual_prefix_cost / central_cost
        actual_check_relative_cost = (
            actual_prefix_costs[config.actual_cost_check_horizon] / central_cost
        )
        central_tail_beyond_actual = closed_loop_tail_cost(
            closed_loop,
            solution.P,
            start_power=config.actual_cost_horizon,
            normalize_by=model.bus_count,
        )
        rows.append(
            {
                "case_name": "2d-mesh-size-scaling",
                "side": side,
                "node_count": case.node_count,
                "edge_count": case.edge_count,
                "diameter": case.diameter,
                "edge_weight": config.edge_weight,
                "eta_power": config.eta_power,
                "eta": eta,
                "dt": config.dt,
                "kappa": config.kappa,
                "kappa_bar": config.kappa_bar,
                "time_horizon": config.time_horizon,
                "response_horizon": config.response_horizon,
                "actual_cost_horizon": config.actual_cost_horizon,
                "actual_cost_check_horizon": config.actual_cost_check_horizon,
                "r_dyn": config.r_dyn,
                "footprint_fraction": footprint_fraction,
                "average_immediate_halo": complexity.average_fan_in,
                "maximum_immediate_halo": complexity.maximum_fan_in,
                "average_maximum_footprint": complexity.average_footprint,
                "maximum_spatial_footprint": complexity.maximum_footprint,
                "average_tap_count": complexity.average_taps_per_controller,
                "maximum_tap_count": complexity.maximum_taps_per_controller,
                "central_cost": central_cost,
                "dense_prefix_cost": dense_prefix_cost,
                "dense_relative_prefix_cost": dense_prefix_cost / central_cost,
                "relative_weighted_omitted_cost": omitted_cost / central_cost,
                "relative_weighted_omitted_prefix_cost": (
                    omitted_prefix_cost / central_cost
                ),
                "relative_weighted_beyond_prefix_cost": (
                    omitted_beyond_prefix / central_cost
                ),
                "residual_l1_row": residual_row,
                "residual_l1_col": residual_col,
                "residual_schur_bound": gamma,
                "direct_truncation_certified": int(np.isfinite(gamma) and gamma < 1.0),
                "actual_achieved_prefix_cost": actual_prefix_cost,
                "actual_achieved_relative_cost": actual_relative_cost,
                "actual_achieved_normalized_cost_gap": actual_relative_cost - 1.0,
                "actual_cost_horizon_change": (
                    actual_relative_cost - actual_check_relative_cost
                ),
                "central_relative_tail_beyond_actual_horizon": (
                    central_tail_beyond_actual / central_cost
                ),
                "central_spectral_radius": solution.spectral_radius,
            }
        )

    path = results_dir / "2d-mesh-size-scaling.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    _plot_mesh_size_scaling(
        rows,
        output_dir / "figures" / "2d-mesh-size-scaling",
        config=config,
    )
    return rows


def _plot_mesh_size_scaling(
    rows: Sequence[dict[str, float | int | str]],
    output_stem: Path,
    *,
    config: MeshSizeScalingConfig,
    column_layout: bool = False,
) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    node_count = np.asarray([float(row["node_count"]) for row in rows])
    actual_gap = np.asarray(
        [float(row["actual_achieved_normalized_cost_gap"]) for row in rows]
    )
    omitted_energy = np.asarray(
        [float(row["relative_weighted_omitted_cost"]) for row in rows]
    )
    gamma = np.asarray([float(row["residual_schur_bound"]) for row in rows])
    halo = np.asarray([float(row["average_immediate_halo"]) for row in rows])
    footprint = np.asarray([float(row["average_maximum_footprint"]) for row in rows])
    taps = np.asarray([float(row["average_tap_count"]) for row in rows])
    halo_bound = _two_dimensional_grid_ball_bound(config.kappa)
    footprint_bound = _two_dimensional_grid_ball_bound(config.kappa_bar)
    tap_bound = float(
        sum(
            _two_dimensional_grid_ball_bound(
                min(config.kappa + config.r_dyn * lag, config.kappa_bar)
            )
            for lag in range(config.time_horizon + 1)
        )
    )

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
            "font.size": 8.0,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(
        3 if column_layout else 1,
        1 if column_layout else 3,
        figsize=(3.36, 3.85) if column_layout else (7.15, 2.18),
        dpi=360,
        constrained_layout=True,
        sharex=column_layout,
    )

    axes[0].semilogy(
        node_count,
        np.maximum(actual_gap, 1.0e-16),
        "-o",
        color="#214F80",
        linewidth=1.35,
        markersize=3.6,
        label="achieved LQR cost gap",
    )
    axes[0].semilogy(
        node_count,
        omitted_energy,
        "--s",
        color="#6A9FB5",
        linewidth=1.15,
        markersize=3.2,
        label="relative omitted response energy",
    )
    if not column_layout:
        axes[0].set_xlabel(r"network size $N$")
    axes[0].set_ylabel("normalized LQR quantity")
    axes[0].legend(frameon=False, fontsize=8.0, loc="best")

    axes[1].plot(
        node_count,
        gamma,
        "-o",
        color="#B45532",
        linewidth=1.35,
        markersize=3.6,
    )
    axes[1].axhline(
        1.0,
        color="#5E6A70",
        linestyle="--",
        linewidth=1.0,
        label="small gain threshold",
    )
    if not column_layout:
        axes[1].set_xlabel(r"network size $N$")
    axes[1].set_ylabel(r"residual bound $\gamma_\theta$")
    axes[1].legend(frameon=False, fontsize=8.0, loc="best")

    axes[2].semilogy(
        node_count,
        halo,
        "-o",
        color="#28766A",
        linewidth=1.2,
        markersize=3.4,
        label=(
            r"$C_{\rm halo}=C_{\rm foot}$"
            if config.kappa == config.kappa_bar
            else r"$C_{\rm halo}$"
        ),
    )
    if config.kappa != config.kappa_bar:
        axes[2].semilogy(
            node_count,
            footprint,
            "-s",
            color="#4B9C86",
            linewidth=1.2,
            markersize=3.2,
            label=r"$C_{\rm foot}$",
        )
    axes[2].semilogy(
        node_count,
        taps,
        "-^",
        color="#7A6B3A",
        linewidth=1.2,
        markersize=3.4,
        label=r"$C_{\rm tap}$",
    )
    axes[2].axhline(
        halo_bound,
        color="#28766A",
        linestyle=":",
        linewidth=1.0,
        label="uniform spatial bound",
    )
    if config.kappa != config.kappa_bar:
        axes[2].axhline(
            footprint_bound,
            color="#4B9C86",
            linestyle=":",
            linewidth=1.0,
        )
    axes[2].axhline(
        tap_bound,
        color="#7A6B3A",
        linestyle=":",
        linewidth=1.0,
        label="uniform tap bound",
    )
    axes[2].set_xlabel(r"network size $N$")
    axes[2].set_ylabel("average block count per node")
    axes[2].legend(frameon=False, fontsize=8.0, loc="best")

    for axis in axes:
        axis.set_xticks(node_count)
        axis.grid(True, which="major", color="#D8E0E5", linewidth=0.45)
        axis.grid(True, which="minor", color="#EEF2F4", linewidth=0.3)
        for spine in axis.spines.values():
            spine.set_color("#71808A")
            spine.set_linewidth(0.66)
        axis.tick_params(labelsize=8.0, width=0.62, colors="#25323A")

    if column_layout:
        for axis in axes[:-1]:
            axis.tick_params(axis="x", which="both", labelbottom=False)

    for suffix in ("pdf", "png"):
        fig.savefig(output_stem.with_suffix(f".{suffix}"), bbox_inches="tight")
    plt.close(fig)
    if not column_layout:
        _plot_mesh_size_scaling(
            rows,
            output_stem.with_name(f"{output_stem.name}-column"),
            config=config,
            column_layout=True,
        )


def _two_dimensional_grid_ball_bound(radius: int) -> float:
    """Maximum number of nodes in a Manhattan ball of the given radius."""

    return float(1 + 2 * radius * (radius + 1))
