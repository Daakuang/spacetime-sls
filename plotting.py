"""The four numerical figures in the paper."""

import math

import matplotlib.pyplot as plt
import numpy as np

LABEL_SIZE, TICK_SIZE, LEGEND_SIZE = 8.8, 8.0, 8.0
MARKERS = ("o", "s", "^", "D", "v")
LINESTYLES = ("-", "--", "-.", ":")
GRID_COLOR, SPINE_COLOR = "#DCE6E8", "#8DA0A6"
FIT_COLOR = FRONTIER_COLOR = "#25323A"
SPATIAL_COLOR, FOOTPRINT_COLOR = "#3F9E91", "#B86F52"
SINGLE_COLUMN_FIGSIZE, PARETO_STACK_FIGSIZE = (3.36, 2.18), (3.36, 3.70)
PANEL_PALETTES = {
    "halo": ("#8BC9C0", "#3F9E91", "#176B61", "#0F4D47"),
    "memory": ("#B6C4E3", "#667DB8", "#374F9A", "#223066"),
    "footprint": ("#E0AE92", "#B86F52", "#7C3E29", "#522719"),
}


def positive(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values) & (values > 0)]
    floor = max(float(finite.min()) * 0.5, 1e-16) if finite.size else 1e-16
    return np.where(np.isfinite(values) & (values > 0), values, floor)


def log_fit(
    ax, x, y, *, start_index, color=FIT_COLOR, label="_nolegend_", envelope=False
):
    x, y = np.asarray(x)[start_index:], np.asarray(y)[start_index:]
    valid = np.isfinite(x) & np.isfinite(y) & (y > 0)
    if valid.sum() < 2:
        return
    x, y = x[valid], y[valid]
    slope, intercept = np.polyfit(x, np.log(y), 1)
    ax.semilogy(
        x,
        np.exp(intercept + slope * x),
        "--",
        color=color,
        linewidth=0.95 if envelope else 0.82,
        alpha=0.72 if envelope else 0.55,
        label=label,
    )


def style_axes(ax, screening=False):
    major, minor, alpha, minor_alpha, spine, tick = (
        (0.6, 0.38, 0.9, 0.5, 0.68, 0.65)
        if screening
        else (0.56, 0.34, 0.82, 0.45, 0.66, 0.62)
    )
    ax.grid(True, which="major", color=GRID_COLOR, linewidth=major, alpha=alpha)
    ax.grid(True, which="minor", color=GRID_COLOR, linewidth=minor, alpha=minor_alpha)
    for edge in ax.spines.values():
        edge.set_color(SPINE_COLOR)
        edge.set_linewidth(spine)
    ax.tick_params(labelsize=TICK_SIZE, width=tick, colors="#25323A")
    ax.xaxis.label.set_size(LABEL_SIZE)
    ax.yaxis.label.set_size(LABEL_SIZE)


def save_figure(fig, path, tight=True):
    for suffix in (".pdf", ".png"):
        fig.savefig(
            path.with_suffix(suffix), dpi=360, bbox_inches="tight" if tight else None
        )
    plt.close(fig)


def architecture_key(row):
    return tuple(int(row[key]) for key in ("kappa", "kappa_bar", "memory"))


def feasible_rows(rows):
    return [
        r
        for r in rows
        if r["status"] == "feasible"
        and math.isfinite(float(r["loss"]))
        and float(r["loss"]) > 0
    ]


def frontier(rows, metric):
    best, result = math.inf, []
    for row in sorted(rows, key=lambda r: (float(r["taps"]), float(r[metric]))):
        if float(row[metric]) < best:
            best = float(row[metric])
            result.append(row)
    return result


def configure():
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


def series_groups(rows, *, slice_name):
    grouped = {}
    for row in rows:
        if row["slice"] != slice_name:
            continue
        key = (int(float(row["series_order"])), row["series_label"])
        grouped.setdefault(key, []).append(row)
    output = []
    for (_order, label), series in sorted(grouped.items(), key=lambda item: item[0]):
        output.append((label, sorted(series, key=lambda row: float(row["value"]))))
    return output


def parameter_sweeps(tail_rows, residual_rows, exact_rows, output_stem):
    """Plot the three resource sweeps for the three stages of the analysis."""
    configure()
    fig, axes = plt.subplots(
        3, 3, figsize=(7.15, 5.65), dpi=360, constrained_layout=True
    )
    row_specs = (
        (tail_rows, "tail_excess", "response tail\nabove minimum"),
        (residual_rows, "gamma_excess", "residual bound\nabove minimum"),
        (exact_rows, "loss_excess", "exact SLS gap\nabove minimum"),
    )
    column_specs = (
        ("halo", "varying immediate radius $\\kappa$", "$\\kappa$"),
        ("memory", "varying memory horizon $T$", "$T$"),
        ("footprint", "varying maximum footprint $\\bar\\kappa$", "$\\bar\\kappa$"),
    )
    for row_index, (rows, metric, ylabel) in enumerate(row_specs):
        for column_index, (slice_name, title, xlabel) in enumerate(column_specs):
            ax = axes[row_index, column_index]
            groups = series_groups(rows, slice_name=slice_name)
            palette = PANEL_PALETTES[slice_name]
            for series_index, (label, series) in enumerate(groups):
                x = np.asarray([float(row["value"]) for row in series], dtype=float)
                raw_y = np.asarray([float(row[metric]) for row in series], dtype=float)
                valid = np.isfinite(raw_y) & (raw_y > 0.0)
                if not np.any(valid):
                    continue
                y = positive(raw_y[valid])
                color = palette[series_index % len(palette)]
                if row_index == 2 and slice_name == "memory":
                    source_count = int(float(series[0]["source_count"]))
                    label = f"all sources, $n_{{\\rm src}}={source_count}$"
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
                log_fit(
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
                    "$(\\kappa,\\bar\\kappa)$"
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
            style_axes(ax)
    save_figure(fig, output_stem, tight=False)


def pointwise_decay(rows, output_stem):
    point_excess = []
    point_sum = []
    envelope = {}
    for row in rows:
        time = int(float(row["time"]))
        distance = int(float(row["distance"]))
        excess = max(distance - time, 0)
        if excess <= 0:
            continue
        state = float(row["state_norm"])
        control = float(row["input_norm"])
        response_sum = state + control
        if not math.isfinite(response_sum) or response_sum <= 0.0:
            continue
        point_excess.append(float(excess))
        point_sum.append(response_sum)
        envelope[excess] = max(envelope.get(excess, 0), response_sum)
    if not point_excess:
        return []
    x = np.asarray(point_excess, dtype=float)
    y = np.asarray(point_sum, dtype=float)
    envelope_x = np.asarray(sorted(envelope), dtype=float)
    envelope_sum = positive([envelope[int(value)] for value in envelope_x])
    fig, ax = plt.subplots(
        figsize=SINGLE_COLUMN_FIGSIZE, dpi=360, constrained_layout=True
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
    log_fit(
        ax,
        envelope_x,
        envelope_sum,
        start_index=0,
        label="linear fit on log scale",
        envelope=True,
    )
    ax.set_yscale("log")
    ax.set_xlabel("distance beyond cone $[d_{\\mathcal{G}}(i,j)-t]_+$")
    ax.set_ylabel("$\\|\\Phi_x^\\star[t]_{ij}\\|+\\|\\Phi_u^\\star[t]_{ij}\\|$")
    ax.legend(frameon=False, fontsize=LEGEND_SIZE, loc="best")
    style_axes(ax)
    return save_figure(fig, output_stem)


def size_scaling(rows, output_stem):
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    node_count = np.asarray([float(row["nodes"]) for row in rows])
    actual_gap = np.asarray([float(row["loss_880"]) for row in rows])
    omitted_energy = np.asarray([float(row["tail"]) for row in rows])
    gamma = np.asarray([float(row["gamma"]) for row in rows])
    halo = np.asarray([float(row["halo"]) for row in rows])
    taps = np.asarray([float(row["taps"]) for row in rows])
    # A radius-four Manhattan ball has at most 1 + 2*r*(r+1) nodes.
    halo_bound = 1 + 2 * 4 * 5
    tap_bound = (160 + 1) * halo_bound
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
        3, 1, figsize=(3.36, 3.85), dpi=360, constrained_layout=True, sharex=True
    )
    axes[0].semilogy(
        node_count,
        np.maximum(actual_gap, 1e-16),
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
    axes[0].set_ylabel("normalized LQR quantity")
    axes[0].legend(frameon=False, fontsize=8.0, loc="best")
    axes[1].plot(
        node_count, gamma, "-o", color="#B45532", linewidth=1.35, markersize=3.6
    )
    axes[1].axhline(
        1.0,
        color="#5E6A70",
        linestyle="--",
        linewidth=1.0,
        label="small gain threshold",
    )
    axes[1].set_ylabel("residual bound $\\gamma_\\theta$")
    axes[1].legend(frameon=False, fontsize=8.0, loc="best")
    axes[2].semilogy(
        node_count,
        halo,
        "-o",
        color="#28766A",
        linewidth=1.2,
        markersize=3.4,
        label="$C_{\\rm halo}=C_{\\rm foot}$",
    )
    axes[2].semilogy(
        node_count,
        taps,
        "-^",
        color="#7A6B3A",
        linewidth=1.2,
        markersize=3.4,
        label="$C_{\\rm tap}$",
    )
    axes[2].axhline(
        halo_bound,
        color="#28766A",
        linestyle=":",
        linewidth=1.0,
        label="uniform spatial bound",
    )
    axes[2].axhline(
        tap_bound,
        color="#7A6B3A",
        linestyle=":",
        linewidth=1.0,
        label="uniform tap bound",
    )
    axes[2].set_xlabel("network size $N$")
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
    for axis in axes[:-1]:
        axis.tick_params(axis="x", which="both", labelbottom=False)
    save_figure(fig, output_stem)


def candidate_selection(proxy_rows, exact_rows, validation_rows, figures_dir):
    finite_proxy = [
        r
        for r in proxy_rows
        if math.isfinite(float(r["tail"])) and float(r["tail"]) > 0
    ]
    proxy_front = frontier(finite_proxy, "tail")
    finite_exact = feasible_rows(exact_rows)
    infeasible_exact = [r for r in exact_rows if r["status"] == "infeasible_structural"]
    exact_front = frontier(finite_exact, "loss")
    finite_validation = feasible_rows(validation_rows)
    cost_values = np.asarray(
        [float(row["taps"]) for row in finite_proxy + finite_exact + finite_validation],
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
    proxy_by_arch = {architecture_key(row): row for row in finite_proxy}
    matched_exact = [
        (row, proxy_by_arch[architecture_key(row)])
        for row in finite_exact
        if architecture_key(row) in proxy_by_arch
    ]
    matched_validation = [
        (row, proxy_by_arch[architecture_key(row)])
        for row in finite_validation
        if architecture_key(row) in proxy_by_arch
    ]
    matched_infeasible = [
        (row, proxy_by_arch[architecture_key(row)])
        for row in infeasible_exact
        if architecture_key(row) in proxy_by_arch
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
        [float(row["taps"]) for row in finite_proxy],
        positive([float(row["tail"]) for row in finite_proxy]),
        s=5.5,
        color="#AAB9BF",
        alpha=0.22,
        linewidth=0.0,
        label="candidate control architectures",
    )
    px = np.asarray([float(row["taps"]) for row in proxy_front])
    py = positive([float(row["tail"]) for row in proxy_front])
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
            [float(proxy["taps"]) for _, proxy in exact_screen_points],
            positive([float(proxy["tail"]) for _, proxy in exact_screen_points]),
            s=22,
            facecolor="none",
            edgecolor=SPATIAL_COLOR,
            linewidth=0.9,
            label="tested by exact QP",
            zorder=5,
        )
    if matched_infeasible:
        ax.scatter(
            [float(proxy["taps"]) for _, proxy in matched_infeasible],
            positive([float(proxy["tail"]) for _, proxy in matched_infeasible]),
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
    ax.set_ylabel("normalized response tail $p_\\theta$")
    ax.tick_params(labelbottom=False)
    ax.legend(frameon=False, fontsize=LEGEND_SIZE, loc="best")
    style_axes(ax, screening=True)
    ax = axes[1]
    for family, marker, label, size, color in [
        (
            "pure_cone",
            "o",
            "full response, $\\bar\\kappa=\\kappa+v_{\\rm cone}T$",
            26,
            "#6E5A8A",
        ),
        (
            "saturated",
            "s",
            "full response, independent $\\bar\\kappa$",
            31,
            FOOTPRINT_COLOR,
        ),
    ]:
        family_rows = [row for row in finite_exact if str(row["family"]) == family]
        if not family_rows:
            continue
        ax.scatter(
            [float(row["taps"]) for row in family_rows],
            positive([float(row["loss"]) for row in family_rows]),
            color=color,
            marker=marker,
            s=size,
            edgecolor="#25323A",
            linewidth=0.42,
            alpha=0.8,
            label=label,
            zorder=3,
        )
    fx = np.asarray([float(row["taps"]) for row in exact_front])
    fy = positive([float(row["loss"]) for row in exact_front])
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
            [float(row["taps"]) for row in finite_validation],
            positive([float(row["loss"]) for row in finite_validation]),
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
    ax.set_xlabel("average tap count $C_{\\rm tap}$")
    ax.set_ylabel("exact SLS performance loss")
    ax.legend(frameon=False, fontsize=LEGEND_SIZE, loc="best")
    style_axes(ax, screening=True)
    save_figure(fig, figures_dir / "fig07-candidate-selection")
