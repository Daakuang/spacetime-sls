"""Reproduce the numerical results in
Spatiotemporal Response Decay for Near-Optimal Distributed LQR
via System Level Synthesis, by Chenchen Zhou and Jose Matias.

Run `python reproduce.py` for Figures 4-7 and Table I, or
`python reproduce.py recompute` to recompute the experiments first."""

from __future__ import annotations

import os

# Set these before importing NumPy and SciPy for consistent CPU use.
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_variable] = "2"
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np
from control import (
    Architecture,
    build_mesh,
    evaluate_direct,
    solve_exact,
)
from experiments import (
    DecayFigureConfig,
    MeshArchitectureScreenConfig,
    MeshSizeScalingConfig,
    _compute_single_source_raw_and_shell_rows,
    _configure_matplotlib,
    _direct_truncation_slice_rows,
    _plot_architecture_screening_validation,
    _plot_mesh_size_scaling,
    _plot_pointwise_offcone_decay,
    _tail_slice_rows,
    plot_parameter_sweep_matrix,
    run_mesh_architecture_screen,
    run_mesh_size_scaling,
)


def data_file(name, data_dir=None):
    return (
        Path(data_dir) if data_dir else Path(__file__).resolve().parent / "data"
    ) / name


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean_json(v) for v in value]
    if hasattr(value, "tolist"):
        return clean_json(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(
        json.dumps(clean_json(value), indent=2, allow_nan=False), encoding="utf-8"
    )


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def figures(output: Path, data_dir=None):
    """Draw Figs. 4--7; schematic Figs. 1--3 are not numerical outputs."""

    _configure_matplotlib()
    plot_parameter_sweep_matrix(
        read_csv(data_file("response-tails.csv", data_dir)),
        read_csv(data_file("direct-residuals.csv", data_dir)),
        read_csv(data_file("exact-parameter-sweeps.csv", data_dir)),
        output / "fig04-parameter-sweeps",
    )
    _plot_pointwise_offcone_decay(
        read_csv(data_file("pointwise-decay.csv", data_dir)),
        output / "fig05-pointwise-decay",
        r_dyn=1,
    )
    _plot_mesh_size_scaling(
        read_csv(data_file("size-scaling.csv", data_dir)),
        output / "fig06-size-scaling",
        config=MeshSizeScalingConfig(kappa=4, kappa_bar=4, time_horizon=160),
        column_layout=True,
    )
    _plot_architecture_screening_validation(
        read_csv(data_file("screening.csv", data_dir)),
        read_csv(data_file("exact-grid.csv", data_dir)),
        read_csv(data_file("exact-selected-sources.csv", data_dir)),
        output,
    )
    for suffix in (".pdf", ".png"):
        (output / ("paper-mesh-architecture-pareto-all-points" + suffix)).replace(
            output / ("fig07-candidate-selection" + suffix)
        )


def table(output: Path, data_dir=None):
    """Export Table I with raw ratios, not just rounded typeset numbers."""

    model = build_mesh()
    static = evaluate_direct(model, Architecture(2, 2, 0))
    rows = [
        dict(
            kappa=2,
            kappa_bar=2,
            memory=0,
            average_taps=static["resources"]["average_taps_per_controller"],
            direct_loss_percent=100 * static["relative_loss"],
            direct_cost_kind="Lyapunov",
            exact_loss_percent="",
            exact_feasibility="infeasible_structural",
        )
    ]
    exact_rows = read_csv(data_file("exact-configurations.csv", data_dir))
    for footprint in (2, 3):
        direct = json.loads(data_file(f"direct-{footprint}.json", data_dir).read_text())
        exact = next(
            r
            for r in exact_rows
            if r["kappa"] == "2"
            and r["kappa_bar"] == str(footprint)
            and r["memory"] == "159"
            and r["source_mode"] == "all"
        )
        rows.append(
            dict(
                kappa=2,
                kappa_bar=footprint,
                memory=159,
                average_taps=float(exact["taps_per_node"]),
                direct_loss_percent=100 * direct["relative_losses"]["880"],
                direct_cost_kind="880-term estimate",
                exact_loss_percent=100 * float(exact["relative_loss"]),
                exact_feasibility=exact["feasibility"],
            )
        )
    write_csv(output / "table01.csv", rows)
    write_json(output / "table01.json", rows)
    return rows


def exact_inventory(scope: str):
    """Return the architectures and disturbance sources used in the paper."""
    rows = read_csv(data_file("exact-configurations.csv"))
    if scope == "table":
        return [
            r
            for r in rows
            if r["reference_id"] in ("all-106", "all-107", "table-static")
        ]
    if scope == "full":
        return [r for r in rows if r["reference_id"].startswith("all-")]
    if scope == "selected":
        return [r for r in rows if r["reference_id"].startswith("sampled-")]
    if scope == "slices":
        return [r for r in rows if r["reference_id"].startswith("sweep-")]
    return rows


def _experiment_key(row, *, slices=False):
    """Match an exact-SLS result to its architecture and disturbance sources."""
    return (
        int(row["kappa"]),
        18 if row["kappa_bar"] == "none" else int(row["kappa_bar"]),
        int(row["time_horizon"] if slices else row["memory"]),
        row["sources"],
    )


def _update_exact_figure_data(data_dir, configurations, results):
    """Write figure data from newly computed costs and feasibility results."""
    for name in ("exact-grid.csv", "exact-selected-sources.csv"):
        rows = read_csv(data_file(name))
        for row in rows:
            result = results[row["reference_id"]]
            loss = result["relative_loss"]
            row.update(
                normalized_cost_gap=loss if loss is not None else math.nan,
                feasible_column_cost_gap=loss if loss is not None else math.nan,
                qp_feasible_fraction=int(result["classification"] == "feasible"),
                qp_feasibility=result["classification"],
                implementation_taps_per_controller=result["resources"][
                    "average_taps_per_controller"
                ],
                architecture_cost=result["resources"]["total_tap_blocks"],
            )
        write_csv(data_dir / name, rows)
    by_architecture = {
        _experiment_key(row): results[row["reference_id"]] for row in configurations
    }
    rows = read_csv(data_file("exact-parameter-sweeps.csv"))
    for row in rows:
        loss = by_architecture[_experiment_key(row, slices=True)]["relative_loss"]
        row["raw_gap"] = loss if loss is not None else math.nan
    groups = {}
    for row in rows:
        groups.setdefault((row["slice"], row["series_order"]), []).append(row)
    for group in groups.values():
        finite = [r["raw_gap"] for r in group if math.isfinite(r["raw_gap"])]
        floor = min(finite) if finite else math.nan
        for row in group:
            row["floor"] = floor
            row["excess_gap"] = max(row["raw_gap"] - floor, 0.0)
    write_csv(data_dir / "exact-parameter-sweeps.csv", rows)


def recompute(output: Path, *, time_limit=3600):
    """Recompute the paper data, then draw every numerical figure and Table I."""
    data_dir = output / "data"
    exact_dir = output / "exact"
    data_dir.mkdir(parents=True, exist_ok=True)
    exact_dir.mkdir(parents=True, exist_ok=True)
    times = tuple(h - 1 for h in MeshArchitectureScreenConfig.time_horizons)
    candidates = run_mesh_architecture_screen(
        MeshArchitectureScreenConfig(
            output_dir=output / "screening", time_horizons=times
        )
    )
    write_csv(data_dir / "screening.csv", candidates.rows)
    tail_grid = run_mesh_architecture_screen(
        MeshArchitectureScreenConfig(output_dir=output / "response-tails")
    )
    write_csv(data_dir / "response-tails.csv", _tail_slice_rows(tail_grid.rows))
    config = DecayFigureConfig()
    raw, _ = _compute_single_source_raw_and_shell_rows(config)
    write_csv(data_dir / "pointwise-decay.csv", raw)
    write_csv(data_dir / "direct-residuals.csv", _direct_truncation_slice_rows(config))
    scaling = run_mesh_size_scaling(
        MeshSizeScalingConfig(
            output_dir=output / "scaling", kappa=4, kappa_bar=4, time_horizon=160
        )
    )
    write_csv(data_dir / "size-scaling.csv", scaling)
    model = build_mesh()
    for footprint in (2, 3):
        direct = evaluate_direct(model, Architecture(2, footprint, 159))
        write_json(data_dir / f"direct-{footprint}.json", direct)
    configurations = exact_inventory("all")
    results = {}
    for index, row in enumerate(configurations, 1):
        print(
            f"Exact SLS {index}/{len(configurations)}: {row['reference_id']}",
            flush=True,
        )
        kappa, footprint, memory, sources = _experiment_key(row)
        result, _ = solve_exact(
            model,
            Architecture(kappa, footprint, memory),
            sources=[int(j) for j in sources.split(";")],
            settings={"time_limit": time_limit},
        )
        results[row["reference_id"]] = result
        write_json(exact_dir / (row["reference_id"] + ".json"), result)
        row.update(
            relative_loss=result["relative_loss"]
            if result["relative_loss"] is not None
            else math.nan,
            feasibility=result["classification"],
            taps_per_node=result["resources"]["average_taps_per_controller"],
        )
    write_csv(data_dir / "exact-configurations.csv", configurations)
    _update_exact_figure_data(data_dir, configurations, results)
    figures(output, data_dir)
    table(output, data_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    descriptions = {
        "paper": "Generate numerical Figs. 4-7 and Table I from saved data",
        "recompute": "Recompute all paper experiments and generate the figures and table",
        "figures": "Generate numerical Figs. 4-7 from saved data",
        "table": "Generate Table I",
        "direct": "Compute one direct-truncation controller",
        "exact": "Compute one exact localized SLS controller",
        "screen": "Recompute the paper's architecture screening",
        "nonqp": "Recompute decay and direct-residual figure data",
        "scaling": "Recompute the network-size experiment",
        "exact-batch": "Recompute the exact-SLS results for the paper",
    }
    for name, description in descriptions.items():
        command = commands.add_parser(name, help=description)
        command.add_argument("--output", type=Path)
        if name in ("direct", "exact"):
            command.add_argument("--side", type=int, default=10)
            command.add_argument("--kappa", type=int, default=2)
            command.add_argument("--footprint", type=int, default=3)
            command.add_argument(
                "--memory", type=int, default=159, help="Last filter tap T"
            )
        if name == "direct":
            command.add_argument("--prefixes", type=int, nargs="+", default=[720, 880])
        if name in ("exact", "exact-batch", "recompute"):
            command.add_argument("--time-limit", type=float, default=3600)
        if name == "exact":
            command.add_argument(
                "--sources", type=int, nargs="+", help="Omit to solve every source"
            )
        if name == "exact-batch":
            command.add_argument(
                "--scope",
                choices=("table", "full", "slices", "selected"),
                default="table",
            )

    args = parser.parse_args(sys.argv[1:] or ["paper"])

    base = Path(__file__).resolve().parent
    output = (args.output or base / "results" / args.command).resolve()
    if output == base or output.is_relative_to(base / "data"):
        parser.error(
            "Choose an output folder outside the source and input-data folders"
        )
    output.mkdir(parents=True, exist_ok=True)

    if args.command in ("paper", "figures"):
        figures(output)
    if args.command in ("paper", "table"):
        table(output)
    elif args.command == "recompute":
        recompute(output, time_limit=args.time_limit)
    elif args.command == "direct":
        result = evaluate_direct(
            build_mesh(args.side),
            Architecture(args.kappa, args.footprint, args.memory),
            prefix_lengths=args.prefixes,
        )
        write_json(output / "direct.json", result)
    elif args.command == "exact":
        result, response = solve_exact(
            build_mesh(args.side),
            Architecture(args.kappa, args.footprint, args.memory),
            sources=args.sources,
            settings={"time_limit": args.time_limit},
        )
        write_json(output / "exact.json", result)
        if response is not None:
            np.savez_compressed(
                output / "response.npz", phi_x=response.phi_x, phi_u=response.phi_u
            )
    elif args.command == "screen":
        times = tuple(h - 1 for h in MeshArchitectureScreenConfig.time_horizons)
        run_mesh_architecture_screen(
            MeshArchitectureScreenConfig(output_dir=output, time_horizons=times)
        )
    elif args.command == "nonqp":
        config = DecayFigureConfig()
        # Figure 4 samples filter memory T; the candidate grid samples horizon T+1.
        tail_grid = run_mesh_architecture_screen(
            MeshArchitectureScreenConfig(output_dir=output / "tail-grid")
        )
        write_csv(output / "tail-slices.csv", _tail_slice_rows(tail_grid.rows))
        raw, shell = _compute_single_source_raw_and_shell_rows(config)
        write_csv(output / "pointwise-raw.csv", raw)
        write_csv(output / "pointwise-shell.csv", shell)
        write_csv(
            output / "direct-residual-slices.csv", _direct_truncation_slice_rows(config)
        )
    elif args.command == "scaling":
        run_mesh_size_scaling(
            MeshSizeScalingConfig(
                output_dir=output, kappa=4, kappa_bar=4, time_horizon=160
            )
        )
    elif args.command == "exact-batch":
        model = build_mesh()
        inventory = exact_inventory(args.scope)
        for index, row in enumerate(inventory, 1):
            print(
                f"Exact configuration {index}/{len(inventory)}: {row['reference_id']}",
                flush=True,
            )
            result, _ = solve_exact(
                model,
                Architecture(
                    int(row["kappa"]),
                    18 if row["kappa_bar"] == "none" else int(row["kappa_bar"]),
                    int(row["memory"]),
                ),
                sources=[int(j) for j in row["sources"].split(";")],
                settings={"time_limit": args.time_limit},
            )
            write_json(output / (row["reference_id"] + ".json"), result)
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
