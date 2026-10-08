"""Reproduce Figures 4-7 and Table I of
Spatiotemporal Response Decay for Near-Optimal Distributed LQR
via System Level Synthesis (Chenchen Zhou and Jose Matias).
"""

import os

for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "2"
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import csv
from pathlib import Path

import plotting
from control import build_mesh, direct_controller, solve_exact, taps_per_node
from experiments import (
    architecture,
    direct_residuals,
    pointwise_responses,
    response_tails,
    size_scaling,
    subtract_floor,
)

BASE = Path(__file__).resolve().parent


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def exact_sweeps(data):
    configurations = {r["id"]: r for r in read_csv(data / "exact-configurations.csv")}
    rows = read_csv(data / "exact-parameter-sweeps.csv")
    for row in rows:
        result = configurations[row["id"]]
        row["loss"] = float(result["loss"])
        row["source_count"] = (
            100 if result["sources"] == "all" else len(result["sources"].split(";"))
        )
    subtract_floor(rows, "loss")
    return rows


def figures_and_table(data, output):
    exact = read_csv(data / "exact-configurations.csv")
    tails = read_csv(data / "response-tails.csv")
    residuals = read_csv(data / "direct-residuals.csv")
    subtract_floor(tails, "tail", last=True)
    subtract_floor(residuals, "gamma")
    plotting.configure()
    plotting.parameter_sweeps(
        tails,
        residuals,
        exact_sweeps(data),
        output / "fig04-parameter-sweeps",
    )
    plotting.pointwise_decay(
        read_csv(data / "pointwise-decay.csv"), output / "fig05-pointwise-decay"
    )
    plotting.size_scaling(
        read_csv(data / "size-scaling.csv"), output / "fig06-size-scaling"
    )
    plotting.candidate_selection(
        read_csv(data / "screening.csv"),
        [r for r in exact if r["group"] == "grid"],
        [r for r in exact if r["group"] == "selected"],
        output,
    )
    mesh = build_mesh()
    static = dict(kappa=2, kappa_bar=2, memory=0, **direct_controller(mesh, 2, 2, 0))
    table = []
    for direct in [static, *read_csv(data / "direct.csv")]:
        theta = architecture(direct)
        result = next(
            r for r in exact if architecture(r) == theta and r["sources"] == "all"
        )
        table.append(
            dict(
                kappa=theta[0],
                kappa_bar=theta[1],
                memory=theta[2],
                average_taps=taps_per_node(mesh.distances, *theta),
                direct_loss_percent=100 * float(direct["loss_880"]),
                direct_cost_kind="Lyapunov" if theta[2] == 0 else "880-term estimate",
                exact_loss_percent=100 * float(result["loss"])
                if result["status"] == "feasible"
                else "",
                exact_feasibility=result["status"],
            )
        )
    write_csv(output / "table01.csv", table)


def recompute(output):
    """The supplied CSVs specify the paper's parameter grids and source sets."""
    data = output / "data"
    data.mkdir(exist_ok=True)
    mesh = build_mesh()
    print("Computing response tails and truncation residuals...", flush=True)
    tail_rows = read_csv(BASE / "data/response-tails.csv")
    screen_rows = read_csv(BASE / "data/screening.csv")
    response_tails(mesh, tail_rows + screen_rows)
    write_csv(data / "response-tails.csv", tail_rows)
    write_csv(data / "screening.csv", screen_rows)
    write_csv(
        data / "direct-residuals.csv",
        direct_residuals(mesh, read_csv(BASE / "data/direct-residuals.csv")),
    )
    write_csv(data / "pointwise-decay.csv", pointwise_responses(mesh))
    scaling = []
    for side in (5, 7, 10, 12):
        print(f"Computing the {side} x {side} mesh...", flush=True)
        scaling.append(size_scaling(side))
    write_csv(data / "size-scaling.csv", scaling)
    direct = []
    for footprint in (2, 3):
        print(
            f"Computing the direct controller with footprint {footprint}...", flush=True
        )
        direct.append(
            dict(
                kappa=2,
                kappa_bar=footprint,
                memory=159,
                **direct_controller(mesh, 2, footprint, 159),
            )
        )
    write_csv(data / "direct.csv", direct)
    configurations = read_csv(BASE / "data/exact-configurations.csv")
    computed = {}
    for index, row in enumerate(configurations, 1):
        print(f"Exact SLS {index}/{len(configurations)}: {row['id']}", flush=True)
        theta = architecture(row)
        key = (*theta, row["sources"])
        if key not in computed:
            sources = (
                None
                if row["sources"] == "all"
                else [int(j) for j in row["sources"].split(";")]
            )
            computed[key] = solve_exact(mesh, *theta, sources=sources)
        result = computed[key]
        row.update(
            loss=result["loss"],
            status=result["status"],
            taps=taps_per_node(mesh.distances, *theta),
        )
        # Save each finished solve so that long runs retain their results.
        write_csv(data / "exact-configurations.csv", configurations[:index])
    write_csv(
        data / "exact-parameter-sweeps.csv",
        read_csv(BASE / "data/exact-parameter-sweeps.csv"),
    )
    failures = [
        r["id"] for r in configurations if r["status"] == "numerically_inconclusive"
    ]
    if failures:
        raise RuntimeError(
            "Exact solves did not converge; results saved for: " + ", ".join(failures)
        )
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute all experiments before plotting (can take substantial time).",
    )
    args = parser.parse_args()
    output = BASE / "results" / ("recompute" if args.recompute else "paper")
    output.mkdir(parents=True, exist_ok=True)
    data = recompute(output) if args.recompute else BASE / "data"
    figures_and_table(data, output)
    print(f"Saved Figures 4-7 and Table I to {output}")


if __name__ == "__main__":
    main()
