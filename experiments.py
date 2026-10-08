"""Calculations for Figures 4-7 and Table I."""

import numpy as np
from control import (
    ETA,
    achieved_costs,
    build_mesh,
    lqr_responses,
    residual,
    residual_bound,
    response_energy,
    taps_per_node,
    temporal_tail,
    truncate,
)


def architecture(row):
    return tuple(int(row[name]) for name in ("kappa", "kappa_bar", "memory"))


def subtract_floor(rows, metric, *, last=False):
    """Subtract each curve's limiting sampled value, as in Figure 4."""
    groups = {}
    for row in rows:
        groups.setdefault((row["slice"], row["series_order"]), []).append(row)
    for group in groups.values():
        ordered = sorted(group, key=lambda row: float(row["value"]))
        values = [
            float(row[metric]) for row in ordered if np.isfinite(float(row[metric]))
        ]
        floor = (values[-1] if last else min(values)) if values else np.nan
        for row in group:
            row[metric + "_excess"] = max(float(row[metric]) - floor, 0.0)


def response_tails(mesh, rows):
    """Omitted spatial energy plus the infinite Riccati temporal tail."""
    n = mesh.B.shape[1]
    horizon = max(int(row["memory"]) for row in rows) + 1
    X, U = lqr_responses(mesh, horizon)
    diameter = int(mesh.distances.max())
    profile = np.zeros((horizon + 1, diameter + 1))
    for t in range(1, horizon + 1):
        state = X[t].reshape(n, 2, n, 2)
        inputs = U[t].reshape(n, n, 2)
        energy = ETA**2 * np.square(state[:, 0]).sum(axis=2) + np.square(inputs).sum(
            axis=2
        )
        for d in range(diameter + 1):
            profile[t, d] = energy[mesh.distances == d].sum()
    temporal = {T: temporal_tail(mesh, T + 1) for T in {int(r["memory"]) for r in rows}}
    for row in rows:
        kappa, footprint, memory = architecture(row)
        spatial = sum(
            profile[t + 1, min(kappa + t, footprint) + 1 :].sum()
            for t in range(memory + 1)
        )
        row["tail"] = float((spatial + temporal[memory]) / np.trace(mesh.P))
        if "taps" in row:
            row["taps"] = taps_per_node(mesh.distances, kappa, footprint, memory)
    return rows


def direct_residuals(mesh, rows):
    X, U = lqr_responses(mesh, max(int(row["memory"]) for row in rows) + 1)
    computed = {}
    for row in rows:
        theta = architecture(row)
        if theta not in computed:
            x, u = truncate(X, U, mesh.distances, *theta)
            computed[theta] = residual_bound(residual(mesh, x, u))
        row["gamma"] = computed[theta]
    return rows


def pointwise_responses(mesh):
    """Figure 5 uses the blocks outside the cone from source node 55."""
    source = 55
    columns = slice(2 * source, 2 * source + 2)
    horizon = int(mesh.distances[:, source].max())
    X, U = lqr_responses(mesh, horizon)
    rows = []
    for time in range(horizon):
        for node in np.flatnonzero(mesh.distances[:, source] > time):
            rows.append(
                dict(
                    node=node,
                    time=time,
                    distance=int(mesh.distances[node, source]),
                    state_norm=float(
                        np.linalg.norm(
                            X[time + 1][2 * node : 2 * node + 2, columns], "fro"
                        )
                    ),
                    input_norm=float(
                        np.linalg.norm(U[time + 1][node : node + 1, columns], "fro")
                    ),
                )
            )
    return rows


def size_scaling(side):
    """Fixed (kappa, kappa_bar, T) = (4, 4, 160), varying mesh size."""
    mesh = build_mesh(side)
    X, U = lqr_responses(mesh, 320)
    x, u = truncate(X, U, mesh.distances, 4, 4, 160)
    delta = residual(mesh, x, u)
    omitted = sum(response_energy(X[t] - x[t], U[t] - u[t]) for t in range(1, len(x)))
    omitted += sum(response_energy(X[t], U[t]) for t in range(len(x), len(X)))
    omitted += temporal_tail(mesh, 320)
    costs = achieved_costs(mesh, x, u, delta)
    central = float(np.trace(mesh.P))
    return dict(
        side=side,
        nodes=side**2,
        tail=omitted / central,
        gamma=residual_bound(delta),
        loss_880=costs[880] / central - 1,
        prefix_change=(costs[880] - costs[720]) / central,
        halo=float(np.sum(mesh.distances <= 4, axis=1).mean()),
        taps=taps_per_node(mesh.distances, 4, 4, 160),
    )
