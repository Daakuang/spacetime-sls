"""Thermal mesh, truncated LQR responses, and exact localized SLS."""

from dataclasses import dataclass

import numpy as np
import osqp
from scipy import sparse
from scipy.linalg import solve_discrete_are, solve_discrete_lyapunov, svdvals

ETA = 1 / 16


@dataclass
class Mesh:
    A: np.ndarray
    B: np.ndarray
    distances: np.ndarray
    P: np.ndarray
    K: np.ndarray


def build_mesh(side=10):
    """Paper model: edge weight 0.05, sampling time 1, eta = 1/16."""
    n = side**2
    row, col = np.divmod(np.arange(n), side)
    distances = abs(row[:, None] - row) + abs(col[:, None] - col)
    adjacency = 0.05 * (distances == 1)
    laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
    A = np.eye(2 * n)
    A[0::2, 1::2] = np.eye(n)
    A[1::2, 1::2] -= laplacian
    B = np.zeros((2 * n, n))
    B[1::2] = ETA * np.eye(n)
    Q = np.diag(np.tile([ETA**2, 0], n))
    P = solve_discrete_are(A, B, Q, np.eye(n))
    K = np.linalg.solve(np.eye(n) + B.T @ P @ B, B.T @ P @ A)
    return Mesh(A, B, distances, P, K)


def lqr_responses(mesh, horizon):
    """Index 1 is physical time zero: X[1] = I and U[1] = -K."""
    n = len(mesh.A)
    X = [np.zeros((n, n)), np.eye(n)]
    U = [np.zeros_like(mesh.K), -mesh.K.copy()]
    F = mesh.A - mesh.B @ mesh.K
    for _ in range(2, horizon + 1):
        X.append(F @ X[-1])
        U.append(-mesh.K @ X[-1])
    return X, U


def truncate(X, U, distances, kappa, kappa_bar, memory):
    """Keep filter taps t = 0,...,T inside d <= min(kappa+t, kappa_bar)."""
    x, u = [X[0].copy()], [U[0].copy()]
    for t in range(memory + 1):
        keep = distances <= min(kappa + t, kappa_bar)
        input_mask = np.repeat(keep, 2, axis=1)
        x.append(X[t + 1] * np.repeat(input_mask, 2, axis=0))
        u.append(U[t + 1] * input_mask)
    return x, u


def residual(mesh, X, U):
    """Coefficients of [zI-A, -B] Phi - I, including the FIR terminal row."""
    A, B = sparse.csr_matrix(mesh.A), sparse.csr_matrix(mesh.B)
    delta = [X[1] - np.eye(len(mesh.A))]
    for t in range(1, len(X)):
        next_x = X[t + 1] if t + 1 < len(X) else np.zeros_like(X[t])
        delta.append(next_x - A @ X[t] - B @ U[t])
    return delta


def residual_bound(delta):
    n = delta[0].shape[0] // 2
    rows, cols = np.zeros(n), np.zeros(n)
    for matrix in delta:
        blocks = np.linalg.norm(matrix.reshape(n, 2, n, 2), axis=(1, 3))
        rows += blocks.sum(axis=1)
        cols += blocks.sum(axis=0)
    return float(np.sqrt(rows.max() * cols.max()))


def taps_per_node(distances, kappa, kappa_bar, memory):
    start = np.maximum(distances - kappa, 0)
    taps = np.where(distances <= kappa_bar, np.maximum(memory - start + 1, 0), 0)
    return float(taps.sum(axis=1).mean())


def response_energy(X, U):
    return float(ETA**2 * np.sum(X[0::2] ** 2) + np.sum(U**2))


def temporal_tail(mesh, start):
    F = np.linalg.matrix_power(mesh.A - mesh.B @ mesh.K, start)
    return float(np.trace(F.T @ mesh.P @ F))


def _nonzero_terms(matrices):
    """Sparse corrections keep the long achieved-response calculation small."""
    terms = []
    for lag, matrix in enumerate(matrices):
        matrix = matrix.copy()
        matrix[abs(matrix) <= 1e-14] = 0
        nnz = np.count_nonzero(matrix)
        if nnz:
            operator = (
                sparse.csr_matrix(matrix) if nnz <= 0.25 * matrix.size else matrix
            )
            terms.append((lag, operator))
    return terms


def achieved_costs(mesh, X, U, delta, horizons=(720, 880)):
    """Finite-prefix costs of Phi (I+Delta)^-1, not of the local filters."""
    n, H = len(mesh.A), len(X) - 1
    F = mesh.A - mesh.B @ mesh.K
    # The paper's masks keep X[1] = I, so Delta[0] = 0.
    delta_terms = _nonzero_terms(delta[1:])
    central = np.eye(n)
    dx, du = [np.zeros_like(X[0])], [np.zeros_like(U[0])]
    for t in range(1, H + 1):
        dx.append(X[t] - central)
        du.append(U[t] + mesh.K @ central)
        central = F @ central
    correction_x, correction_u = _nonzero_terms(dx), _nonzero_terms(du)
    F_H = central
    inverse = {0: np.eye(n)}
    convolution = np.zeros((n, n))
    energy, costs = 0.0, {}
    for order in range(max(horizons)):
        if order:
            value = np.zeros((n, n))
            for lag, term in delta_terms:
                if order - lag - 1 in inverse:
                    value -= term @ inverse[order - lag - 1]
            if np.max(abs(value)) > 1e-14:
                inverse[order] = value
        convolution = inverse.get(order, np.zeros((n, n))) + F @ convolution
        if order - H in inverse:
            convolution -= F_H @ inverse[order - H]
        x, u = convolution.copy(), -mesh.K @ convolution
        for t, term in correction_x:
            if order - t + 1 in inverse:
                x += term @ inverse[order - t + 1]
        for t, term in correction_u:
            if order - t + 1 in inverse:
                u += term @ inverse[order - t + 1]
        energy += response_energy(x, u)
        if order + 1 in horizons:
            costs[order + 1] = energy
        inverse.pop(order - H - 1, None)
    return costs


def direct_controller(mesh, kappa, kappa_bar, memory):
    X, U = lqr_responses(mesh, memory + 1)
    X, U = truncate(X, U, mesh.distances, kappa, kappa_bar, memory)
    delta = residual(mesh, X, U)
    gamma = residual_bound(delta)
    weights = np.tile([0.25, 1.0], len(mesh.B[0]))
    scaled = float(
        sum(svdvals(d * weights[:, None] / weights[None, :])[0] for d in delta)
    )
    result = dict(gamma=gamma, scaled_bound=scaled, loss_720=np.nan, loss_880=np.nan)
    if memory == 0:
        F = mesh.A + mesh.B @ U[1]
        if max(abs(np.linalg.eigvals(F))) < 1:
            Q = np.diag(np.tile([ETA**2, 0], len(mesh.B[0])))
            value = solve_discrete_lyapunov(F.T, Q + U[1].T @ U[1])
            result["loss_880"] = float(np.trace(value) / np.trace(mesh.P) - 1)
    elif gamma < 1 or scaled < 1:
        for horizon, cost in achieved_costs(mesh, X, U, delta).items():
            result[f"loss_{horizon}"] = float(cost / np.trace(mesh.P) - 1)
    return result


OSQP_SETTINGS = dict(
    verbose=False,
    eps_abs=1e-10,
    eps_rel=1e-10,
    eps_prim_inf=1e-10,
    eps_dual_inf=1e-10,
    max_iter=100000,
    polishing=False,
    polish_refine_iter=10,
    adaptive_rho=True,
    adaptive_rho_interval=50,
    scaled_termination=False,
    check_termination=25,
    scaling=10,
    warm_starting=True,
    time_limit=3600,
)


def _column_qp(mesh, source, kappa, kappa_bar, memory):
    """One source node, two disturbance coordinates, one shared sparse QP."""
    A, B, H = mesh.A, mesh.B, memory + 1
    x_rows, u_rows, x_offsets, u_offsets, weights = [], [], [], [], []
    count = 0
    for t in range(H):
        keep = mesh.distances[:, source] <= min(kappa + t, kappa_bar)
        x = np.flatnonzero(np.repeat(keep, 2))
        u = np.flatnonzero(keep)
        x_rows.append(x)
        u_rows.append(u)
        x_offsets.append(count)
        u_offsets.append(count + len(x))
        weights.extend(np.where(x % 2 == 0, 2 * ETA**2, 0))
        weights.extend(np.full(len(u), 2.0))
        count += len(x) + len(u)
    P = sparse.diags(weights, format="csc")
    rows, cols, values = [], [], []

    def add_block(matrix, row_offset, col_offset):
        block = sparse.coo_matrix(matrix)
        rows.extend(block.row + row_offset)
        cols.extend(block.col + col_offset)
        values.extend(block.data)

    add_block(np.eye(len(x_rows[0])), 0, x_offsets[0])
    row_count = len(x_rows[0])
    for t in range(H):
        active = np.any(abs(A[:, x_rows[t]]) > 1e-12, axis=1)
        active |= np.any(abs(B[:, u_rows[t]]) > 1e-12, axis=1)
        if t + 1 < H:
            active[x_rows[t + 1]] = True
        selected = np.flatnonzero(active)
        if t + 1 < H:
            selector = (selected[:, None] == x_rows[t + 1]).astype(float)
            add_block(selector, row_count, x_offsets[t + 1])
        sign = -1 if t + 1 < H else 1
        add_block(sign * A[np.ix_(selected, x_rows[t])], row_count, x_offsets[t])
        add_block(sign * B[np.ix_(selected, u_rows[t])], row_count, u_offsets[t])
        row_count += len(selected)
    E = sparse.coo_matrix((values, (rows, cols)), shape=(row_count, count)).tocsc()
    rhs = np.zeros((row_count, 2))
    for component in range(2):
        rhs[np.flatnonzero(x_rows[0] == 2 * source + component), component] = 1
    solver = osqp.OSQP()
    solver.setup(P=P, q=np.zeros(count), A=E, l=rhs[:, 0], u=rhs[:, 0], **OSQP_SETTINGS)
    energy, primal_max, dual_max, sls_max = 0.0, 0.0, 0.0, 0.0
    accepted = True
    for component in range(2):
        solver.update(l=rhs[:, component], u=rhs[:, component])
        solver.warm_start(x=np.zeros(count), y=np.zeros(row_count))
        result = solver.solve(raise_error=False)
        if result.info.status != "solved":
            return np.nan, np.inf, np.inf, np.inf
        z, dual = result.x, result.y
        primal = float(max(abs(E @ z - rhs[:, component])))
        stationarity = float(max(abs(P @ z + E.T @ dual)))
        X, U = np.zeros((H + 1, len(A))), np.zeros((H + 1, B.shape[1]))
        for t in range(H):
            X[t + 1, x_rows[t]] = z[x_offsets[t] : x_offsets[t] + len(x_rows[t])]
            U[t + 1, u_rows[t]] = z[u_offsets[t] : u_offsets[t] + len(u_rows[t])]
        initial = np.zeros(len(A))
        initial[2 * source + component] = 1
        sls_error = float(max(abs(X[1] - initial)))
        for t in range(1, H + 1):
            next_x = X[t + 1] if t < H else np.zeros(len(A))
            sls_error = max(sls_error, float(max(abs(next_x - A @ X[t] - B @ U[t]))))
            energy += response_energy(X[t], U[t])
        primal_max = max(primal_max, primal)
        dual_max = max(dual_max, stationarity)
        sls_max = max(sls_max, sls_error)
        accepted &= max(primal, stationarity, sls_error) <= 1e-7
    return (
        energy if accepted and np.isfinite(energy) else np.nan,
        primal_max,
        dual_max,
        sls_max,
    )


def solve_exact(mesh, kappa, kappa_bar, memory, sources=None):
    """A timeout or failed residual check is inconclusive, not infeasible."""
    sources = list(range(mesh.B.shape[1])) if sources is None else sources
    # For kappa >= 1 the radius-one deadbeat controller takes two steps.
    # For kappa = 0 the leading front must reach the entire source footprint.
    eccentricity = max(mesh.distances[:, source].max() for source in sources)
    infeasible = memory == 0 or (
        kappa == 0 and (kappa_bar < eccentricity or memory < eccentricity + 1)
    )
    if infeasible:
        return dict(loss=np.nan, status="infeasible_structural", residual=np.nan)
    columns = [_column_qp(mesh, j, kappa, kappa_bar, memory) for j in sources]
    energy = sum(column[0] for column in columns)
    coordinates = [i for j in sources for i in (2 * j, 2 * j + 1)]
    denominator = float(sum(mesh.P[i, i] for i in coordinates))
    return dict(
        loss=energy / denominator - 1,
        status="feasible" if np.isfinite(energy) else "numerically_inconclusive",
        residual=max(column[3] for column in columns),
    )
