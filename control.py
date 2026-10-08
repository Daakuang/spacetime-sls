"""Thermal-mesh model, response truncation, and localized SLS optimization."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from fractions import Fraction
from math import comb
from typing import Sequence

import networkx as nx
import numpy as np
import osqp
from scipy import sparse
from scipy.linalg import (
    eigvals,
    lu_factor,
    lu_solve,
    solve_discrete_are,
    solve_discrete_lyapunov,
    svdvals,
)


@dataclass(frozen=True)
class LQRSolution:
    P: np.ndarray
    K: np.ndarray
    spectral_radius: float


def solve_lqr(model: object) -> LQRSolution:
    """Solve the centralized discrete algebraic Riccati equation."""

    P = solve_discrete_are(model.A, model.B, model.Q, model.R)
    gain_matrix = model.R + model.B.T @ P @ model.B
    K = np.linalg.solve(gain_matrix, model.B.T @ P @ model.A)
    spectral_radius = float(np.max(np.abs(eigvals(model.A - model.B @ K))))
    return LQRSolution(P=P, K=K, spectral_radius=spectral_radius)


def interleaved_two_component_permutation(node_count: int) -> np.ndarray:
    """Return indices mapping ``[x_1..x_N, v_1..v_N]`` to node-interleaved order."""

    if node_count <= 0:
        raise ValueError("node_count must be positive")
    return np.asarray(
        [index for node in range(node_count) for index in (node, node_count + node)],
        dtype=int,
    )


@dataclass(frozen=True)
class MeshCase:
    """Weighted square mesh used for Shin et al.'s 2D HVAC example."""

    side: int
    edge_weight: float
    graph: nx.Graph
    weighted_adjacency: np.ndarray
    laplacian: np.ndarray
    distances: np.ndarray

    @property
    def node_count(self) -> int:
        return int(self.side * self.side)

    @property
    def bus_count(self) -> int:
        return self.node_count

    @property
    def edge_count(self) -> int:
        return int(self.graph.number_of_edges())

    @property
    def diameter(self) -> int:
        return int(nx.diameter(self.graph))


@dataclass(frozen=True)
class MeshLQRModel:
    """Discrete-time LQR model on a weighted 2D mesh."""

    case: MeshCase
    eta: float
    dt: float
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    laplacian: np.ndarray
    distances: np.ndarray

    @property
    def bus_count(self) -> int:
        return self.case.node_count

    @property
    def state_dim(self) -> int:
        return int(self.A.shape[0])

    @property
    def input_dim(self) -> int:
        return int(self.B.shape[1])


def build_mesh_case(*, side: int = 10, edge_weight: float = 0.05) -> MeshCase:
    """Build the row-major square mesh used by Shin et al.'s `2d-mesh` case."""

    if side < 2:
        raise ValueError("side must be at least 2")
    if edge_weight <= 0.0:
        raise ValueError("edge_weight must be positive")

    raw_graph = nx.grid_2d_graph(side, side)
    mapping = {
        (row, col): row * side + col for row in range(side) for col in range(side)
    }
    graph = nx.relabel_nodes(raw_graph, mapping)
    nx.set_edge_attributes(graph, edge_weight, "weight")

    node_count = side * side
    weighted_adjacency = np.zeros((node_count, node_count), dtype=float)
    for source, target in graph.edges():
        weighted_adjacency[source, target] = edge_weight
        weighted_adjacency[target, source] = edge_weight
    laplacian = np.diag(weighted_adjacency.sum(axis=1)) - weighted_adjacency
    distances = _all_pairs_distances(graph, node_count)

    return MeshCase(
        side=side,
        edge_weight=edge_weight,
        graph=graph,
        weighted_adjacency=weighted_adjacency,
        laplacian=laplacian,
        distances=distances,
    )


def build_mesh_lqr_model(
    case: MeshCase | None = None,
    *,
    side: int = 10,
    edge_weight: float = 0.05,
    eta: float = 2.0**-1,
    dt: float = 1.0,
) -> MeshLQRModel:
    """Construct the discrete-time LQR model on a weighted 2D mesh."""

    if eta <= 0.0:
        raise ValueError("eta must be positive")
    if dt <= 0.0:
        raise ValueError("dt must be positive")

    mesh_case = case or build_mesh_case(side=side, edge_weight=edge_weight)
    node_count = mesh_case.node_count
    identity = np.eye(node_count)
    zeros = np.zeros((node_count, node_count))
    laplacian = mesh_case.laplacian

    A_stacked = np.block(
        [
            [identity, dt * identity],
            [zeros, identity - dt * laplacian],
        ]
    )
    B_stacked = np.vstack([zeros, dt * eta * identity])
    C_stacked = np.hstack([eta * identity, zeros])
    permutation = interleaved_two_component_permutation(node_count)
    A = A_stacked[np.ix_(permutation, permutation)]
    B = B_stacked[permutation, :]
    C = C_stacked[:, permutation]
    Q = C.T @ C
    R = identity.copy()

    return MeshLQRModel(
        case=mesh_case,
        eta=eta,
        dt=dt,
        A=A,
        B=B,
        C=C,
        Q=Q,
        R=R,
        laplacian=laplacian,
        distances=mesh_case.distances,
    )


def _all_pairs_distances(graph: nx.Graph, node_count: int) -> np.ndarray:
    lengths = dict(nx.all_pairs_shortest_path_length(graph))
    distances = np.zeros((node_count, node_count), dtype=int)
    for source, targets in lengths.items():
        for target, distance in targets.items():
            distances[source, target] = distance
    return distances


@dataclass(frozen=True)
class SLSResponse:
    """Finite prefix of state-feedback SLS response coefficients."""

    phi_x: list[np.ndarray]
    phi_u: list[np.ndarray]
    terminal_zero: bool = False

    @property
    def horizon(self) -> int:
        return len(self.phi_x) - 1


@dataclass(frozen=True)
class SLSResidual:
    delta: list[np.ndarray]


def dense_lqr_responses(
    A: np.ndarray,
    B: np.ndarray,
    K: np.ndarray,
    *,
    horizon: int,
) -> SLSResponse:
    """Generate centralized LQR Markov coefficients in the SLS convention."""

    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    nx = A.shape[0]
    nu = B.shape[1]
    closed_loop = A - B @ K
    phi_x = [np.zeros((nx, nx))]
    phi_u = [np.zeros((nu, nx))]
    phi_x.append(np.eye(nx))
    phi_u.append(-K.copy())

    for tau in range(2, horizon + 1):
        next_x = closed_loop @ phi_x[tau - 1]
        phi_x.append(next_x)
        phi_u.append(-K @ next_x)

    return SLSResponse(phi_x=phi_x, phi_u=phi_u, terminal_zero=False)


def compute_sls_residual(
    A: np.ndarray,
    B: np.ndarray,
    response: SLSResponse,
) -> SLSResidual:
    """Compute coefficients of [zI-A -B] Phi - I for a finite response."""

    nx = A.shape[0]
    horizon = response.horizon
    delta: list[np.ndarray] = [response.phi_x[1] - np.eye(nx)]
    last_dynamic_index = horizon if response.terminal_zero else horizon - 1

    for t in range(1, last_dynamic_index + 1):
        next_x = (
            np.zeros_like(response.phi_x[0]) if t == horizon else response.phi_x[t + 1]
        )
        delta.append(next_x - A @ response.phi_x[t] - B @ response.phi_u[t])

    return SLSResidual(delta=delta)


def residual_schur_bound(row_l1: float, column_l1: float) -> float:
    """Schur-type H-infinity upper bound from row/column block-l1 norms."""

    if row_l1 < 0.0 or column_l1 < 0.0:
        raise ValueError("row_l1 and column_l1 must be nonnegative")
    return math.sqrt(row_l1 * column_l1)


def closed_loop_tail_cost(
    closed_loop: np.ndarray,
    value_matrix: np.ndarray,
    *,
    start_power: int,
    normalize_by: int = 1,
) -> float:
    """Return sum_{t=start_power}^infty tr(F^{tT} S F^t).

    ``value_matrix`` is the Riccati/Lyapunov tail matrix satisfying
    ``P = S + F.T @ P @ F``.  With the SLS convention used here,
    ``Phi_x[1] = I`` is physical time 0, so a conic/FIR mask that keeps
    physical times ``0,...,T`` has temporal tail ``start_power = T + 1``.
    """

    if start_power < 0:
        raise ValueError("start_power must be nonnegative")
    if normalize_by <= 0:
        raise ValueError("normalize_by must be positive")
    transition = np.linalg.matrix_power(closed_loop, int(start_power))
    return float(np.trace(transition.T @ value_matrix @ transition)) / normalize_by


def h2_cost(
    response: SLSResponse,
    Q: np.ndarray,
    R: np.ndarray,
    *,
    normalize_by: int = 1,
) -> float:
    value = 0.0
    for tau in range(1, response.horizon + 1):
        value += float(np.trace(response.phi_x[tau].T @ Q @ response.phi_x[tau]))
        value += float(np.trace(response.phi_u[tau].T @ R @ response.phi_u[tau]))
    return value / normalize_by


def weighted_error_h2(
    reference: SLSResponse,
    candidate: SLSResponse,
    Q: np.ndarray,
    R: np.ndarray,
    *,
    normalize_by: int = 1,
) -> float:
    horizon = min(reference.horizon, candidate.horizon)
    value = 0.0
    for tau in range(1, horizon + 1):
        dx = candidate.phi_x[tau] - reference.phi_x[tau]
        du = candidate.phi_u[tau] - reference.phi_u[tau]
        value += float(np.trace(dx.T @ Q @ dx))
        value += float(np.trace(du.T @ R @ du))
    return value / normalize_by


def _block_slice(index: int, block_size: int) -> slice:
    start = index * block_size
    return slice(start, start + block_size)


@dataclass(frozen=True)
class ArchitectureComplexity:
    average_fan_in: float
    maximum_fan_in: int
    average_footprint: float
    maximum_footprint: int
    average_taps_per_controller: float
    maximum_taps_per_controller: int
    total_tap_blocks: int


def implementation_complexity(
    distances: np.ndarray,
    *,
    kappa: int,
    time_horizon: int,
    r_dyn: int = 1,
    kappa_bar: int | None = None,
) -> ArchitectureComplexity:
    """Count implementation resources for the saturated-cone support.

    ``kappa`` is the zero-delay halo. ``kappa_bar`` is the maximum footprint;
    when omitted, the pure-cone slice ``kappa + r_dyn * T`` is used.  The tap
    count is the number of admissible realization-filter coefficient blocks.
    """

    if r_dyn <= 0:
        raise ValueError("r_dyn must be positive")
    if kappa_bar is None:
        kappa_bar = kappa + r_dyn * time_horizon
    if kappa_bar < kappa:
        raise ValueError("kappa_bar must be at least kappa")

    finite = np.isfinite(distances)
    halo = finite & (distances <= kappa)
    footprint = finite & (distances <= kappa_bar)
    fan_in = np.sum(halo, axis=1)
    footprint_count = np.sum(footprint, axis=1)

    delayed_start = np.ceil(np.maximum(distances - kappa, 0.0) / r_dyn)
    allowed_lags = np.maximum(time_horizon - delayed_start + 1, 0)
    allowed_lags = np.where(footprint, allowed_lags, 0)
    taps = np.sum(allowed_lags, axis=1).astype(int)
    return ArchitectureComplexity(
        average_fan_in=float(np.mean(fan_in)),
        maximum_fan_in=int(np.max(fan_in)),
        average_footprint=float(np.mean(footprint_count)),
        maximum_footprint=int(np.max(footprint_count)),
        average_taps_per_controller=float(np.mean(taps)),
        maximum_taps_per_controller=int(np.max(taps)),
        total_tap_blocks=int(np.sum(taps)),
    )


@dataclass(frozen=True)
class TruncationScreeningMetrics:
    residual_l1_row: float
    residual_l1_col: float
    residual_schur_bound: float
    tail_l1: float
    nominal_prefix_cost: float
    response_l1: float


def lqr_fir_true_response_prefix_costs(
    nominal: SLSResponse,
    residual: SLSResidual,
    *,
    q: np.ndarray,
    r: np.ndarray,
    horizons: Sequence[int],
    normalize_by: int,
    closed_loop: np.ndarray,
    feedback_gain: np.ndarray,
) -> dict[int, float]:
    """Compute achieved-response prefix costs at multiple horizons in one pass."""

    requested_horizons = tuple(sorted(set(int(value) for value in horizons)))
    if not requested_horizons or requested_horizons[0] < 1:
        raise ValueError("horizons must contain positive integers")
    horizon = requested_horizons[-1]

    nx = nominal.phi_x[1].shape[0]
    delta0 = _matrix_at(residual.delta, 0, (nx, nx))
    solve_g0 = _make_g0_solver(delta0, nx)
    delta_terms = _nonzero_operator_terms(
        residual.delta,
        start=1,
        stop=min(len(residual.delta), horizon),
    )
    effective_horizon = max(
        (
            tau
            for tau in range(1, nominal.horizon + 1)
            if not _is_numerically_zero(nominal.phi_x[tau])
            or not _is_numerically_zero(nominal.phi_u[tau])
        ),
        default=0,
    )
    if effective_horizon < 1:
        return {value: 0.0 for value in requested_horizons}

    correction_x: list[tuple[int, np.ndarray | sparse.csr_matrix]] = []
    correction_u: list[tuple[int, np.ndarray | sparse.csr_matrix]] = []
    central_x = np.eye(nx)
    for tau in range(1, effective_horizon + 1):
        dx = nominal.phi_x[tau] - central_x
        du = nominal.phi_u[tau] + feedback_gain @ central_x
        converted_x = _as_nonzero_operator(dx)
        converted_u = _as_nonzero_operator(du)
        if converted_x is not None:
            correction_x.append((tau, converted_x))
        if converted_u is not None:
            correction_u.append((tau, converted_u))
        central_x = closed_loop @ central_x
    closed_loop_power_horizon = central_x

    max_delta_lag = max((lag for lag, _ in delta_terms), default=0)
    keep_depth = max(effective_horizon, max_delta_lag)
    h_window: dict[int, np.ndarray] = {0: solve_g0(np.eye(nx))}
    central_convolution_previous = np.zeros((nx, nx))
    value = 0.0
    requested_set = set(requested_horizons)
    prefix_costs: dict[int, float] = {}

    for order in range(horizon):
        if order > 0:
            accum = np.zeros((nx, nx))
            for lag, delta_lag in delta_terms:
                if lag > order:
                    break
                h_prev = h_window.get(order - lag)
                if h_prev is not None:
                    accum += delta_lag @ h_prev
            h_order = -solve_g0(accum)
            if not _is_numerically_zero(h_order):
                h_window[order] = h_order

        h_order = h_window.get(order)
        if h_order is None:
            h_order = np.zeros((nx, nx))
        central_convolution = h_order + closed_loop @ central_convolution_previous
        if order >= effective_horizon:
            old_h = h_window.get(order - effective_horizon)
            if old_h is not None:
                central_convolution -= closed_loop_power_horizon @ old_h

        x_value = central_convolution.copy()
        for tau, correction in correction_x:
            lag = order - tau + 1
            if lag < 0:
                break
            h_lag = h_window.get(lag)
            if h_lag is not None:
                x_value += correction @ h_lag

        u_value = -feedback_gain @ central_convolution
        for tau, correction in correction_u:
            lag = order - tau + 1
            if lag < 0:
                break
            h_lag = h_window.get(lag)
            if h_lag is not None:
                u_value += correction @ h_lag

        value += float(np.trace(x_value.T @ q @ x_value))
        value += float(np.trace(u_value.T @ r @ u_value))
        prefix_horizon = order + 1
        if prefix_horizon in requested_set:
            prefix_costs[prefix_horizon] = value / normalize_by
        central_convolution_previous = central_convolution

        min_keep = order - keep_depth
        if min_keep > 0:
            for key in tuple(h_window):
                if key < min_keep:
                    del h_window[key]

    return prefix_costs


def _make_g0_solver(delta0: np.ndarray, nx: int):
    if np.count_nonzero(delta0) == 0:
        return lambda rhs: rhs.copy()
    lu_and_piv = lu_factor(np.eye(nx) + delta0)
    return lambda rhs: lu_solve(lu_and_piv, rhs)


def _nonzero_operator_terms(
    matrices: list[np.ndarray],
    *,
    start: int,
    stop: int,
) -> list[tuple[int, np.ndarray | sparse.csr_matrix]]:
    terms: list[tuple[int, np.ndarray | sparse.csr_matrix]] = []
    for index in range(start, min(stop, len(matrices))):
        matrix = matrices[index]
        if matrix.size == 0:
            continue
        converted = _as_nonzero_operator(matrix)
        if converted is None:
            continue
        terms.append((index, converted))
    return terms


def _is_numerically_zero(matrix: np.ndarray, *, tolerance: float = 1.0e-14) -> bool:
    return matrix.size == 0 or float(np.max(np.abs(matrix))) <= tolerance


def _as_nonzero_operator(
    matrix: np.ndarray,
    *,
    tolerance: float = 1.0e-14,
) -> np.ndarray | sparse.csr_matrix | None:
    if _is_numerically_zero(matrix, tolerance=tolerance):
        return None
    cleaned = matrix.copy()
    cleaned[np.abs(cleaned) <= tolerance] = 0.0
    nnz = int(np.count_nonzero(cleaned))
    density = nnz / cleaned.size
    if density <= 0.25:
        return sparse.csr_matrix(cleaned)
    return cleaned


def fast_conic_truncate_response(
    response: SLSResponse,
    *,
    distances: np.ndarray,
    kappa: int,
    kappa_bar: int | None = None,
    time_horizon: int,
    state_block_size: int = 2,
    input_block_size: int = 1,
    r_dyn: int = 1,
) -> SLSResponse:
    """Vectorized equivalent of conic_truncate_response."""

    if kappa_bar is not None and kappa_bar < kappa:
        raise ValueError("kappa_bar must be at least kappa")
    bus_count = distances.shape[0]
    phi_x = [matrix.copy() for matrix in response.phi_x]
    phi_u = [matrix.copy() for matrix in response.phi_u]

    for tau in range(1, response.horizon + 1):
        physical_time = tau - 1
        if physical_time <= time_horizon:
            radius = kappa + r_dyn * physical_time
            if kappa_bar is not None:
                radius = min(radius, kappa_bar)
            keep_bus = distances <= radius
        else:
            keep_bus = np.zeros((bus_count, bus_count), dtype=bool)
        x_mask = np.repeat(
            np.repeat(keep_bus, state_block_size, axis=0), state_block_size, axis=1
        )
        u_mask = np.repeat(
            np.repeat(keep_bus, input_block_size, axis=0), state_block_size, axis=1
        )
        phi_x[tau] *= x_mask
        phi_u[tau] *= u_mask

    return SLSResponse(phi_x=phi_x, phi_u=phi_u, terminal_zero=True)


def fast_block_row_l1_norm(
    matrices: list[np.ndarray],
    *,
    bus_count: int,
    row_block_size: int,
    source_block_size: int = 2,
) -> float:
    """Vectorized maximum physical block-row l1 sum."""

    if not matrices:
        return 0.0
    row_sums = np.zeros(bus_count)
    for matrix in matrices:
        if matrix.size == 0:
            continue
        blocks = matrix.reshape(
            bus_count,
            row_block_size,
            bus_count,
            source_block_size,
        )
        row_sums += np.linalg.norm(blocks, axis=(1, 3)).sum(axis=1)
    return float(row_sums.max())


def fast_block_column_l1_norm(
    matrices: list[np.ndarray],
    *,
    bus_count: int,
    column_block_size: int,
    row_block_size: int = 2,
) -> float:
    """Vectorized maximum physical block-column l1 sum."""

    if not matrices:
        return 0.0
    column_sums = np.zeros(bus_count)
    for matrix in matrices:
        if matrix.size == 0:
            continue
        blocks = matrix.reshape(
            bus_count,
            row_block_size,
            bus_count,
            column_block_size,
        )
        column_sums += np.linalg.norm(blocks, axis=(1, 3)).sum(axis=0)
    return float(column_sums.max())


def fast_truncation_screening_metrics(
    response: SLSResponse,
    *,
    A: np.ndarray | sparse.spmatrix,
    B: np.ndarray | sparse.spmatrix,
    Q: np.ndarray,
    R: np.ndarray,
    distances: np.ndarray,
    kappa: int,
    kappa_bar: int | None = None,
    time_horizon: int,
    bus_count: int,
    normalize_by: int = 1,
    state_block_size: int = 2,
    input_block_size: int = 1,
    r_dyn: int = 1,
) -> TruncationScreeningMetrics:
    """Compute screening metrics without materializing truncated responses."""

    if kappa_bar is not None and kappa_bar < kappa:
        raise ValueError("kappa_bar must be at least kappa")
    if normalize_by <= 0:
        raise ValueError("normalize_by must be positive")

    mask_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    residual_row_sums = np.zeros(bus_count)
    residual_column_sums = np.zeros(bus_count)
    state_tail_row_sums = np.zeros(bus_count)
    input_tail_row_sums = np.zeros(bus_count)
    state_response_row_sums = np.zeros(bus_count)
    input_response_row_sums = np.zeros(bus_count)
    nominal_prefix_cost = 0.0
    q_diagonal = _diagonal_entries(Q)
    r_diagonal = _diagonal_entries(R)

    identity = np.eye(A.shape[0])
    previous_tail_x: np.ndarray | None = None
    previous_tail_u: np.ndarray | None = None
    previous_has_kept = False
    last_truncated_x: np.ndarray | None = None
    last_truncated_u: np.ndarray | None = None
    last_has_kept = False

    for tau in range(1, response.horizon + 1):
        radius = _support_radius(
            kappa,
            kappa_bar,
            time_horizon,
            physical_time=tau - 1,
            r_dyn=r_dyn,
        )
        has_kept = radius is not None
        dense_x = response.phi_x[tau]
        dense_u = response.phi_u[tau]
        if radius is None:
            truncated_x = np.zeros_like(dense_x)
            truncated_u = np.zeros_like(dense_u)
            tail_x = dense_x
            tail_u = dense_u
        else:
            state_mask, input_mask = _support_masks(
                distances,
                radius=radius,
                state_block_size=state_block_size,
                input_block_size=input_block_size,
                cache=mask_cache,
            )
            truncated_x = dense_x * state_mask
            truncated_u = dense_u * input_mask
            tail_x = dense_x - truncated_x
            tail_u = dense_u - truncated_u

        _accumulate_block_row_sums(
            tail_x,
            state_tail_row_sums,
            bus_count=bus_count,
            row_block_size=state_block_size,
            source_block_size=state_block_size,
        )
        _accumulate_block_row_sums(
            tail_u,
            input_tail_row_sums,
            bus_count=bus_count,
            row_block_size=input_block_size,
            source_block_size=state_block_size,
        )
        if has_kept:
            _accumulate_block_row_sums(
                truncated_x,
                state_response_row_sums,
                bus_count=bus_count,
                row_block_size=state_block_size,
                source_block_size=state_block_size,
            )
            _accumulate_block_row_sums(
                truncated_u,
                input_response_row_sums,
                bus_count=bus_count,
                row_block_size=input_block_size,
                source_block_size=state_block_size,
            )
            nominal_prefix_cost += _weighted_frobenius_cost(
                truncated_x,
                Q,
                diagonal=q_diagonal,
            )
            nominal_prefix_cost += _weighted_frobenius_cost(
                truncated_u,
                R,
                diagonal=r_diagonal,
            )

        if tau == 1:
            delta0 = truncated_x - identity
            _accumulate_residual_norm_sums(
                delta0,
                residual_row_sums,
                residual_column_sums,
                bus_count=bus_count,
                block_size=state_block_size,
            )
        elif previous_tail_x is not None and previous_tail_u is not None:
            if previous_has_kept or has_kept:
                delta = A @ previous_tail_x + B @ previous_tail_u - tail_x
                _accumulate_residual_norm_sums(
                    delta,
                    residual_row_sums,
                    residual_column_sums,
                    bus_count=bus_count,
                    block_size=state_block_size,
                )

        previous_tail_x = tail_x
        previous_tail_u = tail_u
        previous_has_kept = has_kept
        last_truncated_x = truncated_x
        last_truncated_u = truncated_u
        last_has_kept = has_kept

    if last_has_kept and last_truncated_x is not None and last_truncated_u is not None:
        terminal_delta = -(A @ last_truncated_x) - (B @ last_truncated_u)
        _accumulate_residual_norm_sums(
            terminal_delta,
            residual_row_sums,
            residual_column_sums,
            bus_count=bus_count,
            block_size=state_block_size,
        )

    residual_l1_row = float(residual_row_sums.max())
    residual_l1_col = float(residual_column_sums.max())
    return TruncationScreeningMetrics(
        residual_l1_row=residual_l1_row,
        residual_l1_col=residual_l1_col,
        residual_schur_bound=residual_schur_bound(residual_l1_row, residual_l1_col),
        tail_l1=max(
            float(state_tail_row_sums.max()),
            float(input_tail_row_sums.max()),
        ),
        nominal_prefix_cost=nominal_prefix_cost / normalize_by,
        response_l1=max(
            float(state_response_row_sums.max()),
            float(input_response_row_sums.max()),
        ),
    )


def _support_radius(
    kappa: int,
    kappa_bar: int | None,
    time_horizon: int,
    *,
    physical_time: int,
    r_dyn: int,
) -> int | None:
    if physical_time > time_horizon:
        return None
    radius = kappa + r_dyn * physical_time
    if kappa_bar is not None:
        radius = min(radius, kappa_bar)
    return radius


def _support_masks(
    distances: np.ndarray,
    *,
    radius: int,
    state_block_size: int,
    input_block_size: int,
    cache: dict[int, tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    if radius not in cache:
        keep_bus = distances <= radius
        cache[radius] = (
            np.repeat(
                np.repeat(keep_bus, state_block_size, axis=0),
                state_block_size,
                axis=1,
            ),
            np.repeat(
                np.repeat(keep_bus, input_block_size, axis=0),
                state_block_size,
                axis=1,
            ),
        )
    return cache[radius]


def _accumulate_residual_norm_sums(
    matrix: np.ndarray,
    row_sums: np.ndarray,
    column_sums: np.ndarray,
    *,
    bus_count: int,
    block_size: int,
) -> None:
    blocks = matrix.reshape(bus_count, block_size, bus_count, block_size)
    block_norms = np.linalg.norm(blocks, axis=(1, 3))
    row_sums += block_norms.sum(axis=1)
    column_sums += block_norms.sum(axis=0)


def _accumulate_block_row_sums(
    matrix: np.ndarray,
    row_sums: np.ndarray,
    *,
    bus_count: int,
    row_block_size: int,
    source_block_size: int,
) -> None:
    blocks = matrix.reshape(
        bus_count,
        row_block_size,
        bus_count,
        source_block_size,
    )
    row_sums += np.linalg.norm(blocks, axis=(1, 3)).sum(axis=1)


def _weighted_frobenius_cost(
    matrix: np.ndarray,
    weight: np.ndarray,
    *,
    diagonal: np.ndarray | None,
) -> float:
    if diagonal is not None:
        return float(np.einsum("i,ij,ij->", diagonal, matrix, matrix))
    return float(np.sum(matrix * (weight @ matrix)))


def _diagonal_entries(weight: np.ndarray) -> np.ndarray | None:
    diagonal = np.diag(weight)
    if np.allclose(weight, np.diag(diagonal)):
        return diagonal
    return None


def _matrix_at(
    matrices: list[np.ndarray],
    index: int,
    shape: tuple[int, int],
) -> np.ndarray:
    if 0 <= index < len(matrices):
        return matrices[index]
    return np.zeros(shape)


NATIVE_OSQP_SETTINGS = {
    "verbose": False,
    "eps_abs": 1e-9,
    "eps_rel": 1e-9,
    "eps_prim_inf": 1e-10,
    "eps_dual_inf": 1e-10,
    "max_iter": 100000,
    "polishing": True,
    "polish_refine_iter": 10,
    "adaptive_rho": True,
    "adaptive_rho_interval": 50,
    "scaled_termination": False,
    "check_termination": 25,
    "scaling": 10,
    "warm_starting": True,
}


@dataclass(frozen=True)
class _ColumnQP:
    status: str
    objective_value: float
    phi_x: list[np.ndarray]
    phi_u: list[np.ndarray]
    feasibility: str = "unknown"
    native_diagnostics: tuple[dict, ...] = ()


def _solve_column_qp_local_dynamics_native_osqp(
    A: np.ndarray,
    B: np.ndarray,
    q_sqrt: np.ndarray,
    r_sqrt: np.ndarray,
    *,
    distances: np.ndarray,
    source_bus: int,
    kappa: int,
    time_horizon: int,
    state_block_size: int,
    input_block_size: int,
    r_dyn: int,
    kappa_bar: int | None = None,
    settings: dict | None = None,
) -> _ColumnQP:
    source_dim = state_block_size
    nx = A.shape[0]
    nu = B.shape[1]

    actual_settings = {**NATIVE_OSQP_SETTINGS, **(settings or {})}
    data = _build_native_local_dynamics_qp(
        A,
        B,
        q_sqrt,
        r_sqrt,
        distances=distances,
        source_bus=source_bus,
        kappa=kappa,
        time_horizon=time_horizon,
        state_block_size=state_block_size,
        input_block_size=input_block_size,
        r_dyn=r_dyn,
        kappa_bar=kappa_bar,
    )
    solver = osqp.OSQP()
    first_rhs = data.rhs_by_component[0]
    try:
        solver.setup(
            P=data.P,
            q=np.zeros(data.P.shape[0]),
            A=data.Aeq,
            l=first_rhs,
            u=first_rhs,
            **actual_settings,
        )
    except Exception as exc:
        return _empty_column_qp(
            status="solver_error",
            objective_value=float("nan"),
            nx=nx,
            nu=nu,
            source_dim=source_dim,
            time_horizon=time_horizon,
            native_diagnostics=(
                {
                    "component": -1,
                    "raw_status": "setup_error",
                    "error": repr(exc),
                    "settings": actual_settings,
                },
            ),
        )
    component_solutions = []
    statuses = []
    diagnostics = []
    objective_value = 0.0
    for component, rhs in enumerate(data.rhs_by_component):
        try:
            solver.update_settings(
                polishing=actual_settings["polishing"],
                check_termination=actual_settings["check_termination"],
            )
            if component > 0:
                solver.update(l=rhs, u=rhs)
            # Reset between disturbance coordinates.
            solver.warm_start(
                x=np.zeros(data.P.shape[0]), y=np.zeros(data.Aeq.shape[0])
            )
            result = solver.solve(raise_error=False)
        except Exception as exc:
            statuses.append("solver_error")
            component_solutions.append(np.zeros(data.P.shape[0]))
            diagnostics.append(
                {
                    "component": component,
                    "raw_status": "solver_error",
                    "error": repr(exc),
                    "settings": actual_settings,
                }
            )
            continue
        status = _native_osqp_status(str(result.info.status))
        statuses.append(status)
        x = (
            np.asarray(result.x, dtype=float)
            if result.x is not None
            else np.full(data.P.shape[0], np.nan)
        )
        y = (
            np.asarray(result.y, dtype=float)
            if result.y is not None
            else np.full(data.Aeq.shape[0], np.nan)
        )
        primal = float(np.max(np.abs(data.Aeq @ x - rhs), initial=0.0))
        dual = float(np.max(np.abs(data.P @ x + data.Aeq.T @ y), initial=0.0))
        cert = getattr(result, "prim_inf_cert", None)
        certificate_valid = False
        cert_residual = cert_rhs = float("nan")
        if "infeasible" in status and cert is not None:
            cert = np.asarray(cert, dtype=float)
            norm = float(np.max(np.abs(cert), initial=0.0))
            if np.isfinite(norm) and norm > 0:
                cert = cert / norm
                cert_residual = float(np.max(np.abs(data.Aeq.T @ cert), initial=0.0))
                cert_rhs = float(rhs @ cert)
                certificate_valid = cert_residual <= 1e-9 and abs(cert_rhs) > 1e-7
        diagnostics.append(
            {
                "component": component,
                "raw_status": str(result.info.status),
                "status": status,
                "status_val": int(result.info.status_val),
                "iterations": int(result.info.iter),
                "solver_objective": float(result.info.obj_val),
                "solver_primal_residual": float(result.info.prim_res),
                "solver_dual_residual": float(result.info.dual_res),
                "constraint_residual_inf": primal,
                "stationarity_residual_inf": dual,
                "candidate_feasible": bool(np.isfinite(primal) and primal <= 1e-7),
                "certificate_valid_numerically": bool(certificate_valid),
                "certificate_atv_inf": cert_residual,
                "certificate_btv": cert_rhs,
                "polish_status": int(result.info.status_polish),
                "setup_time": float(result.info.setup_time),
                "solve_time": float(result.info.solve_time),
                "polish_time": float(result.info.polish_time),
                "run_time": float(result.info.run_time),
                "rho_updates": int(result.info.rho_updates),
                "rho_estimate": float(result.info.rho_estimate),
                "settings": actual_settings,
                "x": x,
                "y": y,
                "primal_infeasibility_certificate": cert
                if "infeasible" in status
                else None,
            }
        )
        component_solutions.append(x)
        objective_value += float(result.info.obj_val)

    phi_x, phi_u = _embed_native_local_dynamics_solution(
        component_solutions,
        data,
        nx=nx,
        nu=nu,
        source_dim=source_dim,
    )
    successful = all(s in {"optimal", "optimal_inaccurate"} for s in statuses)
    candidate_feasible = all(d.get("candidate_feasible", False) for d in diagnostics)
    return _ColumnQP(
        status=_combined_native_component_status(statuses),
        objective_value=objective_value
        if successful and candidate_feasible
        else float("nan"),
        phi_x=phi_x,
        phi_u=phi_u,
        feasibility=(
            "feasible"
            if candidate_feasible
            else "infeasible_certificate"
            if any(d.get("certificate_valid_numerically", False) for d in diagnostics)
            else "unknown"
        ),
        native_diagnostics=tuple(diagnostics),
    )


@dataclass(frozen=True)
class _NativeLocalDynamicsQP:
    P: sparse.csc_matrix
    Aeq: sparse.csc_matrix
    rhs_by_component: list[np.ndarray]
    x_rows_by_tau: list[np.ndarray]
    u_rows_by_tau: list[np.ndarray]
    x_offsets: list[int]
    u_offsets: list[int]


def _build_native_local_dynamics_qp(
    A: np.ndarray,
    B: np.ndarray,
    q_sqrt: np.ndarray,
    r_sqrt: np.ndarray,
    *,
    distances: np.ndarray,
    source_bus: int,
    kappa: int,
    time_horizon: int,
    state_block_size: int,
    input_block_size: int,
    r_dyn: int,
    kappa_bar: int | None,
) -> _NativeLocalDynamicsQP:
    source_dim = state_block_size
    x_rows_by_tau: list[np.ndarray] = [np.asarray([], dtype=int)]
    u_rows_by_tau: list[np.ndarray] = [np.asarray([], dtype=int)]
    x_offsets = [-1] * (time_horizon + 1)
    u_offsets = [-1] * (time_horizon + 1)
    objective_blocks = []
    variable_count = 0

    for tau in range(1, time_horizon + 1):
        x_mask, u_mask = _native_qp_support_masks(
            distances,
            source_bus=source_bus,
            tau=tau,
            kappa=kappa,
            kappa_bar=kappa_bar,
            state_block_size=state_block_size,
            input_block_size=input_block_size,
            r_dyn=r_dyn,
        )
        x_rows = np.flatnonzero(x_mask)
        u_rows = np.flatnonzero(u_mask)
        x_rows_by_tau.append(x_rows)
        u_rows_by_tau.append(u_rows)

        x_offsets[tau] = variable_count
        objective_blocks.append(_native_weight_block(q_sqrt, x_rows))
        variable_count += len(x_rows)

        u_offsets[tau] = variable_count
        objective_blocks.append(_native_weight_block(r_sqrt, u_rows))
        variable_count += len(u_rows)

    P = sparse.block_diag(objective_blocks, format="csc")
    if P.shape != (variable_count, variable_count):
        P = sparse.csc_matrix((variable_count, variable_count))

    A_sparse = sparse.csr_matrix(A)
    B_sparse = sparse.csr_matrix(B)
    constraint_rows: list[int] = []
    constraint_cols: list[int] = []
    constraint_data: list[float] = []
    rhs_entries: list[tuple[int, int, float]] = []
    row_cursor = 0

    source_rows = _block_slice(source_bus, state_block_size)
    for local_index, global_row in enumerate(x_rows_by_tau[1]):
        constraint_rows.append(row_cursor + local_index)
        constraint_cols.append(x_offsets[1] + local_index)
        constraint_data.append(1.0)
        if source_rows.start <= int(global_row) < source_rows.stop:
            rhs_entries.append(
                (
                    row_cursor + local_index,
                    int(global_row) - source_rows.start,
                    1.0,
                )
            )
    row_cursor += len(x_rows_by_tau[1])

    for tau in range(1, time_horizon):
        row_set = _dynamics_row_set(
            A,
            B,
            x_rows=x_rows_by_tau[tau],
            u_rows=u_rows_by_tau[tau],
            next_x_rows=x_rows_by_tau[tau + 1],
        )
        _add_next_state_selector(
            constraint_rows,
            constraint_cols,
            constraint_data,
            row_start=row_cursor,
            col_start=x_offsets[tau + 1],
            selected_rows=row_set,
            local_rows=x_rows_by_tau[tau + 1],
        )
        _add_sparse_matrix_block(
            constraint_rows,
            constraint_cols,
            constraint_data,
            row_start=row_cursor,
            col_start=x_offsets[tau],
            matrix=A_sparse[row_set][:, x_rows_by_tau[tau]],
            scale=-1.0,
        )
        _add_sparse_matrix_block(
            constraint_rows,
            constraint_cols,
            constraint_data,
            row_start=row_cursor,
            col_start=u_offsets[tau],
            matrix=B_sparse[row_set][:, u_rows_by_tau[tau]],
            scale=-1.0,
        )
        row_cursor += len(row_set)

    terminal_rows = _dynamics_row_set(
        A,
        B,
        x_rows=x_rows_by_tau[time_horizon],
        u_rows=u_rows_by_tau[time_horizon],
        next_x_rows=None,
    )
    _add_sparse_matrix_block(
        constraint_rows,
        constraint_cols,
        constraint_data,
        row_start=row_cursor,
        col_start=x_offsets[time_horizon],
        matrix=A_sparse[terminal_rows][:, x_rows_by_tau[time_horizon]],
        scale=1.0,
    )
    _add_sparse_matrix_block(
        constraint_rows,
        constraint_cols,
        constraint_data,
        row_start=row_cursor,
        col_start=u_offsets[time_horizon],
        matrix=B_sparse[terminal_rows][:, u_rows_by_tau[time_horizon]],
        scale=1.0,
    )
    row_cursor += len(terminal_rows)

    Aeq = sparse.coo_matrix(
        (constraint_data, (constraint_rows, constraint_cols)),
        shape=(row_cursor, variable_count),
    ).tocsc()
    rhs_by_component = [np.zeros(row_cursor, dtype=float) for _ in range(source_dim)]
    for row_index, component, value in rhs_entries:
        rhs_by_component[component][row_index] = value
    return _NativeLocalDynamicsQP(
        P=P,
        Aeq=Aeq,
        rhs_by_component=rhs_by_component,
        x_rows_by_tau=x_rows_by_tau,
        u_rows_by_tau=u_rows_by_tau,
        x_offsets=x_offsets,
        u_offsets=u_offsets,
    )


def _native_weight_block(
    weight_sqrt: np.ndarray, rows: np.ndarray
) -> sparse.csc_matrix:
    if len(rows) == 0:
        return sparse.csc_matrix((0, 0))
    if _is_diagonal_matrix(weight_sqrt):
        weights = np.diag(weight_sqrt)[rows]
        return sparse.diags(2.0 * np.square(weights), format="csc")
    weight = sparse.csc_matrix(weight_sqrt[:, rows])
    return (2.0 * (weight.T @ weight)).tocsc()


def _add_sparse_matrix_block(
    target_rows: list[int],
    target_cols: list[int],
    target_data: list[float],
    *,
    row_start: int,
    col_start: int,
    matrix: sparse.spmatrix,
    scale: float,
) -> None:
    coo = matrix.tocoo()
    if coo.nnz == 0:
        return
    target_rows.extend((row_start + coo.row).tolist())
    target_cols.extend((col_start + coo.col).tolist())
    target_data.extend((scale * coo.data).tolist())


def _add_next_state_selector(
    target_rows: list[int],
    target_cols: list[int],
    target_data: list[float],
    *,
    row_start: int,
    col_start: int,
    selected_rows: np.ndarray,
    local_rows: np.ndarray,
) -> None:
    local_index = {int(row): index for index, row in enumerate(local_rows)}
    for output_index, row in enumerate(selected_rows):
        index = local_index.get(int(row))
        if index is None:
            continue
        target_rows.append(row_start + output_index)
        target_cols.append(col_start + index)
        target_data.append(1.0)


def _embed_native_local_dynamics_solution(
    component_solutions: Sequence[np.ndarray],
    data: _NativeLocalDynamicsQP,
    *,
    nx: int,
    nu: int,
    source_dim: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    time_horizon = len(data.x_rows_by_tau) - 1
    phi_x = [np.zeros((nx, source_dim)) for _ in range(time_horizon + 1)]
    phi_u = [np.zeros((nu, source_dim)) for _ in range(time_horizon + 1)]
    for component, solution in enumerate(component_solutions):
        for tau in range(1, time_horizon + 1):
            x_rows = data.x_rows_by_tau[tau]
            u_rows = data.u_rows_by_tau[tau]
            x_offset = data.x_offsets[tau]
            u_offset = data.u_offsets[tau]
            phi_x[tau][x_rows, component] = solution[x_offset : x_offset + len(x_rows)]
            phi_u[tau][u_rows, component] = solution[u_offset : u_offset + len(u_rows)]
    return phi_x, phi_u


def _native_osqp_status(status: str) -> str:
    normalized = status.strip().lower()
    if normalized == "solved":
        return "optimal"
    if normalized == "solved inaccurate":
        return "optimal_inaccurate"
    if normalized == "primal infeasible":
        return "infeasible"
    if normalized == "primal infeasible inaccurate":
        return "infeasible_inaccurate"
    if normalized == "dual infeasible":
        return "unbounded"
    if normalized == "dual infeasible inaccurate":
        return "unbounded_inaccurate"
    return normalized.replace(" ", "_")


def _combined_native_component_status(statuses: Sequence[str]) -> str:
    if all(status == "optimal" for status in statuses):
        return "optimal"
    if all(status in {"optimal", "optimal_inaccurate"} for status in statuses):
        return "optimal_inaccurate"
    unique = set(statuses)
    return next(iter(unique)) if len(unique) == 1 else "mixed"


def _empty_column_qp(
    *,
    status: str,
    objective_value: float,
    nx: int,
    nu: int,
    source_dim: int,
    time_horizon: int,
    native_diagnostics: tuple[dict, ...] = (),
) -> _ColumnQP:
    return _ColumnQP(
        status=status,
        objective_value=objective_value,
        phi_x=[np.zeros((nx, source_dim)) for _ in range(time_horizon + 1)],
        phi_u=[np.zeros((nu, source_dim)) for _ in range(time_horizon + 1)],
        native_diagnostics=native_diagnostics,
    )


def _dynamics_row_set(
    A: np.ndarray,
    B: np.ndarray,
    *,
    x_rows: np.ndarray,
    u_rows: np.ndarray,
    next_x_rows: np.ndarray | None,
) -> np.ndarray:
    active = np.zeros(A.shape[0], dtype=bool)
    if len(x_rows):
        active |= np.any(np.abs(A[:, x_rows]) > 1e-12, axis=1)
    if len(u_rows):
        active |= np.any(np.abs(B[:, u_rows]) > 1e-12, axis=1)
    if next_x_rows is not None and len(next_x_rows):
        active[next_x_rows] = True
    return np.flatnonzero(active)


def _is_diagonal_matrix(matrix: np.ndarray) -> bool:
    return np.allclose(matrix, np.diag(np.diag(matrix)))


def _native_qp_support_masks(
    distances: np.ndarray,
    *,
    source_bus: int,
    tau: int,
    kappa: int,
    kappa_bar: int | None = None,
    state_block_size: int,
    input_block_size: int,
    r_dyn: int,
) -> tuple[np.ndarray, np.ndarray]:
    physical_time = tau - 1
    radius = kappa + r_dyn * physical_time
    if kappa_bar is not None:
        if kappa_bar < kappa:
            raise ValueError("kappa_bar must be at least kappa")
        radius = min(radius, kappa_bar)
    keep_bus = distances[:, source_bus] <= radius
    x_mask = np.repeat(keep_bus, state_block_size)
    u_mask = np.repeat(keep_bus, input_block_size)
    return x_mask, u_mask


def _psd_sqrt(matrix: np.ndarray) -> np.ndarray:
    diagonal = np.diag(matrix)
    if np.allclose(matrix, np.diag(diagonal)):
        return np.diag(np.sqrt(np.maximum(diagonal, 0.0)))
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return (
        eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0))) @ eigenvectors.T
    )


def thermal_mesh_feasibility(model, *, kappa, kappa_bar, response_horizon, source):
    """Return an explicit obstruction or a constructive feasible response route.

    At distance d, the leading temperature response at index d+1 is
    (# shortest paths) (dt*edge_weight)^d. With kappa=0, no admissible
    input can alter this leading front; its integral appears one step later.
    Once the source cone reaches every node, two steps with K_d finish it.
    For kappa>=1, the two-step, radius-one K_d is admissible immediately.
    """
    n = model.bus_count
    # Authenticate this special structural argument against the actual plant.
    A, B = model.A, model.B
    if not (
        np.array_equal(A[0::2, 0::2], np.eye(n))
        and np.array_equal(A[0::2, 1::2], model.dt * np.eye(n))
        and np.count_nonzero(A[1::2, 0::2]) == 0
        and np.array_equal(A[1::2, 1::2], np.eye(n) - model.dt * model.laplacian)
        and np.count_nonzero(B[0::2]) == 0
        and np.array_equal(B[1::2], model.dt * model.eta * np.eye(n))
    ):
        return {
            "classification": "unknown",
            "reason": "plant does not match thermal mesh structure",
        }
    h = int(response_horizon)
    ecc = int(np.max(model.distances[:, source]))
    base = {
        "source": source,
        "eccentricity": ecc,
        "response_horizon": h,
        "last_filter_tap": h - 1,
    }
    if h == 1:
        return {
            **base,
            "classification": "infeasible_structural",
            "witness": "initial_integral",
            "component": 0,
            "target": source,
            "response_index": 2,
            "forced_value_exact": "1",
            "forced_value": 1.0,
            "reason": "Phi_x[1]=I and B has zero integral rows, so terminal integral cannot vanish",
        }
    if kappa >= 1:
        return {
            **base,
            "classification": "feasible_constructive",
            "reason": "radius-one K_d has (A-BK_d)^2=0; H>=2 and kappa_bar>=kappa>=1",
        }
    if kappa != 0 or h < 1:
        return {
            **base,
            "classification": "unknown",
            "reason": "outside the parameter range of the feasibility criterion",
        }
    if (kappa_bar is None or kappa_bar >= ecc) and h >= ecc + 2:
        return {
            **base,
            "classification": "feasible_constructive",
            "reason": "wait ecc steps, then two deadbeat steps; full source footprint is available",
        }
    if kappa_bar is not None and kappa_bar < ecc and kappa_bar + 2 <= h:
        d = kappa_bar + 1
        index, kind, component_row = d + 1, "outside_mask_temperature", 1
    elif h <= ecc:
        d = h
        index, kind, component_row = h + 1, "terminal_temperature", 1
    else:
        d = h - 1
        index, kind, component_row = h + 1, "terminal_integral", 0
    target = int(np.flatnonzero(model.distances[:, source] == d)[0])
    sr, sc = divmod(source, model.case.side)
    tr, tc = divmod(target, model.case.side)
    paths = comb(d, abs(sr - tr))
    dt = Fraction(str(model.dt))
    coefficient = paths * (dt * Fraction(str(model.case.edge_weight))) ** d
    if component_row == 0:
        coefficient *= dt
    return {
        **base,
        "classification": "infeasible_structural",
        "witness": kind,
        "component": 1,
        "target": target,
        "state_component": component_row,
        "distance": d,
        "shortest_paths": paths,
        "response_index": index,
        "forced_value_exact": str(coefficient),
        "forced_value": float(coefficient),
        "reason": "admissible controls lag the leading temperature front by one step; positive shortest-path contributions cannot cancel",
    }


@dataclass(frozen=True)
class Architecture:
    """Immediate radius kappa, maximum footprint kappa_bar, and memory T.

    At filter tap t in 0,...,T, permitted distance is
    min(kappa_bar, kappa+t). The propagation speed is one graph hop per update.
    """

    kappa: int
    kappa_bar: int
    memory: int

    def __post_init__(self):
        for value in (self.kappa, self.kappa_bar, self.memory):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    "Control architecture parameters must be nonnegative integers"
                )
        if self.kappa_bar < self.kappa:
            raise ValueError("kappa_bar must be at least kappa")

    @property
    def response_horizon(self) -> int:
        return self.memory + 1


def build_mesh(side: int = 10):
    """Build the paper's positive thermal mesh (edge=0.05, eta=1/16, dt=1)."""
    if isinstance(side, bool) or not isinstance(side, int) or side < 2:
        raise ValueError("side must be an integer at least two")
    return build_mesh_lqr_model(side=side, edge_weight=0.05, eta=2**-4, dt=1.0)


def _resources(model, architecture):
    return asdict(
        implementation_complexity(
            model.distances,
            kappa=architecture.kappa,
            kappa_bar=architecture.kappa_bar,
            time_horizon=architecture.memory,
            r_dyn=1,
        )
    )


def evaluate_direct(
    model, architecture: Architecture, *, prefix_lengths: Sequence[int] = (720, 880)
):
    """Evaluate local truncation filters and their achieved response cost.

    Dynamic cost is a finite-prefix estimate, not a certified infinite-sum
    upper bound. The local filters can produce a nonlocal, infinite response.
    An inconclusive small-gain test leaves cost unevaluated.
    """
    lengths = tuple(sorted(set(prefix_lengths)))
    if not lengths or any(
        not isinstance(h, int) or h <= architecture.response_horizon for h in lengths
    ):
        raise ValueError("Each response prefix must exceed T+1")
    solution = solve_lqr(model)
    dense = dense_lqr_responses(
        model.A, model.B, solution.K, horizon=architecture.response_horizon + 1
    )
    filters = fast_conic_truncate_response(
        dense,
        distances=model.distances,
        kappa=architecture.kappa,
        kappa_bar=architecture.kappa_bar,
        time_horizon=architecture.memory,
        state_block_size=2,
        input_block_size=1,
        r_dyn=1,
    )
    residual = compute_sls_residual(model.A, model.B, filters)
    row = fast_block_row_l1_norm(
        residual.delta, bus_count=model.bus_count, row_block_size=2
    )
    col = fast_block_column_l1_norm(
        residual.delta, bus_count=model.bus_count, column_block_size=2
    )
    gamma = float(np.sqrt(row * col))
    weights = np.tile([0.25, 1.0], model.bus_count)
    scaled = float(
        sum(svdvals(d * weights[:, None] / weights[None, :])[0] for d in residual.delta)
    )
    result = dict(
        route="direct",
        memory=architecture.memory,
        response_horizon=architecture.response_horizon,
        resources=_resources(model, architecture),
        gamma=gamma,
        scaled_small_gain=scaled,
        stability_certified=gamma < 1 or scaled < 1,
        relative_losses={},
        cost_kind="finite_prefix_estimate",
    )
    if architecture.memory == 0:
        # Static filters give u=Phi_u[1] x. This is an infinite-response cost.
        feedback = filters.phi_u[1]
        closed_loop = model.A + model.B @ feedback
        radius = float(max(abs(np.linalg.eigvals(closed_loop))))
        result.update(
            spectral_radius=radius, stability_certified=radius < 1, cost_kind="lyapunov"
        )
        if radius < 1:
            value = solve_discrete_lyapunov(
                closed_loop.T, model.Q + feedback.T @ model.R @ feedback
            )
            result["relative_loss"] = float(np.trace(value) / np.trace(solution.P) - 1)
    elif result["stability_certified"]:
        costs = lqr_fir_true_response_prefix_costs(
            filters,
            residual,
            q=model.Q,
            r=model.R,
            horizons=lengths,
            normalize_by=model.bus_count,
            closed_loop=model.A - model.B @ solution.K,
            feedback_gain=solution.K,
        )
        result["relative_losses"] = {
            str(h): float(cost / (np.trace(solution.P) / model.bus_count) - 1)
            for h, cost in costs.items()
        }
    return result


def solve_exact(
    model,
    architecture: Architecture,
    *,
    sources: Sequence[int] | None = None,
    settings: dict | None = None,
):
    """Solve the localized SLS quadratic programs for the requested sources.

    Returns (summary, response). A selected-source result is a diagnostic;
    its cost denominator uses those same sources. Structural infeasibility
    returns no response. Numerical failure is reported as inconclusive.
    """
    selected = list(range(model.bus_count)) if sources is None else list(sources)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(
            not isinstance(j, int)
            or isinstance(j, bool)
            or j < 0
            or j >= model.bus_count
            for j in selected
        )
    ):
        raise ValueError("sources must be distinct node indices within the mesh")
    witnesses = [
        thermal_mesh_feasibility(
            model,
            kappa=architecture.kappa,
            kappa_bar=architecture.kappa_bar,
            response_horizon=architecture.response_horizon,
            source=j,
        )
        for j in selected
    ]
    result = dict(
        route="exact",
        memory=architecture.memory,
        response_horizon=architecture.response_horizon,
        source_nodes=selected,
        complete_controller=len(selected) == model.bus_count,
        resources=_resources(model, architecture),
        structural_checks=witnesses,
        relative_loss=None,
        paper_cost_per_source=None,
    )
    if any(w["classification"] == "infeasible_structural" for w in witnesses):
        result["classification"] = "infeasible_structural"
        return result, None
    if not all(w["classification"] == "feasible_constructive" for w in witnesses):
        result["classification"] = "unknown_structure"
        return result, None
    options = dict(eps_abs=1e-10, eps_rel=1e-10, polishing=False, time_limit=3600)
    options.update(settings or {})
    horizon = architecture.response_horizon
    phi_x = [np.zeros((model.state_dim, model.state_dim)) for _ in range(horizon + 1)]
    phi_u = [np.zeros((model.input_dim, model.state_dim)) for _ in range(horizon + 1)]
    diagnostics = []
    accepted = True
    solver_energy = 0.0
    for source in selected:
        column = _solve_column_qp_local_dynamics_native_osqp(
            model.A,
            model.B,
            _psd_sqrt(model.Q),
            _psd_sqrt(model.R),
            distances=model.distances,
            source_bus=source,
            kappa=architecture.kappa,
            time_horizon=horizon,
            state_block_size=2,
            input_block_size=1,
            r_dyn=1,
            kappa_bar=architecture.kappa_bar,
            settings=options,
        )
        component_ok = all(
            d.get("status") == "optimal"
            and d.get("constraint_residual_inf", np.inf) <= 1e-7
            and d.get("stationarity_residual_inf", np.inf) <= 1e-7
            for d in column.native_diagnostics
        )
        accepted &= component_ok and column.feasibility == "feasible"
        solver_energy += column.objective_value
        for t in range(horizon + 1):
            phi_x[t][:, 2 * source : 2 * source + 2] = column.phi_x[t]
            phi_u[t][:, 2 * source : 2 * source + 2] = column.phi_u[t]
        diagnostics.append(
            dict(
                source=source,
                status=column.status,
                components=[
                    {
                        k: v
                        for k, v in d.items()
                        if k not in ("x", "y", "primal_infeasibility_certificate")
                    }
                    for d in column.native_diagnostics
                ],
            )
        )
    response = SLSResponse(phi_x=phi_x, phi_u=phi_u, terminal_zero=True)
    # Recompute the selected-column equality, including initial and terminal rows.
    indices = [coordinate for j in selected for coordinate in (2 * j, 2 * j + 1)]
    residuals = [phi_x[1][:, indices] - np.eye(model.state_dim)[:, indices]]
    for t in range(1, horizon + 1):
        next_x = phi_x[t + 1] if t < horizon else np.zeros_like(phi_x[t])
        residuals.append((next_x - model.A @ phi_x[t] - model.B @ phi_u[t])[:, indices])
    residual_inf = float(max(np.max(abs(r)) for r in residuals))
    block_row_sums = sum(
        np.linalg.norm(
            r.reshape(model.bus_count, 2, len(selected), 2), axis=(1, 3)
        ).sum(axis=1)
        for r in residuals
    )
    energy = float(
        sum(
            np.sum(x * (model.Q @ x)) + np.sum(u * (model.R @ u))
            for x, u in zip(phi_x, phi_u)
        )
    )
    central = solve_lqr(model).P
    denominator = float(sum(central[i, i] for i in indices))
    accepted = bool(accepted and residual_inf <= 1e-7 and np.isfinite(energy))
    result.update(
        classification="feasible" if accepted else "numerically_inconclusive",
        equality_residual_inf=residual_inf,
        equality_residual_block_row_l1=float(max(block_row_sums)),
        solver_vs_response_energy_abs=float(abs(solver_energy - energy)),
        numerical_diagnostics=diagnostics,
        settings=options,
    )
    if accepted:
        result.update(
            relative_loss=energy / denominator - 1,
            paper_cost_per_source=energy / (2 * len(selected)),
        )
    return result, response
