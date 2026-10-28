"""Heston ADI structured-product pricing, node Greeks and Rannacher smoothing.

The public pricing functions return scalar prices, or the solved surface with
return_surface=True. calculate_greeks differentiates that single surface at
(S0, v0) instead of repricing bumped inputs. Dividend input may be a constant
yield, a DividendCarryCurve, or an effective-yield array accompanied by
matching maturities.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from heston_calibration import DividendCarryCurve

try:
    from numba import njit
except ImportError:  # numba is optional: fall back to plain Python

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]

        def _noop(fn):
            return fn

        return _noop


__all__ = [
    "calculate_greeks",
    "dividend_rate_for_tau_step",
    "heston_adi_step",
    "node_greeks",
    "prepare_dividend_input",
    "price_autocallable_heston",
    "price_phoenix_heston",
    "price_snowball_heston",
]


def prepare_dividend_input(q, q_maturities=None):
    """
    Normalize the public dividend input once per pricing call.

    Accepted forms:
      * scalar q: original constant-yield behavior;
      * DividendCarryCurve (or compatible object with cumulative_carry);
      * 1D q array together with q_maturities.
    """
    if hasattr(q, "cumulative_carry"):
        return q
    if np.ndim(q) == 0:
        q_scalar = float(q)
        if not np.isfinite(q_scalar):
            raise ValueError("scalar q must be finite")
        return q_scalar
    if q_maturities is None:
        raise ValueError("q_maturities is required when q is an array")
    return DividendCarryCurve(q_maturities, q)


def dividend_rate_for_tau_step(q_input, T, tau_start, tau_end):
    """
    Dividend rate used by one backward-time ADI step.

    tau runs from maturity back toward valuation time.  The step
    [tau_start, tau_end] corresponds to calendar interval
    [T-tau_end, T-tau_start].  A curve uses the exact interval-average rate
    derived from Q; a scalar returns the original constant q unchanged.
    """
    if np.ndim(q_input) == 0 and not hasattr(q_input, "cumulative_carry"):
        return float(q_input)
    calendar_start = max(0.0, float(T) - float(tau_end))
    calendar_end = max(0.0, float(T) - float(tau_start))
    if hasattr(q_input, "interval_average_rate"):
        return float(q_input.interval_average_rate(calendar_start, calendar_end))
    if hasattr(q_input, "cumulative_carry"):
        carry_change = q_input.cumulative_carry(
            calendar_end
        ) - q_input.cumulative_carry(calendar_start)
        return float(carry_change / (calendar_end - calendar_start))
    raise TypeError("q_input must be a scalar or provide cumulative_carry()")


@njit(cache=True)
def thomas_algorithm(a, b, c, d):
    """Solve tridiagonal system Mx=d (scalar, kept for compatibility)."""
    n = len(b)
    cp = np.zeros(n - 1)
    dp = np.zeros(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n - 1):
        denom = b[i] - a[i - 1] * cp[i - 1]
        cp[i] = c[i] / denom
        dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / denom
    dp[n - 1] = (d[n - 1] - a[n - 2] * dp[n - 2]) / (b[n - 1] - a[n - 2] * cp[n - 2])
    x = np.zeros(n)
    x[n - 1] = dp[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


@njit(cache=True)
def thomas_algorithm_batch(a, b, c, d):
    """Batched Thomas solve: a (B,n-1), b (B,n), c (B,n-1), d (B,n) -> x (B,n)."""
    n = b.shape[1]
    cp = np.empty_like(c)
    dp = np.empty_like(d)
    cp[:, 0] = c[:, 0] / b[:, 0]
    dp[:, 0] = d[:, 0] / b[:, 0]
    for i in range(1, n - 1):
        denom = b[:, i] - a[:, i - 1] * cp[:, i - 1]
        cp[:, i] = c[:, i] / denom
        dp[:, i] = (d[:, i] - a[:, i - 1] * dp[:, i - 1]) / denom
    dp[:, n - 1] = (d[:, n - 1] - a[:, n - 2] * dp[:, n - 2]) / (
        b[:, n - 1] - a[:, n - 2] * cp[:, n - 2]
    )
    x = np.empty_like(d)
    x[:, n - 1] = dp[:, n - 1]
    for i in range(n - 2, -1, -1):
        x[:, i] = dp[:, i] - cp[:, i] * x[:, i + 1]
    return x


# ---------- Prepared Douglas ADI operator ----------
@njit(cache=True)
def _prepare_adi(dt, S, dS, v, dv, r, q, kappa, theta, sigma_v, weight=0.5):
    """Factor invariant spot matrices and the shared variance matrix once.

    weight is the implicit-correction weight: 0.5 for the Crank-Nicolson step,
    1.0 for a backward-Euler half-step. Only the product weight*dt enters.
    """
    n, nv = len(S) - 2, len(v)
    half = weight * dt
    ds2, dv2 = dS * dS, dv * dv
    forward = (r - q) * S[1] >= 0.
    lower_s = np.empty((n, nv))
    denominator_s = np.empty((n, nv))
    cp_s = np.empty((n - 1, nv))
    for row in range(n):
        i = row + 1
        drift = (r - q) * S[i]
        if forward:
            lm, li, lp = 0., -drift / dS, drift / dS
        else:
            lm, li, lp = -drift / dS, drift / dS, 0.
        for j in range(nv):
            D = .5 * v[j] * (S[i] * S[i]) / ds2
            a = -half * (lm + D)
            b = 1. - half * (li - 2. * D)
            c = -half * (lp + D)
            if row == n - 1:
                a -= c
                b += 2. * c
            lower_s[row, j] = a
            denominator = b if row == 0 else b - a * cp_s[row - 1, j]
            denominator_s[row, j] = denominator
            if row < n - 1:
                cp_s[row, j] = c / denominator

    drift_v = kappa * (theta - v)
    diffusion_v = .5 * sigma_v * sigma_v * v
    lower_v = np.empty(nv)
    denominator_v = np.empty(nv)
    cp_v = np.empty(nv - 1)
    for j in range(nv):
        if j == 0:
            lm, li, lp = 0., -drift_v[0] / dv, drift_v[0] / dv
        elif j == nv - 1:
            lm, li, lp = -drift_v[-1] / dv, drift_v[-1] / dv, 0.
        else:
            D = .5 * sigma_v * sigma_v * v[j] / dv2
            lm = -drift_v[j] / (2. * dv) + D
            li = -2. * D
            lp = drift_v[j] / (2. * dv) + D
        a = -half * lm
        b = 1. - half * li
        c = -half * lp
        lower_v[j] = a
        denominator = b if j == 0 else b - a * cp_v[j - 1]
        denominator_v[j] = denominator
        if j < nv - 1:
            cp_v[j] = c / denominator
    return lower_s, denominator_s, cp_s, lower_v, denominator_v, cp_v, drift_v, diffusion_v


@njit(cache=True)
def _adi_step_prepared(U, dt, S, dS, v, dv, r, q, kappa, theta, sigma_v, rho,
                       lower_boundary, factors, lv, rhs, result, weight=0.5):
    """Use fixed factors and scratch; finish all input stencils before output."""
    ns, nv = U.shape
    n = ns - 2
    ds2, dv2 = dS * dS, dv * dv
    half = weight * dt
    forward = (r - q) * S[1] >= 0.
    lower_s, denominator_s, cp_s, lower_v, denominator_v, cp_v, drift_v, diffusion_v = factors

    # Original explicit operators, computed one cell at a time.
    for row in range(n):
        i = row + 1
        s2 = S[i] * S[i]
        drift = (r - q) * S[i]
        for j in range(nv):
            ds = ((U[i + 1, j] - U[i, j]) / dS if forward
                  else (U[i, j] - U[i - 1, j]) / dS)
            dss = (U[i + 1, j] - 2. * U[i, j] + U[i - 1, j]) / ds2
            ls = drift * ds + .5 * (v[j] * s2) * dss
            if j == 0:
                value_v = drift_v[0] * (U[i, 1] - U[i, 0]) / dv
            elif j == nv - 1:
                value_v = drift_v[-1] * (U[i, -1] - U[i, -2]) / dv
            else:
                first = (U[i, j + 1] - U[i, j - 1]) / (2. * dv)
                second = (U[i, j + 1] - 2. * U[i, j] + U[i, j - 1]) / dv2
                value_v = drift_v[j] * first + diffusion_v[j] * second
            lv[row, j] = value_v
            mixed = 0.
            if 0 < j < nv - 1:
                cross = (U[i + 1, j + 1] - U[i + 1, j - 1]
                         - U[i - 1, j + 1] + U[i - 1, j - 1]) / (4. * dS * dv)
                mixed = rho * sigma_v * (S[i] * v[j]) * cross
            elif j == nv - 1:
                ip = 2. * U[i + 1, -1] - U[i + 1, -2]
                im = 2. * U[i - 1, -1] - U[i - 1, -2]
                cross = (ip - U[i + 1, -2] - im + U[i - 1, -2]) / (4. * dS * dv)
                mixed = rho * sigma_v * (S[i] * v[-1]) * cross
            y0 = U[i, j] + dt * (ls + value_v + mixed - r * U[i, j])
            rhs[row, j] = y0 - half * ls

    # Forward/back substitution keeps the original divisions and ordering.
    for row in range(n):
        for j in range(nv):
            value = rhs[row, j]
            if row == 0:
                value -= lower_s[row, j] * lower_boundary
            else:
                value -= lower_s[row, j] * rhs[row - 1, j]
            rhs[row, j] = value / denominator_s[row, j]
    for row in range(n - 2, -1, -1):
        for j in range(nv):
            rhs[row, j] = rhs[row, j] - cp_s[row, j] * rhs[row + 1, j]
    for row in range(n):
        i = row + 1
        result[i, 0] = (rhs[row, 0] - half * lv[row, 0]) / denominator_v[0]
        for j in range(1, nv):
            value = rhs[row, j] - half * lv[row, j]
            result[i, j] = (value - lower_v[j] * result[i, j - 1]) / denominator_v[j]
        for j in range(nv - 2, -1, -1):
            result[i, j] = result[i, j] - cp_v[j] * result[i, j + 1]
    # Boundary variance solves are independent and overwritten in the original.
    result[0, :] = lower_boundary
    result[-1, :] = 2. * result[-2, :] - result[-3, :]
    return result


@njit(cache=True)
def heston_adi_step(U, dt, S, dS, v, dv, r, q, kappa, theta, sigma_v, rho,
                    lower_boundary=0.0, implicit_weight=0.5):
    """Compatible standalone step; leaves input untouched and returns new storage.

    This numerical interface takes scalar q. Product pricers additionally accept
    cumulative-carry curves and supply a scalar average rate for each substep.
    """
    # Preserve the original intermediate casts for non-float64 standalone inputs.
    if not (U.dtype == np.dtype(np.float64) and S.dtype == np.dtype(np.float64)
            and v.dtype == np.dtype(np.float64)):
        return _adi_step_legacy(U, dt, S, dS, v, dv, r, q, kappa, theta,
                                sigma_v, rho, lower_boundary, implicit_weight)
    factors = _prepare_adi(dt, S, dS, v, dv, r, q, kappa, theta, sigma_v, implicit_weight)
    lv = np.empty((len(S) - 2, len(v)))
    rhs = np.empty_like(lv)
    result = np.empty(U.shape)
    return _adi_step_prepared(U, dt, S, dS, v, dv, r, q, kappa, theta, sigma_v,
                              rho, lower_boundary, factors, lv, rhs, result, implicit_weight)


@njit(cache=True)
def _adi_step_legacy(
    U,
    dt,
    S,
    dS,
    v,
    dv,
    r,
    q,
    kappa,
    theta,
    sigma_v,
    rho,
    lower_boundary=0.0,
    implicit_weight=0.5,
):
    """Retain original dtype-specific intermediates for standalone compatibility."""
    Ns, Nv = U.shape
    w = float(implicit_weight)

    S2 = S * S
    a_convS = (r - q) * S
    b_conv_v = kappa * (theta - v)
    diff_v = 0.5 * sigma_v * sigma_v * v

    # ------ Explicit step: Y0 = U + dt*(L_S + L_v + L_C - r)U ------
    L_S_U = np.zeros_like(U)
    # upwind sign is constant on the interior (S > 0): pick the stencil once
    if a_convS[1] >= 0.0:  # forward difference
        dU_dS = (U[2:, :] - U[1:-1, :]) / dS
    else:  # backward difference
        dU_dS = (U[1:-1, :] - U[:-2, :]) / dS
    d2U_dS2 = (U[2:, :] - 2.0 * U[1:-1, :] + U[:-2, :]) / (dS * dS)
    L_S_U[1:-1, :] = (
        a_convS[1:-1, None] * dU_dS + 0.5 * (v[None, :] * S2[1:-1, None]) * d2U_dS2
    )

    L_v_U = np.zeros_like(U)
    L_v_U[:, 0] = b_conv_v[0] * (U[:, 1] - U[:, 0]) / dv
    L_v_U[:, -1] = b_conv_v[-1] * (U[:, -1] - U[:, -2]) / dv
    dU_dv = (U[:, 2:] - U[:, :-2]) / (2.0 * dv)
    d2U_dv2 = (U[:, 2:] - 2.0 * U[:, 1:-1] + U[:, :-2]) / (dv * dv)
    L_v_U[:, 1:-1] = b_conv_v[1:-1][None, :] * dU_dv + diff_v[1:-1][None, :] * d2U_dv2

    L_C_U = np.zeros_like(U)
    d2U_dSdv = (U[2:, 2:] - U[2:, :-2] - U[:-2, 2:] + U[:-2, :-2]) / (4.0 * dS * dv)
    L_C_U[1:-1, 1:-1] = rho * sigma_v * (S[1:-1, None] * v[1:-1][None, :]) * d2U_dSdv
    # top v boundary (j = Nv-1): linear extrapolation of U in v
    U_ip1_Nv = 2.0 * U[2:, -1] - U[2:, -2]
    U_im1_Nv = 2.0 * U[:-2, -1] - U[:-2, -2]
    d2U_dSdv_top = (U_ip1_Nv - U[2:, -2] - U_im1_Nv + U[:-2, -2]) / (4.0 * dS * dv)
    L_C_U[1:-1, -1] = rho * sigma_v * (S[1:-1] * v[-1]) * d2U_dSdv_top

    Y0 = U + dt * (L_S_U + L_v_U + L_C_U - r * U)

    # ------ S-implicit sweep (all v rows solved at once) ------
    Nint = Ns - 2
    if a_convS[1] >= 0.0:
        Lm_c = np.zeros(Nint)
        Li_c = -a_convS[1:-1] / dS
        Lp_c = a_convS[1:-1] / dS
    else:
        Lm_c = -a_convS[1:-1] / dS
        Li_c = a_convS[1:-1] / dS
        Lp_c = np.zeros(Nint)
    D = 0.5 * v[None, :] * S2[1:-1, None] / (dS * dS)  # (Nint, Nv)
    a_S = -w * dt * (Lm_c[:, None] + D).T  # (Nv, Nint)
    b_S = 1.0 - w * dt * (Li_c[:, None] - 2.0 * D).T
    c_S = -w * dt * (Lp_c[:, None] + D).T
    rhs = (Y0[1:-1, :] - w * dt * L_S_U[1:-1, :]).T
    # The first interior equation contains a_S[:, 0] * U(S=0).  Move the
    # known Dirichlet value to the right-hand side before solving.
    rhs[:, 0] -= a_S[:, 0] * lower_boundary
    # Eliminate Y[B] = 2*Y[L] - Y[L-1] from the last interior equation.
    a_S[:, -1] -= c_S[:, -1]
    b_S[:, -1] += 2.0 * c_S[:, -1]
    interior = thomas_algorithm_batch(a_S[:, 1:], b_S, c_S[:, :-1], rhs)
    Y1 = np.zeros_like(U)
    Y1[0, :] = lower_boundary
    Y1[1:-1, :] = interior.T
    # zero gamma at S_max: linear extrapolation
    Y1[-1, :] = 2.0 * Y1[-2, :] - Y1[-3, :]

    # ------ v-implicit sweep (all S rows solved at once) ------
    Lm_v = np.empty(Nv)
    Li_v = np.empty(Nv)
    Lp_v = np.empty(Nv)
    Lm_v[0] = 0.0
    Li_v[0] = -b_conv_v[0] / dv
    Lp_v[0] = b_conv_v[0] / dv
    Lm_v[-1] = -b_conv_v[-1] / dv
    Li_v[-1] = b_conv_v[-1] / dv
    Lp_v[-1] = 0.0
    diff2 = 0.5 * sigma_v * sigma_v * v[1:-1] / (dv * dv)
    Lm_v[1:-1] = -b_conv_v[1:-1] / (2.0 * dv) + diff2
    Li_v[1:-1] = -2.0 * diff2
    Lp_v[1:-1] = b_conv_v[1:-1] / (2.0 * dv) + diff2

    a_v = np.broadcast_to(-w * dt * Lm_v[None, :], (Ns, Nv))
    b_v = np.broadcast_to(1.0 - w * dt * Li_v[None, :], (Ns, Nv))
    c_v = np.broadcast_to(-w * dt * Lp_v[None, :], (Ns, Nv))
    rhs_v = Y1 - w * dt * L_v_U
    Y2 = thomas_algorithm_batch(a_v[:, 1:], b_v, c_v[:, :-1], rhs_v)

    # enforce S boundaries again
    Y2[0, :] = lower_boundary
    Y2[-1, :] = 2.0 * Y2[-2, :] - Y2[-3, :]
    return Y2


class _ADIRollback:
    """Own factors and scratch for one valuation; rotate only retired states.

    Exact matrix keys retain at most four factor sets. Every interval computes
    its rate once for both states, using the root grid endpoints unchanged.
    """

    def __init__(self, S, dS, v, dv, r, q_input, T, kappa, theta, sigma_v, rho):
        self.grid = (S, dS, v, dv, r)
        self.model = (kappa, theta, sigma_v, rho)
        self.q_input, self.T = q_input, T
        self.factor_cache = {}
        self.factor_preparations = self.cache_hits = 0
        self.lv = np.empty((len(S) - 2, len(v)))
        self.rhs = np.empty_like(self.lv)
        self.output = np.empty((len(S), len(v)))

    def _factors(self, h, weight, q_step):
        """Reuse exact implicit matrices without rounding interval yields."""
        key = (weight * h, q_step)
        if key in self.factor_cache:
            factors = self.factor_cache.pop(key)
            self.cache_hits += 1
        else:
            if len(self.factor_cache) == 4:
                self.factor_cache.pop(next(iter(self.factor_cache)))
            factors = _prepare_adi(h, *self.grid, q_step, *self.model[:3], weight)
            self.factor_preparations += 1
        self.factor_cache[key] = factors
        return factors

    def advance(self, states, dt, tau_start, tau_end, lower_terminal,
                *, weight=0.5, substeps=1):
        """Advance privately owned states with the original subinterval arithmetic.

        A returned state takes the spare buffer; its retired input becomes the
        next spare. All input stencils finish before any output is written.
        """
        h = dt / substeps
        result = list(states)
        for i in range(substeps):
            start = tau_start + i * h
            end = tau_end if i == substeps - 1 else start + h
            q_step = dividend_rate_for_tau_step(self.q_input, self.T, start, end)
            boundary = lower_terminal * np.exp(-self.grid[-1] * end)
            factors = self._factors(h, weight, q_step)
            for j, U in enumerate(result):
                result[j] = _adi_step_prepared(
                    U, h, *self.grid, q_step, *self.model, boundary,
                    factors, self.lv, self.rhs, self.output, weight)
                self.output = U
        return result


def _integer_trading_days(values, label):
    """Canonicalize integer-aligned day offsets with absolute tolerance 1e-8."""
    values = np.asarray(values, dtype=float)
    rounded = np.rint(values)
    if (np.any(~np.isfinite(values)) or np.any(values < 0)
            or np.any(np.abs(values - rounded) > 1e-8)
            or np.any(rounded >= np.iinfo(np.int64).max)):
        raise ValueError(f"{label} must be finite, nonnegative whole trading days (tolerance 1e-8 days)")
    return rounded.astype(np.int64)


def _trading_schedule(T, obs_times, steps_per_day, *, obs_in_years=False):
    """One regular grid; stable first-row duplicates retain aligned metadata.

    Days beyond maturity are filtered after integer canonicalization. Markers
    include maturity at zero and valuation at N*m; no dates add extra nodes.
    """
    if (isinstance(steps_per_day, (bool, np.bool_))
            or not isinstance(steps_per_day, (int, np.integer)) or steps_per_day <= 0):
        raise ValueError("steps_per_day must be a positive integer")
    N = int(_integer_trading_days(float(T) * 252, "T in years (252*T trading days)"))
    observations = np.atleast_1d(np.asarray(obs_times, dtype=float))
    if observations.ndim != 1:
        raise ValueError("obs_times must be a one-dimensional observation schedule")
    units = "obs_times in years (252*obs_times trading days)" if obs_in_years else "obs_times in trading days"
    days = _integer_trading_days(observations * (252 if obs_in_years else 1), units)
    order = np.argsort(days, kind="stable")
    order = order[days[order] <= N]
    if order.size == 0:
        raise ValueError("obs_times has no supported observations within maturity")
    order = order[np.r_[True, np.diff(days[order]) != 0]]
    days = days[order]
    events = {(N - int(day)) * int(steps_per_day): i for i, day in enumerate(days)}
    times = np.arange(N * int(steps_per_day) + 1, dtype=np.int64) / (252 * steps_per_day)
    return N, days, order, events, times


def _event_values(values, count, label):
    """Broadcast a scalar or retain one metadata value per original event row."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        return np.full(count, float(values))
    if values.ndim != 1 or values.size != count:
        raise ValueError(f"{label} must have one value per obs_times row")
    return values


def _pricing_grid(S0, v0, theta, knock_out, Ns, Nv, S_max, v_max):
    """Build automatic or explicit domains containing the evaluation point."""
    S_max = max(3.0 * S0, 2.0 * float(np.max(knock_out))) if S_max is None else float(S_max)
    v_max = max(0.5, 4 * theta, 4 * v0) if v_max is None else float(v_max)
    if not (np.isfinite(S_max) and np.isfinite(v_max) and S_max > 0 and v_max > 0
            and 0 <= S0 <= S_max and 0 <= v0 <= v_max):
        raise ValueError("Domains must be positive, finite and contain S0 and v0")
    S = np.linspace(0, S_max, Ns)
    v = np.linspace(0, v_max, Nv)
    return S, S[1] - S[0], v, v[1] - v[0]


def _initial_value(U, S, v, S0, v0):
    """Interpolate one scalar price, rejecting points outside the physical grid."""
    interp = RegularGridInterpolator((S, v), U, bounds_error=True, fill_value=None)
    return float(interp((S0, v0)))


def price_snowball_heston(
    S0,
    T,
    coupon,
    knock_out,
    H_in,
    obs_times,
    r,
    q,
    kappa,
    theta,
    sigma_v,
    rho,
    v0,
    Ns=80,
    Nv=40,
    *,
    daily_ki=True,
    historical_KI=False,
    SPD=0.0,
    pay_times=None,
    mg=0.0,
    mg_ist=None,
    K=None,
    q_maturities=None,
    steps_per_day=1,
    S_max=None,
    v_max=None,
    return_surface=False,
    rannacher_steps=2,
):
    """Snowball scalar net payoff price.

    T is years; obs_times are whole trading days. Explicit pay_times supply
    inception-relative accrual-day counts for the caller's 30/360 convention.
    Without pay_times, obs_days/252 is a compatibility fallback, not a
    date-based 30/360 calculation. Accrual ends at the last observation.
    steps_per_day subdivides the 252-day PDE grid without
    adding observations. Daily KI includes pre-observation valuation.
    K defaults to S0; q accepts a scalar, curve, or array with q_maturities.
    historical_KI selects the already-knocked-in state at valuation.
    rannacher_steps backward-Euler half-steps are taken after the terminal
    payoff and after every observed event (0 restores plain Crank-Nicolson).
    return_surface returns (S, v, U) of the priced state instead of the scalar.
    Options after Ns/Nv are keyword-only. Explicit domains reject extrapolation."""
    rannacher_steps = int(rannacher_steps)
    if rannacher_steps < 0:
        raise ValueError("rannacher_steps must be a non-negative integer")
    q_input = prepare_dividend_input(q, q_maturities)
    count = np.size(obs_times)
    N, obs_days, order, events, tau_points = _trading_schedule(
        T, obs_times, steps_per_day, obs_in_years=False)
    T = N / 252.0
    dt = 1.0 / (252 * steps_per_day)
    coupon = _event_values(coupon, count, "coupon")[order]
    knock_out = _event_values(knock_out, count, "knock_out")[order]
    pay_yr = (obs_days / 252.0 if pay_times is None else
              _event_values(pay_times, count, "pay_times")[order] / 360.0)
    if mg_ist is None:
        mg_ist = r
    mgi = mg * mg_ist
    if K is None:
        K = S0

    period_len = np.diff(np.concatenate(([0.0], pay_yr)))
    acc_coupon = np.cumsum(coupon * period_len)
    full_coupon = acc_coupon[-1]
    # Contractual accrual ends at the final observation, independently of PDE T.
    full_pay = pay_yr[-1]

    S, dS, v, dv = _pricing_grid(S0, v0, theta, knock_out, Ns, Nv, S_max, v_max)

    payoff_in = np.where(S >= K, 1.0, np.maximum(S / K, SPD))
    U_in = np.tile((payoff_in - 1.0)[:, None], (1, Nv)) - mgi * full_pay
    if historical_KI:
        U_noin = None
    elif daily_ki:
        U_noin = np.full((Ns, Nv), full_coupon - mgi * full_pay)
    else:
        payoff_noin = np.where(S <= H_in, payoff_in - 1.0, full_coupon) - mgi * full_pay
        U_noin = np.tile(payoff_noin[:, None], (1, Nv))

    terminal_idx = events.get(0)
    if terminal_idx is not None:
        # Maturity KO uses the aligned contractual redemption in both states.
        redemption = acc_coupon[terminal_idx] - mgi * pay_yr[terminal_idx]
        mask_out = S >= knock_out[terminal_idx]
        if U_noin is not None:
            U_noin[mask_out, :] = redemption
        U_in[mask_out, :] = redemption

    if daily_ki and not historical_KI:
        # Maturity itself is a knock-in observation, including at T=0.
        U_noin[S <= H_in, :] = U_in[S <= H_in, :]

    # At S=0 the underlying cannot recover or knock out.  Both state values
    # therefore equal the knocked-in terminal loss, discounted while stepping
    # backward.  Initialize the no-in boundary as well for the first PDE step.
    lower_terminal = float(U_in[0, 0])
    if U_noin is not None:
        U_noin[0, :] = lower_terminal

    rollback = _ADIRollback(S, dS, v, dv, r, q_input, T, kappa, theta, sigma_v, rho)
    for step in range(1, len(tau_points)):
        # Terminal payoff (first step) and every observed event inject grid-scale
        # content; follow them with Rannacher backward-Euler half-steps.
        rannacher = rannacher_steps > 0 and (step == 1 or (step - 1) in events)
        active = [] if historical_KI else [U_noin]
        if daily_ki or historical_KI:
            active.append(U_in)
        active = rollback.advance(
            active, dt, tau_points[step - 1], tau_points[step], lower_terminal,
            weight=1.0 if rannacher else 0.5,
            substeps=rannacher_steps if rannacher else 1,
        )
        if not historical_KI:
            U_noin = active[0]
        if daily_ki or historical_KI:
            U_in = active[-1]
            if daily_ki and not historical_KI and step % steps_per_day == 0:
                # Numerical substeps do not add contractual daily KI observations.
                mask_in = S <= H_in
                U_noin[mask_in, :] = U_in[mask_in, :]
        idx = events.get(step)
        if idx is not None:
            redemption = acc_coupon[idx] - mgi * pay_yr[idx]
            mask_out = S >= knock_out[idx]
            if U_noin is not None:
                U_noin[mask_out, :] = redemption
            if daily_ki or historical_KI:
                U_in[mask_out, :] = redemption

    value = U_in if historical_KI else U_noin
    if return_surface:
        return S, v, value
    return _initial_value(value, S, v, S0, v0)


def price_phoenix_heston(
    S0,
    T,
    coupon,
    knock_out,
    H_in,
    obs_times,
    r,
    q,
    kappa,
    theta,
    sigma_v,
    rho,
    v0,
    Ns=80,
    Nv=40,
    *,
    H_coupon=None,
    daily_ki=True,
    historical_KI=False,
    SPD=0.0,
    mg=0.0,
    mg_ist=None,
    full_term=None,
    K=None,
    q_maturities=None,
    steps_per_day=1,
    S_max=None,
    v_max=None,
    return_surface=False,
    rannacher_steps=2,
):
    """Phoenix scalar net payoff price with absolute observation coupons.

    T and obs_times are years aligned to whole 252-trading-day offsets.
    steps_per_day adds PDE substeps without extra observations; daily KI
    includes pre-observation valuation. full_term is the margin horizon from
    issue; K is the issue strike, defaulting to S0 for a fresh trade.
    historical_KI selects the already-knocked-in state at valuation.
    rannacher_steps backward-Euler half-steps are taken after the terminal
    payoff and after every observed event (0 restores plain Crank-Nicolson).
    return_surface returns (S, v, U) of the priced state instead of the scalar.
    q accepts a scalar, curve, or array with q_maturities. Options after Ns/Nv
    are keyword-only. Explicit domains reject extrapolation."""
    rannacher_steps = int(rannacher_steps)
    if rannacher_steps < 0:
        raise ValueError("rannacher_steps must be a non-negative integer")
    q_input = prepare_dividend_input(q, q_maturities)
    count = np.size(obs_times)
    N, obs_days, order, events, tau_points = _trading_schedule(
        T, obs_times, steps_per_day, obs_in_years=True)
    T = N / 252.0
    dt = 1.0 / (252 * steps_per_day)
    coupon = _event_values(coupon, count, "coupon")[order]
    knock_out = _event_values(knock_out, count, "knock_out")[order]
    obs_times = obs_days / 252.0
    if mg_ist is None:
        mg_ist = r
    mgi = mg * mg_ist
    if K is None:
        K = S0

    if H_coupon is None:
        H_coupon = H_in

    # maturity margin over the full horizon (from issue); pass full_term (the trade's
    # total term from issue, e.g. 2.0) for a seasoned trade, else falls back to T
    if full_term is None:
        full_term = obs_times[-1]
    full_pay = max(full_term, T)

    S, dS, v, dv = _pricing_grid(S0, v0, theta, knock_out, Ns, Nv, S_max, v_max)

    payoff_in = np.where(S >= K, 1.0, np.maximum(S / K, SPD))
    U_in = np.tile((payoff_in - 1.0)[:, None], (1, Nv)) - mgi * full_pay
    if historical_KI:
        U_noin = None
    elif daily_ki:
        U_noin = np.full((Ns, Nv), -mgi * full_pay)
    else:
        payoff_noin = np.where(S <= H_in, payoff_in - 1.0, 0.0) - mgi * full_pay
        U_noin = np.tile(payoff_noin[:, None], (1, Nv))

    terminal_idx = events.get(0)
    if terminal_idx is not None:
        coupon_last = float(coupon[terminal_idx])
        mask_c = S >= H_coupon
        if U_noin is not None:
            U_noin[mask_c, :] += coupon_last
        U_in[mask_c, :] += coupon_last
        mask_out = S >= knock_out[terminal_idx]
        if U_noin is not None:
            U_noin[mask_out, :] = coupon_last - mgi * full_pay
        U_in[mask_out, :] = coupon_last - mgi * full_pay

    if daily_ki and not historical_KI:
        # Observe terminal knock-in after maturity coupon/knock-out events.
        U_noin[S <= H_in, :] = U_in[S <= H_in, :]

    # At S=0 the underlying cannot reach either the coupon or knock-out
    # barrier.  Both state values reduce to the knocked-in terminal loss.
    lower_terminal = float(U_in[0, 0])
    if U_noin is not None:
        U_noin[0, :] = lower_terminal

    rollback = _ADIRollback(S, dS, v, dv, r, q_input, T, kappa, theta, sigma_v, rho)
    for step in range(1, len(tau_points)):
        # Terminal payoff (first step) and every observed event inject grid-scale
        # content; follow them with Rannacher backward-Euler half-steps.
        rannacher = rannacher_steps > 0 and (step == 1 or (step - 1) in events)
        active = [] if historical_KI else [U_noin]
        if daily_ki or historical_KI:
            active.append(U_in)
        active = rollback.advance(
            active, dt, tau_points[step - 1], tau_points[step], lower_terminal,
            weight=1.0 if rannacher else 0.5,
            substeps=rannacher_steps if rannacher else 1,
        )
        if not historical_KI:
            U_noin = active[0]
        if daily_ki or historical_KI:
            U_in = active[-1]
            if daily_ki and not historical_KI and step % steps_per_day == 0:
                # Numerical substeps do not add contractual daily KI observations.
                mask_in = S <= H_in
                U_noin[mask_in, :] = U_in[mask_in, :]
        idx = events.get(step)
        if idx is not None:
            mask_c = S >= H_coupon
            if U_noin is not None:
                U_noin[mask_c, :] += coupon[idx]
            U_in[mask_c, :] += coupon[idx]
            mask_out = S >= knock_out[idx]
            if U_noin is not None:
                U_noin[mask_out, :] = coupon[idx] - mgi * obs_times[idx]
            U_in[mask_out, :] = coupon[idx] - mgi * obs_times[idx]

    value = U_in if historical_KI else U_noin
    if return_surface:
        return S, v, value
    return _initial_value(value, S, v, S0, v0)


def price_autocallable_heston(
    S0,
    T,
    coupon,
    knock_out,
    obs_times,
    premium,
    r,
    q,
    kappa,
    theta,
    sigma_v,
    rho,
    v0,
    Ns=80,
    Nv=40,
    *,
    pay_times=None,
    mg=0.0,
    mg_ist=None,
    q_maturities=None,
    steps_per_day=1,
    S_max=None,
    v_max=None,
    return_surface=False,
    rannacher_steps=2,
):
    """Autocallable scalar net payoff price.

    T is years; obs_times are whole trading days. Explicit pay_times supply
    inception-relative accrual-day counts for the caller's 30/360 convention.
    Without pay_times, obs_days/252 is a compatibility fallback, not a
    date-based 30/360 calculation. Accrual ends at the last observation.
    steps_per_day adds PDE substeps without extra events.
    rannacher_steps backward-Euler half-steps are taken after the terminal
    payoff and after every observed event (0 restores plain Crank-Nicolson).
    return_surface returns (S, v, U) instead of the scalar.
    q accepts a scalar, curve, or array with q_maturities. Options after Ns/Nv
    are keyword-only. Explicit domains reject extrapolation."""
    rannacher_steps = int(rannacher_steps)
    if rannacher_steps < 0:
        raise ValueError("rannacher_steps must be a non-negative integer")
    q_input = prepare_dividend_input(q, q_maturities)
    count = np.size(obs_times)
    N, obs_days, order, events, tau_points = _trading_schedule(
        T, obs_times, steps_per_day, obs_in_years=False)
    T = N / 252.0
    dt = 1.0 / (252 * steps_per_day)
    coupon = _event_values(coupon, count, "coupon")[order]
    knock_out = _event_values(knock_out, count, "knock_out")[order]
    pay_yr = (obs_days / 252.0 if pay_times is None else
              _event_values(pay_times, count, "pay_times")[order] / 360.0)
    if mg_ist is None:
        mg_ist = r
    mgi = mg * mg_ist

    period_len = np.diff(np.concatenate(([0.0], pay_yr)))
    acc_coupon = np.cumsum(coupon * period_len)
    full_coupon = acc_coupon[-1]
    # Premium and margin share the final observation's contractual horizon.
    full_pay = pay_yr[-1]

    S, dS, v, dv = _pricing_grid(S0, v0, theta, knock_out, Ns, Nv, S_max, v_max)

    U = np.full((Ns, Nv), -premium * full_pay - mgi * full_pay)
    terminal_idx = events.get(0)
    if terminal_idx is not None:
        mask_out = S >= knock_out[terminal_idx]
        U[mask_out, :] = full_coupon - premium * full_pay - mgi * full_pay

    lower_terminal = float(U[0, 0])

    rollback = _ADIRollback(S, dS, v, dv, r, q_input, T, kappa, theta, sigma_v, rho)
    for step in range(1, len(tau_points)):
        # Terminal payoff (first step) and every observed event inject grid-scale
        # content; follow them with Rannacher backward-Euler half-steps.
        rannacher = rannacher_steps > 0 and (step == 1 or (step - 1) in events)
        (U,) = rollback.advance(
            [U], dt, tau_points[step - 1], tau_points[step], lower_terminal,
            weight=1.0 if rannacher else 0.5,
            substeps=rannacher_steps if rannacher else 1,
        )
        idx = events.get(step)
        if idx is not None:
            redemption = acc_coupon[idx] - premium * pay_yr[idx] - mgi * pay_yr[idx]
            mask_out = S >= knock_out[idx]
            U[mask_out, :] = redemption

    if return_surface:
        return S, v, U
    return _initial_value(U, S, v, S0, v0)


def _stencil_bounds(S, S0):
    """Interior node and cell weight used at S0, plus the spot stencil bounds."""
    dS = float(S[1] - S[0])
    i = int(np.clip(np.searchsorted(S, S0, side="right") - 1, 1, len(S) - 3))
    weight = (S0 - S[i]) / dS
    lo, hi = i - 1, i + 2
    if abs(weight) < 1e-12:
        hi = i + 1
    elif abs(weight - 1.0) < 1e-12:
        lo = i
    return i, float(weight), lo, hi


def node_greeks(S, v, U, S0, v0):
    """Delta, gamma and variance vega from central node differences.

    Derivatives are taken at the four nodes of the cell containing (S0, v0) and
    averaged with that cell's bilinear weights, the same definition as the
    production grid_greeks. dV/dv0 is unscaled; multiply by 0.02*sqrt(v0) for a
    per-volatility-point vega.
    """
    dS, dv = float(S[1] - S[0]), float(v[1] - v[0])
    if not (S[1] <= S0 < S[-3] and v[1] <= v0 < v[-3]):
        raise ValueError("evaluation point must lie inside the differentiable grid interior")
    i = int(np.clip(np.searchsorted(S, S0, side="right") - 1, 1, len(S) - 3))
    j = int(np.clip(np.searchsorted(v, v0, side="right") - 1, 1, len(v) - 3))
    wS, wv = (S0 - S[i]) / dS, (v0 - v[j]) / dv
    delta = gamma = variance_vega = 0.0
    for ii, wi in ((i, 1.0 - wS), (i + 1, wS)):
        for jj, wj in ((j, 1.0 - wv), (j + 1, wv)):
            w = wi * wj
            delta += w * (U[ii + 1, jj] - U[ii - 1, jj]) / (2.0 * dS)
            gamma += w * (U[ii + 1, jj] - 2.0 * U[ii, jj] + U[ii - 1, jj]) / (dS * dS)
            variance_vega += w * (U[ii, jj + 1] - U[ii, jj - 1]) / (2.0 * dv)
    return float(delta), float(gamma), float(variance_vega)


def calculate_greeks(pricing_function, parameters):
    """Node Greeks from one PDE solve; no bump and no moving domain.

    parameters must supply S0/v0 and every pricing input; it is not modified.
    delta = dV/dS, gamma = d2V/dS2, vega = dV/dv0 (variance vega) and
    vol_vega = vega*0.02*sqrt(v0) per volatility point, all per unit notional.
    stencil_crosses_barrier flags an evaluation stencil that straddles the
    knock-out or knock-in level, where gamma is a discrete event derivative
    rather than a smooth surface derivative.
    """
    if not callable(pricing_function):
        raise TypeError("pricing_function must be callable")
    base = dict(parameters)
    for name in ("S0", "v0"):
        if name not in base:
            raise KeyError(f"parameters is missing: {name}")
        if float(base[name]) <= 0.0:
            raise ValueError(f"{name} must be positive")

    spot, variance = float(base["S0"]), float(base["v0"])
    S, v, U = pricing_function(**base, return_surface=True)
    delta, gamma, variance_vega = node_greeks(S, v, U, spot, variance)
    _, _, lo, hi = _stencil_bounds(S, spot)
    barrier_values = [np.asarray(base[name], dtype=float).ravel()
                      for name in ("knock_out", "H_in") if name in base]
    crosses = any(np.any((level >= S[lo]) & (level <= S[hi])) for level in barrier_values)
    return {
        "delta": delta,
        "gamma": gamma,
        "vega": variance_vega,
        "vol_vega": variance_vega * 0.02 * np.sqrt(variance),
        "stencil_crosses_barrier": bool(crosses),
    }
