"""Heston calibration, vanilla pricing, and calibration diagnostics.

This module has no market-specific configuration or top-level execution.
Dividend inputs may be constant yields or :class:`DividendCarryCurve` objects.
Calibrations may optionally use prior-day parameters as a normalized historical
regularizer and as a differential-evolution warm start.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize
from scipy.special import log_ndtr, roots_legendre

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
    "converged_prices",
    "evaluate_surface",
    "DEFAULT_HISTORY_TOLERANCES",
    "DividendCarryCurve",
    "bs_call_price",
    "bs_implied_vol",
    "bs_implied_vol_vec",
    "calibrate_heston_gatheral",
    "compute_iv_error",
    "cumulative_q_at",
    "effective_q_at",
    "heston_call_price",
    "heston_call_prices",
    "heston_char_func",
    "heston_char_funcs",
    "heston_forward_cf",
    "heston_put_price",
    "historical_parameter_distance",
    "historical_parameter_report",
    "load_vol_surface",
    "prepare_market",
    "run_no_arbitrage_checks",
]


HESTON_PARAM_NAMES = ("v0", "kappa", "theta", "sigma", "rho")

# Tolerances are measured after transformation. log1p(x) corresponds roughly
# to an x relative move for a positive parameter; rho is measured in Fisher-z
# (atanh) space. v0 is deliberately allowed to react more quickly.
DEFAULT_HISTORY_TOLERANCES = {
    "v0": np.log1p(0.30),
    "kappa": np.log1p(0.25),
    "theta": np.log1p(0.20),
    "sigma": np.log1p(0.20),
    "rho": 0.20,
}

GLOBAL_PARAMETER_BOUNDS = {
    "v0": (0.0001, 0.3),
    "kappa": (1.0, 10.0),
    "theta": (0.0001, 0.12),
    "sigma": (0.01, 2.0),
    "rho": (-0.999, 0.999),
}


@lru_cache(maxsize=32)
def _quad_rule(n, max_u):
    """Composite Gauss-Legendre avoids ill-conditioned very-high-order rules.

    Keep small requested grids as single-panel rules for legacy diagnostics.
    At >=128 nodes, graded origin panels resolve CF curvature and panels of at
    most 32 frequency units preserve tail resolution;
    n specifies the approximate total node budget (rounded up per panel).
    """
    if n >= 128:
        edges = np.unique(np.r_[0, np.array([.25, .5, 1, 2, 4, 8, 16]),
                               np.arange(32, max_u, 32), max_u])
        edges = edges[edges <= max_u]
        panels = len(edges) - 1
        per_panel = max(4, int(np.ceil(n / panels)))
        x, w = roots_legendre(per_panel)
        widths = np.diff(edges) / 2
        centers = (edges[1:] + edges[:-1]) / 2
        return (centers[:, None] + widths[:, None] * x).ravel(), (widths[:, None] * w).ravel()
    x, w = roots_legendre(n)
    return (x + 1) * max_u / 2, w * max_u / 2


# =============================================================================
# Deterministic dividend curve
# =============================================================================
class DividendCarryCurve:
    """
    Dividend curve built from maturity-specific continuously compounded yields.

    At each knot T_i, effective_yields[i] is the zero/effective dividend yield
    over [0, T_i].  Cumulative carry Q_i = T_i * q_i is linearly interpolated.
    Endpoint extrapolation keeps the endpoint effective yield constant:
      Q(T) = q_first * T  before the first knot,
      Q(T) = q_last  * T  after the last knot.
    """

    def __init__(self, maturities, effective_yields):
        maturities = np.asarray(maturities, dtype=float)
        effective_yields = np.asarray(effective_yields, dtype=float)

        if maturities.ndim != 1 or effective_yields.ndim != 1:
            raise ValueError("maturities and effective_yields must be 1D arrays")
        if maturities.size == 0 or maturities.size != effective_yields.size:
            raise ValueError(
                "maturities and effective_yields must have equal non-zero length"
            )
        if not np.all(np.isfinite(maturities)) or not np.all(
            np.isfinite(effective_yields)
        ):
            raise ValueError("dividend-curve inputs must all be finite")
        if np.any(maturities <= 0.0) or np.any(np.diff(maturities) <= 0.0):
            raise ValueError("maturities must be positive and strictly increasing")

        self.maturities = maturities.copy()
        self.effective_yields = effective_yields.copy()
        self.carry_knots = self.maturities * self.effective_yields

    @staticmethod
    def _restore_shape(value, scalar):
        return float(value) if scalar else value

    def cumulative_carry(self, T):
        """Return Q(T) = integral_0^T q_inst(t) dt."""
        scalar = np.ndim(T) == 0
        T_arr = np.asarray(T, dtype=float)
        if np.any(~np.isfinite(T_arr)) or np.any(T_arr < 0.0):
            raise ValueError("T must be finite and non-negative")

        Q = np.interp(T_arr, self.maturities, self.carry_knots)
        Q = np.where(
            T_arr < self.maturities[0],
            self.effective_yields[0] * T_arr,
            Q,
        )
        Q = np.where(
            T_arr > self.maturities[-1],
            self.effective_yields[-1] * T_arr,
            Q,
        )
        return self._restore_shape(Q, scalar)

    def effective_yield(self, T):
        """Return the constant-equivalent yield q_bar(T) = Q(T) / T."""
        scalar = np.ndim(T) == 0
        T_arr = np.asarray(T, dtype=float)
        Q = np.asarray(self.cumulative_carry(T_arr), dtype=float)
        q_eff = np.full_like(T_arr, self.effective_yields[0], dtype=float)
        np.divide(Q, T_arr, out=q_eff, where=T_arr > 0.0)
        return self._restore_shape(q_eff, scalar)

    def discount_factor(self, T):
        """Return the dividend discount factor exp(-Q(T))."""
        return np.exp(-self.cumulative_carry(T))

    def forward(self, S0, r, T):
        """Return F(0,T) = S0 * exp(r*T - Q(T)) for constant r."""
        T_arr = np.asarray(T, dtype=float)
        result = float(S0) * np.exp(float(r) * T_arr - self.cumulative_carry(T_arr))
        return self._restore_shape(result, np.ndim(T) == 0)

    def interval_forward_yields(self):
        """Piecewise-constant instantaneous yields implied between curve knots."""
        if self.maturities.size < 2:
            return np.empty(0, dtype=float)
        return np.diff(self.carry_knots) / np.diff(self.maturities)

    def knot_table(self, S0=None, r=None):
        """Auditable table of curve inputs and quantities used by pricing."""
        table = pd.DataFrame(
            {
                "T": self.maturities,
                "q_effective": self.effective_yields,
                "Q_cumulative": self.carry_knots,
                "dividend_discount": np.exp(-self.carry_knots),
            }
        )
        if S0 is not None and r is not None:
            table["forward"] = self.forward(S0, r, self.maturities)
        return table


def effective_q_at(q_input, T):
    """Resolve a scalar q or a DividendCarryCurve to q_bar(T)."""
    if hasattr(q_input, "effective_yield"):
        return float(q_input.effective_yield(T))
    if callable(q_input):
        return float(q_input(T))
    return float(q_input)


def cumulative_q_at(q_input, T):
    """Resolve cumulative carry Q(T), retaining scalar-q compatibility."""
    if hasattr(q_input, "cumulative_carry"):
        return float(q_input.cumulative_carry(T))
    return float(T) * effective_q_at(q_input, T)


# =============================================================================
# 1. Black-Scholes with Dividends (for IV conversion)
# =============================================================================
@njit(cache=False)
def _otm_value_scalar(a, b, volatility, T):
    """erfc/log-difference equivalent of the vector OTM pricer for optional JIT."""
    if volatility <= 0:
        return 0.0
    root = volatility * math.sqrt(T)
    d1 = math.log(a / b) / root + root / 2
    d2 = d1 - root
    if a <= b:
        first, second = .5 * math.erfc(-d1 / math.sqrt(2)), .5 * math.erfc(-d2 / math.sqrt(2))
        first_scale, second_scale = a, b
    else:
        first, second = .5 * math.erfc(d2 / math.sqrt(2)), .5 * math.erfc(d1 / math.sqrt(2))
        first_scale, second_scale = b, a
    if first == 0:
        return 0.0
    log_first = math.log(first_scale) + math.log(first)
    if second == 0:
        return math.exp(log_first)
    log_second = math.log(second_scale) + math.log(second)
    return math.exp(log_first) * (-math.expm1(min(log_second - log_first, 0.0)))


@njit(cache=False)
def _iv_kernel(price, b, puts, a, T, max_iter, tol):
    """Bracketed Newton, compiled optionally; integer statuses map to public text."""
    iv = np.full(price.size, np.nan)
    codes = np.zeros(price.size, dtype=np.int64)
    for i in range(price.size):
        p, strike = price[i], b[i]
        if not math.isfinite(p):
            continue
        lower = max(strike - a, 0) if puts[i] else max(a - strike, 0)
        upper = strike if puts[i] else a
        slack = 32 * 2.220446049250313e-16 * max(a, strike)
        if p < lower - slack or p > upper + slack:
            codes[i] = 1
            continue
        tv = p - lower
        if tv <= 0:
            iv[i], codes[i] = 0, (2 if tv == 0 else 8)
            continue
        if p >= upper:
            codes[i] = 3
            continue
        if T == 0:
            codes[i] = 4
            continue
        lo, hi = 0.0, 1.0
        for _ in range(12):
            if _otm_value_scalar(a, strike, hi, T) >= tv:
                break
            hi *= 2
        if _otm_value_scalar(a, strike, hi, T) < tv:
            codes[i] = 5
            continue
        vol, codes[i] = min(.3, hi / 2), 6
        for _ in range(max_iter):
            diff = _otm_value_scalar(a, strike, vol, T) - tv
            if abs(diff) <= min(tol, 1e-11 * abs(tv)) or hi - lo <= 1e-11 * max(1, vol):
                iv[i], codes[i] = vol, 7
                break
            if diff > 0:
                hi = vol
            else:
                lo = vol
            d1 = math.log(a / strike) / (vol * math.sqrt(T)) + vol * math.sqrt(T) / 2
            vega = a * math.sqrt(T / (2 * math.pi)) * math.exp(-d1 * d1 / 2)
            proposal = vol - diff / vega if vega > 0 else math.inf
            vol = proposal if math.isfinite(proposal) and lo < proposal < hi else (lo + hi) / 2
    return iv, codes


def _bs_time_value(a, b, volatility, T):
    """Price the OTM leg with log-CDFs to retain low-vega wing precision."""
    root = np.maximum(np.asarray(volatility), 1e-300) * np.sqrt(T)
    log_ratio = np.log(a / b)
    d1 = log_ratio / root + root / 2
    d2 = d1 - root
    call = a <= b
    log_first = np.where(call, np.log(a) + log_ndtr(d1), np.log(b) + log_ndtr(-d2))
    log_second = np.where(call, np.log(b) + log_ndtr(d2), np.log(a) + log_ndtr(-d1))
    value = np.exp(log_first) * (-np.expm1(np.minimum(log_second - log_first, 0)))
    return np.where(np.asarray(volatility) > 0, value, 0.0)


def bs_call_price(S0, K, T, r, q, sigma):
    """Black-Scholes call including the zero-volatility/expiry limits."""
    K = np.asarray(K, dtype=float)
    if T <= 0:
        return np.maximum(S0 - K, 0.0)
    a, b = S0 * np.exp(-cumulative_q_at(q, T)), K * np.exp(-r * T)
    return np.maximum(a - b, 0.0) + _bs_time_value(a, b, sigma, T)


_IV_STATUSES = np.array(["nonfinite_price", "out_of_bounds", "intrinsic", "upper_bound",
                         "expiry_mismatch", "unbracketed", "no_convergence", "ok", "roundoff_intrinsic"])


def bs_implied_vol_vec(price, S0, K, T, r, q, opt_type="call", max_iter=100,
                       tol=1e-8, *, return_status=False):
    """Safeguarded Newton inversion; invalid prices return NaN and a status.

    Bounds allow only 32 machine epsilons of price roundoff. Exactly zero time
    value has IV=0; the upper bound has no finite IV. Wing convergence is measured
    relative to time value as well as in price units.
    """
    price, K, kinds = np.broadcast_arrays(np.atleast_1d(np.asarray(price, float)),
                                         np.asarray(K, float), np.asarray(opt_type))
    if np.any(~np.isin(kinds, ["call", "put"])):
        raise ValueError("opt_type must be call or put")
    if T < 0 or S0 <= 0 or np.any(K <= 0):
        raise ValueError("require T >= 0 and positive S0, K")
    a, b = S0 * np.exp(-cumulative_q_at(q, T)), K * np.exp(-r * T)
    iv, codes = _iv_kernel(price.ravel().copy(), b.ravel().copy(), (kinds == "put").ravel().copy(),
                           float(a), float(T), int(max_iter), float(tol))
    iv, statuses = iv.reshape(price.shape), _IV_STATUSES[codes].reshape(price.shape)
    return (iv, statuses) if return_status else iv


def bs_implied_vol(price, S0, K, T, r, q, opt_type="call", *, return_status=False):
    """Scalar wrapper; optional status preserves the default scalar interface."""
    iv, status = bs_implied_vol_vec(price, S0, K, T, r, q, opt_type, return_status=True)
    return (float(iv[0]), str(status[0])) if return_status else float(iv[0])


# =============================================================================
# 2. Heston Characteristic Functions (Little Trap)
# =============================================================================
@njit(cache=False)
def _variance_cf_nb(u, T, v0, kappa, theta, sigma, rho, j=2):
    """Decaying-exponential Little Heston Trap, with consistent C and D."""
    u = np.asarray(u, dtype=np.complex128)
    iu = 1j * u
    term = (iu if j == 1 else -iu) - u * u
    if sigma <= 1e-7:
        integrated = theta * T + (v0 - theta) * (-np.expm1(-kappa * T)) / kappa
        return np.exp(term * integrated / 2)
    beta = kappa - (rho * sigma if j == 1 else 0) - rho * sigma * iu
    d = np.sqrt(beta * beta - sigma * sigma * term)
    d = np.where(d.real < 0, -d, d)
    # Exclude analytic endpoints before division, including the first moment.
    endpoint = term == 0
    denominator = np.where(endpoint, 1.0, beta + d)
    h = sigma * sigma * term / denominator
    g = h / denominator
    decay = np.exp(-d * T)
    D = h / (sigma * sigma) * (-np.expm1(-d * T)) / (1 - g * decay)
    C = kappa * theta / (sigma * sigma) * (
        h * T - 2 * np.log1p(g * (-np.expm1(-d * T)) / (1 - g)))
    return np.where(endpoint, 1.0 + 0j, np.exp(C + D * v0))





def heston_char_funcs(u, T, S0, r, q, v0, kappa, theta, sigma, rho):
    """Both probability-measure CFs; u=0 returns one without division."""
    u = np.asarray(u, dtype=float)
    phase = np.exp(1j * u * (np.log(S0) + r * T - cumulative_q_at(q, T)))
    return tuple(phase * _variance_cf(u, T, v0, kappa, theta, sigma, rho, j)
                 for j in (1, 2))


def heston_char_func(u, T, S0, r, q, v0, kappa, theta, sigma, rho, j):
    """Single Heston characteristic function (kept for compatibility)."""
    phi1, phi2 = heston_char_funcs(u, T, S0, r, q, v0, kappa, theta, sigma, rho)
    return phi1 if j == 1 else phi2




def heston_call_prices(S0, K, T, r, q, v0, kappa, theta, sigma, rho, n=256, max_u=500):
    """Raw call prices: retain negative/nonfinite integration failures."""
    K = np.asarray(K, float)
    if T <= 0:
        return np.maximum(S0 - K, 0)
    Q = cumulative_q_at(q, T)
    if sigma <= 1e-7:
        integrated = theta * T + (v0 - theta) * (-np.expm1(-kappa * T)) / kappa
        return bs_call_price(S0, K, T, r, q, np.sqrt(integrated / T))
    u, kernel = _strike_kernel(tuple(K.ravel()), float(np.log(S0) + r * T - Q), int(n), float(max_u))
    phi1, phi2 = (_variance_cf(u, T, v0, kappa, theta, sigma, rho, j) for j in (1, 2))
    p1 = 0.5 + np.real(phi1 @ kernel) / np.pi
    p2 = 0.5 + np.real(phi2 @ kernel) / np.pi
    return (S0 * np.exp(-Q) * p1 - K.ravel() * np.exp(-r * T) * p2).reshape(K.shape)


def heston_call_price(S0, K, T, r, q, v0, kappa, theta, sigma, rho, n=256, max_u=500):
    """Scalar price wrapper."""
    return float(heston_call_prices(S0, K, T, r, q, v0, kappa, theta, sigma, rho, n, max_u))


# =============================================================================
# 3. Calibration Setup (vectorized)
# =============================================================================
def prepare_market(market_data, r, q_curve, S0):
    """Prepare every tenor without silently dropping invalid market quotes."""
    if market_data["T"].isna().any():
        raise ValueError("market contains missing maturities")
    groups = []
    for T, sub in market_data.groupby("T", sort=True):
        K, q = sub["K"].to_numpy(float), effective_q_at(q_curve, T)
        iv = (sub["IV"].to_numpy(float) if "IV" in sub else
              bs_implied_vol_vec(sub["price"], S0, K, T, r, q, sub.get("OptionType", "call")))
        if not np.isfinite(T) or T <= 0 or np.any(~np.isfinite(K)) or np.any(~np.isfinite(iv)) or np.any(iv < 0) or np.any(K <= 0):
            raise ValueError("market contains invalid maturities, strikes or target IVs")
        groups.append((float(T), K, iv, q))
    if not groups:
        raise ValueError("market_data is empty")
    return groups


def _coerce_historical_params(historical_params):
    """Validate and return a plain five-parameter history dictionary."""
    if historical_params is None:
        raise ValueError("historical_params is required when use_historical=True")
    try:
        history = {
            name: float(historical_params[name]) for name in HESTON_PARAM_NAMES
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "historical_params must provide v0, kappa, theta, sigma, and rho"
        ) from exc
    if not all(np.isfinite(list(history.values()))):
        raise ValueError("all historical parameters must be finite")
    if any(history[name] <= 0.0 for name in HESTON_PARAM_NAMES[:-1]):
        raise ValueError("historical v0, kappa, theta, and sigma must be positive")
    if not -1.0 < history["rho"] < 1.0:
        raise ValueError("historical rho must be strictly between -1 and 1")
    return history


def _resolve_history_tolerances(history_tolerances=None):
    tolerances = DEFAULT_HISTORY_TOLERANCES.copy()
    if history_tolerances is not None:
        unknown = set(history_tolerances) - set(HESTON_PARAM_NAMES)
        if unknown:
            raise ValueError(f"unknown history tolerance keys: {sorted(unknown)}")
        tolerances.update({k: float(v) for k, v in history_tolerances.items()})
    if any((not np.isfinite(v)) or v <= 0.0 for v in tolerances.values()):
        raise ValueError("all history tolerances must be finite and positive")
    return tolerances


def _history_transform(name, value):
    value = float(value)
    if name == "rho":
        return float(np.arctanh(np.clip(value, -1.0 + 1e-12, 1.0 - 1e-12)))
    if value <= 0.0:
        return np.inf
    return float(np.log(value))


def historical_parameter_distance(
    params,
    historical_params,
    history_tolerances=None,
    active_names=None,
):
    """Return mean squared normalized movement from historical parameters.

    Positive parameters are compared in log space and ``rho`` in Fisher-z
    space. ``history_tolerances`` overrides any subset of
    :data:`DEFAULT_HISTORY_TOLERANCES`. Fixed parameters can be omitted through
    ``active_names``.
    """
    history = _coerce_historical_params(historical_params)
    tolerances = _resolve_history_tolerances(history_tolerances)
    current = dict(zip(HESTON_PARAM_NAMES, np.asarray(params, dtype=float)))
    names = tuple(HESTON_PARAM_NAMES if active_names is None else active_names)
    if not names:
        return 0.0

    standardized = []
    for name in names:
        if name not in HESTON_PARAM_NAMES:
            raise ValueError(f"unknown active Heston parameter: {name}")
        delta = (
            _history_transform(name, current[name])
            - _history_transform(name, history[name])
        ) / tolerances[name]
        standardized.append(delta)
    standardized = np.asarray(standardized, dtype=float)
    if not np.all(np.isfinite(standardized)):
        return 1e10
    return float(np.mean(standardized**2))


def historical_parameter_report(
    current_params,
    historical_params,
    history_tolerances=None,
    active_names=None,
):
    """Return raw and normalized changes from historical parameters."""
    history = _coerce_historical_params(historical_params)
    tolerances = _resolve_history_tolerances(history_tolerances)
    names = tuple(HESTON_PARAM_NAMES if active_names is None else active_names)
    rows = []
    for name in names:
        if name not in HESTON_PARAM_NAMES:
            raise ValueError(f"unknown active Heston parameter: {name}")
        new = float(current_params[name])
        old = history[name]
        transformed_change = _history_transform(name, new) - _history_transform(
            name, old
        )
        rows.append(
            {
                "parameter": name,
                "historical": old,
                "calibrated": new,
                "raw_change": new - old,
                "relative_change": (new / old - 1.0) if name != "rho" else np.nan,
                "normalized_change": transformed_change / tolerances[name],
            }
        )
    return pd.DataFrame(rows)


def _clip_to_bounds(values, bounds):
    """Keep warm starts strictly interior to avoid DE scaling-roundoff rejection."""
    values = np.asarray(values, dtype=float)
    lower = np.asarray([bound[0] for bound in bounds], dtype=float)
    upper = np.asarray([bound[1] for bound in bounds], dtype=float)
    margin = (upper - lower) * 1e-12
    return np.clip(values, lower + margin, upper - margin)


def _validate_history_within_global_bounds(history, active_names):
    for name in active_names:
        lower, upper = GLOBAL_PARAMETER_BOUNDS[name]
        if not lower <= history[name] <= upper:
            raise ValueError(
                f"historical {name}={history[name]} is outside global bounds "
                f"[{lower}, {upper}]"
            )


def compute_iv_error(params, market_data, r, q_curve, S0, penalty_weight=0.0,
                     penalty_sigma=None, historical_params=None, history_penalty_iv=0.0,
                     history_tolerances=None, history_active_names=None, weight_by_T=False,
                     weight_T_floor=0.05, weight_T_power=1.0, iv_cap=0.25, prepared=None,
                     quad_n=256, max_u=500.0, *, tenor_weights=None, validate=True,
                     return_diagnostics=False):
    """True weighted residual MSE plus explicit regularizers; failures reject.

    iv_cap is retained only for call compatibility and never substitutes errors.
    Zero tenor weight excludes that tenor from optimization, not final diagnostics.
    """
    groups = prepared if prepared is not None else prepare_market(market_data, r, q_curve, S0)
    total, total_w, invalid_iv, integration_failures, included = 0.0, 0.0, 0, 0, 0
    for T, K, target, q in groups:
        weight = (T + weight_T_floor) ** (-weight_T_power) if weight_by_T else 1.0
        weight *= (tenor_weights or {}).get(T, 1.0)
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("tenor weights must be finite and non-negative")
        if weight == 0:
            continue
        if validate:
            price, info = converged_prices(S0, K, T, r, q, params, quad_n, max_u,
                                          max_refinements=2)
        else:
            price = heston_call_prices(S0, K, T, r, q, *params, quad_n, max_u)
            info = {"converged": False}
        iv, iv_status = bs_implied_vol_vec(price, S0, K, T, r, q, return_status=True)
        finite_iv = np.isfinite(iv) & np.isin(iv_status, ["ok", "intrinsic"])
        valid = finite_iv & (info["converged"] or not validate)
        invalid_iv += int(np.sum(~finite_iv))
        included += K.size
        if validate and not info["converged"]:
            integration_failures += K.size
        total += weight * float(np.sum((iv[valid] - target[valid]) ** 2))
        total_w += weight * np.sum(valid)
    actual_mse = total / total_w if total_w else np.nan
    loss = actual_mse if invalid_iv == 0 and integration_failures == 0 and total_w else np.inf
    if penalty_weight > 0 and penalty_sigma is not None:
        loss += penalty_weight * (params[3] - penalty_sigma) ** 2
    if historical_params is not None and history_penalty_iv > 0:
        loss += history_penalty_iv ** 2 * historical_parameter_distance(
            params, historical_params, history_tolerances, history_active_names)
    diagnostics = dict(actual_iv_mse=actual_mse, actual_iv_rmse=float(np.sqrt(actual_mse)),
                       invalid_count=invalid_iv, integration_failure_count=integration_failures,
                       included_quotes=included, objective=float(loss),
                       integration_validation_performed=bool(validate),
                       integration_status=("failed" if integration_failures else "passed") if validate else "not_checked",
                       unchecked_quote_count=0 if validate else included)
    return (loss, diagnostics) if return_diagnostics else loss


def calibrate_heston_gatheral(market_data, r, q_curve, S0, fix_kappa=False,
                             kappa_fixed=2.0, stage1_fixed_rho=-0.7, stage2_penalty=1.0,
                             weight_by_T=False, weight_T_power=1.0, de_quad_n=256,
                             polish_quad_n=256, de_maxiter1=500, de_maxiter2=1000,
                             workers=1, max_u=500.0, use_historical=False,
                             historical_params=None, history_penalty_iv=0.002,
                             history_tolerances=None, warm_start_from_history=True,
                             use_historical_rho_in_stage1=True, *, tenor_weights=None,
                             seed=42, popsize=None, stage2_local_bounds=True):
    """Two-stage calibration with identical weights and converged numerics.

    Both search and polish start from max(de_quad_n, polish_quad_n, 128), then
    validate resolution and tail. Exact fixed-kappa bounds replace epsilon ranges.
    Nested objectives require workers=1; iteration budgets and termination are
    recorded rather than claiming differential evolution found a global optimum.
    Default populations remain 20/30 for stages 1/2; popsize overrides both.
    """
    if workers != 1:
        raise ValueError("workers=1 is required for the nested two-stage objectives")
    if not np.isfinite(history_penalty_iv) or history_penalty_iv < 0 or not np.isfinite(stage2_penalty) or stage2_penalty < 0:
        raise ValueError("regularization coefficients must be finite and non-negative")
    n = max(int(de_quad_n), int(polish_quad_n), 128)
    groups = prepare_market(market_data, r, q_curve, S0)
    history = _coerce_historical_params(historical_params) if use_historical else None
    rho1 = history["rho"] if history and use_historical_rho_in_stage1 else stage1_fixed_rho
    bounds1 = [GLOBAL_PARAMETER_BOUNDS[name] for name in HESTON_PARAM_NAMES[:4]]
    if fix_kappa:
        if not GLOBAL_PARAMETER_BOUNDS["kappa"][0] <= kappa_fixed <= GLOBAL_PARAMETER_BOUNDS["kappa"][1]:
            raise ValueError("fixed kappa is outside global bounds")
        bounds1[1] = (kappa_fixed, kappa_fixed)
    active1 = [name for name in HESTON_PARAM_NAMES[:4] if not (fix_kappa and name == "kappa")]
    active2 = [name for name in HESTON_PARAM_NAMES if not (fix_kappa and name == "kappa")]
    if history is not None:
        _validate_history_within_global_bounds(history, active2)
    common = dict(prepared=groups, quad_n=n, max_u=max_u, historical_params=history,
                  history_penalty_iv=history_penalty_iv, history_tolerances=history_tolerances,
                  weight_by_T=weight_by_T, weight_T_power=weight_T_power, tenor_weights=tenor_weights)

    def obj1(p):
        return compute_iv_error([*p, rho1], market_data, r, q_curve, S0,
                                history_active_names=active1, **common)

    def search(objective, bounds, iterations, x0, default_popsize):
        de = differential_evolution(objective, bounds, seed=seed,
                                    popsize=popsize if popsize is not None else default_popsize,
                                    maxiter=iterations, tol=1e-7, atol=1e-10,
                                    polish=False, x0=x0)
        local = minimize(objective, de.x, method="L-BFGS-B", bounds=bounds,
                         options=dict(maxiter=300, ftol=1e-13, gtol=1e-7))
        best = local if np.isfinite(local.fun) and local.fun <= de.fun else de
        if not np.isfinite(best.fun):
            raise RuntimeError("no numerically valid calibration candidate")
        record = dict(de_success=bool(de.success), de_message=str(de.message),
                      de_iterations=int(de.nit), local_success=bool(local.success),
                      local_message=str(local.message), objective=float(best.fun))
        return best.x, record

    x01 = None
    if history and warm_start_from_history:
        x01 = [history[name] for name in HESTON_PARAM_NAMES[:4]]
        if fix_kappa:
            x01[1] = kappa_fixed
        x01 = _clip_to_bounds(x01, bounds1)
    p1, record1 = search(obj1, bounds1, de_maxiter1, x01, 20)
    v, k, t, sigma1 = p1
    bounds2 = [(v * .7, v * 1.3), (k * .5, k * 2), (t * .7, t * 1.1),
               GLOBAL_PARAMETER_BOUNDS["sigma"], GLOBAL_PARAMETER_BOUNDS["rho"]]
    if not stage2_local_bounds:
        bounds2 = [GLOBAL_PARAMETER_BOUNDS[name] for name in HESTON_PARAM_NAMES]
    for i, name in enumerate(HESTON_PARAM_NAMES):
        lower, upper = bounds2[i]
        if history:
            lower, upper = min(lower, history[name]), max(upper, history[name])
        gl, gu = GLOBAL_PARAMETER_BOUNDS[name]
        bounds2[i] = (max(gl, lower), min(gu, upper))
    if fix_kappa:
        bounds2[1] = (kappa_fixed, kappa_fixed)

    def obj2(p):
        return compute_iv_error(p, market_data, r, q_curve, S0,
                                penalty_weight=stage2_penalty, penalty_sigma=sigma1,
                                history_active_names=active2, **common)

    x02 = ([history[name] for name in HESTON_PARAM_NAMES] if history and
           warm_start_from_history else [*p1, rho1])
    p2, record2 = search(obj2, bounds2, de_maxiter2, _clip_to_bounds(x02, bounds2), 30)
    params = dict(zip(HESTON_PARAM_NAMES, map(float, p2)))
    # Higher initial resolution is an independent final numerical verification.
    fitted = evaluate_surface(market_data, params, S0, r, q_curve, n=2 * n, max_u=2 * max_u)
    loss, diagnostics = compute_iv_error(p2, market_data, r, q_curve, S0,
                                         history_active_names=active2, penalty_weight=stage2_penalty,
                                         penalty_sigma=sigma1, return_diagnostics=True, **common)
    if diagnostics["invalid_count"] or diagnostics["integration_failure_count"] or not fitted["integration_converged"].all() or fitted["Fitted_IV"].isna().any():
        raise RuntimeError("final calibration fails integration validation")
    diagnostics.update(stage1=record1, stage2=record2, stage1_params=[*map(float, p1), float(rho1)],
                       sigma_anchor=float(sigma1), sigma_penalty=float(stage2_penalty * (p2[3] - sigma1) ** 2),
                       search_quad_n=n, search_max_u=max_u, seed=seed,
                       unpenalized_iv_rmse=diagnostics["actual_iv_rmse"],
                       unpenalized_iv_mse=diagnostics["actual_iv_mse"],
                       history_enabled=history is not None,
                       history_penalty_iv=float(history_penalty_iv) if history else 0.0,
                       history_distance=historical_parameter_distance(p2, history, history_tolerances, active2) if history else 0.0)
    diagnostics["history_penalty_contribution"] = diagnostics["history_penalty_iv"] ** 2 * diagnostics["history_distance"]
    fitted.attrs["calibration_diagnostics"] = diagnostics
    return params, fitted


def load_vol_surface(file_name):
    """Load IV percentages from the supplied path using the 244-day year.

    Notebook callers select their own data directory; tests explicitly supply
    the restored workbook paths under Further_work.
    """
    mkt_data = pd.read_excel(file_name)
    mkt_data["expire"] /= 244

    market_data = (
        mkt_data.melt(id_vars=["expire"], var_name="K", value_name="IV")
        .rename(columns={"expire": "T"})
        .assign(
            K=lambda df: pd.to_numeric(df["K"], errors="coerce"),
            IV=lambda df: pd.to_numeric(df["IV"], errors="coerce"),
        )
        .dropna(subset=["T", "K", "IV"])
        .sort_values(by=["T", "K"])
        .reset_index(drop=True)
    )

    market_data["IV"] /= 100

    return market_data


# =============================================================================
# NO-ARBITRAGE CHECK HELPERS  (run right after each calibration)
# 1) Forward / martingale : E[S_T] via characteristic function vs S0 e^((r-q)T)
# 2) Put-call parity      : model C - P = S0 e^-qT - K e^-rT
# 3) Butterfly (convexity): call-price slopes non-decreasing in K  (density>=0)
# 4) Calendar             : implied total variance w = sigma^2 * T non-decreasing in T
# Applied to BOTH the quoted market surface (md) and the calibrated Heston model.
# =============================================================================


def heston_put_price(S0, K, T, r, q, v0, kappa, theta, sigma, rho, n=256, max_u=500):
    """Put via parity; preserve raw integration information."""
    call = heston_call_price(S0, K, T, r, q, v0, kappa, theta, sigma, rho, n, max_u)
    return call - S0 * np.exp(-cumulative_q_at(q, T)) + K * np.exp(-r * T)


def heston_forward_cf(S0, T, r, q, v0, kappa, theta, sigma, rho, s=1.0 - 1e-9):
    """E[S_T**s] from the same trap formula, approaching the first moment."""
    value = S0 ** s * np.exp(s * (r * T - cumulative_q_at(q, T)))
    return float(np.real(value * _variance_cf(np.array([-1j * s]), T, v0, kappa, theta, sigma, rho)[0]))


def run_no_arbitrage_checks(md, P, S0, r, q_curve, label, tol_butt=1e-4,
                            tol_cal=1e-4, fwd_ok=1e-6):
    """All-tenor bounds, monotonicity, convexity, forward and parity diagnostics.

    Same-strike total-variance comparisons are descriptive with varying forwards;
    calendar checks require comparing fixed forward moneyness for a formal test.
    Parity uses constructed puts and is an accounting check, not an independent
    proof of integration accuracy.
    """
    fitted = evaluate_surface(md, P, S0, r, q_curve)
    rows = []
    for T, sub in fitted.groupby("T", sort=True):
        sub = sub.sort_values("K")
        K, C, q = sub["K"].to_numpy(), sub["model_price"].to_numpy(), effective_q_at(q_curve, T)
        a, b = S0 * np.exp(-q * T), K * np.exp(-r * T)
        slopes = np.diff(C) / np.diff(K)
        puts = C - a + b
        forward = S0 * np.exp((r - q) * T)
        cf = heston_forward_cf(S0, T, r, q, *(P[name] for name in HESTON_PARAM_NAMES))
        rows.append(dict(T=T, quote_count=len(sub), invalid_iv=int(sub["Fitted_IV"].isna().sum()),
                         integration_failures=int((~sub["integration_converged"].astype(bool)).sum()),
                         bound_violations=int(np.sum((C < np.maximum(a - b, 0) - 1e-7) | (C > a + 1e-7))),
                         monotonic_violations=int(np.sum(slopes > tol_butt)),
                         convexity_violations=int(np.sum(np.diff(slopes) < -tol_butt)),
                         max_parity_error=float(np.max(np.abs(C - puts - (a - b)))),
                         forward_relative_error=float(abs(cf / forward - 1))))
    table = pd.DataFrame(rows)
    print(label + "\n" + table.to_string(index=False))
    table.attrs["same_strike_variance_decreases"] = int(
        fitted.assign(w=fitted["Fitted_IV"] ** 2 * fitted["T"])
        .sort_values(["K", "T"]).groupby("K")["w"].diff().lt(-tol_cal).sum())
    table.attrs["forward_pass"] = bool((table["forward_relative_error"] < fwd_ok).all())
    return table


# Backward-compatible names used by the legacy notebooks.
data_processer = load_vol_surface
no_arb_checks = run_no_arbitrage_checks


@lru_cache(maxsize=64)
def _strike_kernel(strikes, log_forward, n, max_u):
    """Cache strike phases; forward centering reduces cancellation in CFs."""
    u, w = _quad_rule(n, max_u)
    return u, np.exp(1j * u[:, None] * (log_forward - np.log(strikes))) * (w / (1j * u))[:, None]


def converged_prices(S0, K, T, r, q, params, n=256, max_u=500, *,
                     price_tol=2e-7, iv_tol=2e-6, max_refinements=3):
    """Check both node refinement and tail extension before accepting prices.

    Each level doubles nodes at fixed cutoff, then doubles cutoff with doubled
    nodes again. Failed convergence is explicit and calibration rejects it.
    """
    K = np.asarray(K, float)
    coarse = heston_call_prices(S0, K, T, r, q, *params, n, max_u)
    info = {"converged": False, "n": int(n), "max_u": float(max_u),
            "validation_performed": True, "status": "failed"}
    for _ in range(max_refinements):
        if 4 * n > 8192:
            break
        fine = heston_call_prices(S0, K, T, r, q, *params, 2 * n, max_u)
        tail = heston_call_prices(S0, K, T, r, q, *params, 4 * n, 2 * max_u)
        iv0, s0 = bs_implied_vol_vec(coarse, S0, K, T, r, q, return_status=True)
        iv1, s1 = bs_implied_vol_vec(fine, S0, K, T, r, q, return_status=True)
        iv2, s2 = bs_implied_vol_vec(tail, S0, K, T, r, q, return_status=True)
        finite = (np.isfinite(iv0) & np.isfinite(iv1) & np.isfinite(iv2) &
                  np.isin(s0, ["ok", "intrinsic"]) & np.isin(s1, ["ok", "intrinsic"]) & np.isin(s2, ["ok", "intrinsic"]))
        price_gap = max(float(np.max(np.abs(fine - coarse))), float(np.max(np.abs(tail - fine))))
        iv_gap = max(float(np.max(np.abs(iv1 - iv0))), float(np.max(np.abs(iv2 - iv1))))
        info.update(n=int(4 * n), max_u=float(2 * max_u), price_gap=price_gap,
                    iv_gap=iv_gap, invalid_count=int(np.sum(~finite)))
        if finite.all() and price_gap <= price_tol and iv_gap <= iv_tol:
            info["converged"] = True
            info["status"] = "passed"
            return tail, info
        coarse, n, max_u = tail, 4 * n, 2 * max_u
    return coarse, info


def evaluate_surface(market_data, params, S0, r, q_curve, *, n=256, max_u=500,
                     validate=True):
    """Reprice any parameter set, retaining per-quote failure/convergence status."""
    result = market_data.copy()
    result["integration_converged"] = False
    result["integration_validation_performed"] = bool(validate)
    for T, sub in market_data.groupby("T", sort=True):
        K, q = sub["K"].to_numpy(float), effective_q_at(q_curve, T)
        if validate:
            p = [params[name] for name in HESTON_PARAM_NAMES] if isinstance(params, dict) else params
            prices, info = converged_prices(S0, K, T, r, q, p, n, max_u)
        else:
            p = [params[name] for name in HESTON_PARAM_NAMES] if isinstance(params, dict) else params
            prices = heston_call_prices(S0, K, T, r, q, *p, n, max_u)
            info = dict(converged=False, n=n, max_u=max_u, status="not_checked")
        iv, status = bs_implied_vol_vec(prices, S0, K, T, r, q, return_status=True)
        result.loc[sub.index, "model_price"] = prices
        result.loc[sub.index, "Fitted_IV"] = iv
        result.loc[sub.index, "iv_status"] = status
        result.loc[sub.index, "integration_converged"] = info["converged"]
        result.loc[sub.index, "integration_status"] = info["status"]
        result.loc[sub.index, "quad_n"] = info["n"]
        result.loc[sub.index, "max_u"] = info["max_u"]
        result.loc[sub.index, "q_used"] = q
        result.loc[sub.index, "Q_cumulative"] = cumulative_q_at(q_curve, T)
        result.loc[sub.index, "Forward"] = S0 * np.exp(r * T - cumulative_q_at(q_curve, T))
    if "IV" in result:
        result["IV_error"] = result["Fitted_IV"] - result["IV"]
    return result


def _variance_cf(u, T, v0, kappa, theta, sigma, rho, j=2):
    """Normalize scalar/vector inputs for the shared stable Numba CF kernel."""
    u = np.asarray(u, dtype=np.complex128)
    value = _variance_cf_nb(u.ravel(), T, v0, kappa, theta, sigma, rho, j)
    return value.reshape(u.shape)
