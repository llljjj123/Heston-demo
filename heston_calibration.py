"""Heston calibration, vanilla pricing, and calibration diagnostics.

This module has no market-specific configuration or top-level execution.
Dividend inputs may be constant yields or :class:`DividendCarryCurve` objects.
Calibrations may optionally use prior-day parameters as a normalized historical
regularizer and as a differential-evolution warm start.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.integrate import fixed_quad
from scipy.optimize import differential_evolution, minimize
from scipy.stats import norm

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


_QUAD_CACHE = {}
_EPS = 1e-12

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


def _quad_rule(n, max_u):
    """Cached Gauss-Legendre nodes/weights on [1e-12, max_u]."""
    key = (int(n), float(max_u))
    rule = _QUAD_CACHE.get(key)
    if rule is None:
        x, w = np.polynomial.legendre.leggauss(int(n))
        a, b = _EPS, float(max_u)
        u_nodes = 0.5 * (b - a) * x + 0.5 * (b + a)
        weights = 0.5 * (b - a) * w
        rule = (u_nodes, weights)
        _QUAD_CACHE[key] = rule
    return rule


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
def bs_call_price(S0, K, T, r, q, sigma):
    """Black-Scholes call price with continuous dividend q."""
    if T <= 0:
        return max(S0 - K, 0.0)
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_implied_vol(price, S0, K, T, r, q, opt_type="call"):
    """Newton-Raphson implied volatility (scalar, kept for compatibility)."""
    if price <= 0:
        return 0.0
    sigma = 0.3  # initial guess
    for _ in range(100):
        d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if opt_type == "call":
            model_price = S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(
                -r * T
            ) * norm.cdf(d2)
            vega = S0 * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
        else:
            model_price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(
                -q * T
            ) * norm.cdf(-d1)
            vega = S0 * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
        if vega < 1e-12:
            break
        diff = model_price - price
        if abs(diff) < 1e-8:
            break
        sigma -= diff / vega
    return sigma


# =============================================================================
# numba kernels (plain-Python fallback if numba is unavailable)
# =============================================================================
@njit(cache=False)
def _norm_cdf_arr(x):
    """Normal CDF on an array (scalar math.erf is the only erf numba accepts)."""
    out = np.empty_like(x)
    for i in range(x.size):
        out[i] = 0.5 * (1.0 + math.erf(x[i] / math.sqrt(2.0)))
    return out


@njit(cache=False)
def _norm_pdf_arr(x):
    """Normal PDF on an array."""
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


@njit(cache=False)
def _bs_implied_vol_vec_nb(price, K, S0, T, r, q, is_put, max_iter, tol):
    """Vectorized Newton-Raphson implied volatility (numba kernel)."""
    # convert puts to calls via put-call parity
    price_c = price + is_put * (S0 * np.exp(-q * T) - K * np.exp(-r * T))
    sigma = np.full(price.shape, 0.3)
    positive = price > 0
    active = positive.copy()
    sqT = np.sqrt(T)
    for _ in range(max_iter):
        if not np.any(active):
            break
        d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqT)
        d2 = d1 - sigma * sqT
        model = S0 * np.exp(-q * T) * _norm_cdf_arr(d1) - K * np.exp(
            -r * T
        ) * _norm_cdf_arr(d2)
        vega = S0 * np.exp(-q * T) * _norm_pdf_arr(d1) * sqT
        diff = model - price_c
        good_vega = np.abs(vega) > 1e-12
        step = np.where(good_vega, diff / np.where(good_vega, vega, 1.0), 0.0)
        sigma_new = np.clip(sigma - step, 1e-6, 5.0)
        converged = np.abs(diff) < tol
        sigma = np.where(converged, sigma, sigma_new)
        active &= ~converged
    sigma[~positive] = 0.0
    return sigma


def bs_implied_vol_vec(price, S0, K, T, r, q, opt_type="call", max_iter=100, tol=1e-8):
    """Vectorized Newton-Raphson implied volatility (arrays in, arrays out)."""
    price = np.atleast_1d(np.asarray(price, dtype=float))
    K = np.atleast_1d(np.asarray(K, dtype=float))
    T = float(T)
    if np.ndim(opt_type) == 0:
        is_put = np.full(price.shape, opt_type == "put", dtype=bool)
    else:
        is_put = np.asarray(opt_type) == "put"
    return _bs_implied_vol_vec_nb(
        price, K, S0, T, r, q, is_put, int(max_iter), float(tol)
    )


# =============================================================================
# 2. Heston Characteristic Functions (Little Trap)
# =============================================================================
@njit(cache=False)
def _heston_char_funcs_nb(u, T, S0, r, q, v0, kappa, theta, sigma, rho):
    """Both Heston characteristic functions (Little Trap), j=1 and j=2 (numba kernel)."""
    iu = 1j * u
    sigma2 = sigma * sigma
    a = kappa * theta
    logS0 = np.log(S0)
    # j = 1
    b1 = kappa - rho * sigma
    d1 = np.sqrt((rho * sigma * iu - b1) ** 2 - sigma2 * (iu - u * u))
    d1 = np.where(np.real(d1) >= 0, d1, -d1)
    g1 = (b1 - rho * sigma * iu + d1) / (b1 - rho * sigma * iu - d1)
    exp1 = np.exp(d1 * T)
    num1 = 1.0 - g1 * exp1
    den1 = 1.0 - g1
    D1 = (b1 - rho * sigma * iu + d1) / sigma2 * ((1.0 - exp1) / (1.0 - g1 * exp1))
    C1 = (r - q) * iu * T + a / sigma2 * (
        (b1 - rho * sigma * iu + d1) * T - 2.0 * np.log(num1 / den1)
    )
    phi1 = np.exp(C1 + D1 * v0 + iu * logS0)

    # j = 2
    b2 = kappa
    d2 = np.sqrt((rho * sigma * iu - b2) ** 2 - sigma2 * (-iu - u * u))
    d2 = np.where(np.real(d2) >= 0, d2, -d2)
    g2 = (b2 - rho * sigma * iu + d2) / (b2 - rho * sigma * iu - d2)
    exp2 = np.exp(d2 * T)
    num2 = 1.0 - g2 * exp2
    den2 = 1.0 - g2
    D2 = (b2 - rho * sigma * iu + d2) / sigma2 * ((1.0 - exp2) / (1.0 - g2 * exp2))
    C2 = (r - q) * iu * T + a / sigma2 * (
        (b2 - rho * sigma * iu + d2) * T - 2.0 * np.log(num2 / den2)
    )
    phi2 = np.exp(C2 + D2 * v0 + iu * logS0)
    return phi1, phi2


def heston_char_funcs(u, T, S0, r, q, v0, kappa, theta, sigma, rho):
    """Both Heston characteristic functions (Little Trap), j=1 and j=2, one pass."""
    u = np.asarray(u, dtype=float)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return _heston_char_funcs_nb(u, T, S0, r, q, v0, kappa, theta, sigma, rho)


def heston_char_func(u, T, S0, r, q, v0, kappa, theta, sigma, rho, j):
    """Single Heston characteristic function (kept for compatibility)."""
    phi1, phi2 = heston_char_funcs(u, T, S0, r, q, v0, kappa, theta, sigma, rho)
    return phi1 if j == 1 else phi2


@njit(cache=False)
def _heston_call_prices_nb(S0, K, T, r, q, v0, kappa, theta, sigma, rho, u_nodes, wts):
    """Heston call prices for a 1D strike array K (numba kernel)."""
    phi1, phi2 = _heston_char_funcs_nb(
        u_nodes, T, S0, r, q, v0, kappa, theta, sigma, rho
    )
    logK = np.log(K)
    e = np.exp(-1j * u_nodes[:, None] * logK[None, :])
    integrand1 = np.real(e * phi1[:, None] / (1j * u_nodes[:, None]))
    integrand2 = np.real(e * phi2[:, None] / (1j * u_nodes[:, None]))
    int1 = wts @ integrand1
    int2 = wts @ integrand2
    P1 = 0.5 + int1 / np.pi
    P2 = 0.5 + int2 / np.pi
    call = S0 * np.exp(-q * T) * P1 - K * np.exp(-r * T) * P2
    return np.maximum(call, 0.0)


def heston_call_prices(S0, K, T, r, q, v0, kappa, theta, sigma, rho, n=64, max_u=500):
    """Heston call prices for an array of strikes K in ONE vectorized quadrature."""
    K = np.asarray(K, dtype=float)
    shape = K.shape
    K = K.ravel()
    if T <= 0:
        return np.maximum(S0 - K, 0.0).reshape(shape)
    u_nodes, wts = _quad_rule(n, max_u)
    call = _heston_call_prices_nb(
        S0, K, T, r, q, v0, kappa, theta, sigma, rho, u_nodes, wts
    )
    return call.reshape(shape)


def heston_call_price(S0, K, T, r, q, v0, kappa, theta, sigma, rho, n=64, max_u=500):
    """Scalar Heston call price (kept for compatibility)."""
    if T <= 0:
        return max(S0 - K, 0.0)
    return float(
        heston_call_prices(
            S0,
            np.asarray([K], dtype=float),
            T,
            r,
            q,
            v0,
            kappa,
            theta,
            sigma,
            rho,
            n=n,
            max_u=max_u,
        )[0]
    )


# =============================================================================
# 3. Calibration Setup (vectorized)
# =============================================================================
def prepare_market(market_data, r, q_curve, S0):
    """
    Extract tenors, strikes, target IVs, and q_bar(T) once.

    q_curve can be a DividendCarryCurve or a scalar yield for backward
    compatibility.  Each prepared group is (T, K, target_iv, q_effective).
    """
    T_all = market_data["T"].to_numpy(dtype=float)
    K_all = market_data["K"].to_numpy(dtype=float)
    has_iv = "IV" in market_data.columns

    if not has_iv:
        prices_all = market_data["price"].to_numpy(dtype=float)
        if "OptionType" in market_data.columns:
            option_types = market_data["OptionType"].to_numpy()
        else:
            option_types = np.array(["call"] * len(market_data))

    groups = []
    for T in np.unique(T_all):
        m = T_all == T
        q_T = effective_q_at(q_curve, T)
        if has_iv:
            target_iv = market_data.loc[m, "IV"].to_numpy(dtype=float)
        else:
            target_iv = bs_implied_vol_vec(
                prices_all[m], S0, K_all[m], T, r, q_T, option_types[m]
            )
        groups.append((float(T), K_all[m], target_iv, q_T))
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
    values = np.asarray(values, dtype=float)
    lower = np.asarray([bound[0] for bound in bounds], dtype=float)
    upper = np.asarray([bound[1] for bound in bounds], dtype=float)
    return np.clip(values, lower, upper)


def _validate_history_within_global_bounds(history, active_names):
    for name in active_names:
        lower, upper = GLOBAL_PARAMETER_BOUNDS[name]
        if not lower <= history[name] <= upper:
            raise ValueError(
                f"historical {name}={history[name]} is outside global bounds "
                f"[{lower}, {upper}]"
            )


def compute_iv_error(
    params,
    market_data,
    r,
    q_curve,
    S0,
    penalty_weight=0.0,
    penalty_sigma=None,
    historical_params=None,
    history_penalty_iv=0.0,
    history_tolerances=None,
    history_active_names=None,
    weight_by_T=False,
    weight_T_floor=0.05,
    weight_T_power=1.0,
    iv_cap=0.25,
    prepared=None,
    quad_n=64,
    max_u=500.0,
):
    """Weighted mean squared IV error using the dividend curve at each tenor."""
    v0, kappa, theta, sigma, rho = params
    if prepared is None:
        prepared = prepare_market(market_data, r, q_curve, S0)

    total_err = 0.0
    total_w = 0.0
    for T, K, target_iv, q_T in prepared:
        w = 1.0 / (T + weight_T_floor) ** weight_T_power if weight_by_T else 1.0
        model_price = heston_call_prices(
            S0, K, T, r, q_T, v0, kappa, theta, sigma, rho, n=quad_n, max_u=max_u
        )
        model_iv = bs_implied_vol_vec(model_price, S0, K, T, r, q_T, "call")
        ok = np.isfinite(model_iv) & (model_iv > 0.0) & (model_iv < 3.0)
        errs = np.where(ok, (model_iv - target_iv) ** 2, iv_cap)
        total_err += w * np.sum(errs)
        total_w += w * K.size

    mse = total_err / total_w if total_w > 0 else 1e10

    if penalty_weight > 0 and penalty_sigma is not None:
        mse += penalty_weight * (sigma - penalty_sigma) ** 2
    if historical_params is not None and history_penalty_iv > 0.0:
        history_distance = historical_parameter_distance(
            params,
            historical_params,
            history_tolerances=history_tolerances,
            active_names=history_active_names,
        )
        mse += float(history_penalty_iv) ** 2 * history_distance
    return mse


def calibrate_heston_gatheral(
    market_data,
    r,
    q_curve,
    S0,
    fix_kappa=False,
    kappa_fixed=2.0,
    stage1_fixed_rho=-0.7,
    stage2_penalty=1.0,
    weight_by_T=False,
    weight_T_power=1.0,
    de_quad_n=32,
    polish_quad_n=64,
    de_maxiter1=500,
    de_maxiter2=1000,
    workers=1,
    max_u=500.0,
    use_historical=False,
    historical_params=None,
    history_penalty_iv=0.002,
    history_tolerances=None,
    warm_start_from_history=True,
    use_historical_rho_in_stage1=True,
):
    """Run two-stage Gatheral calibration with optional historical stability.

    Set ``use_historical=True`` and provide the prior day's five Heston
    parameters through ``historical_params`` to activate the normalized
    penalty. ``history_penalty_iv`` is in decimal IV units: ``0.002`` makes an
    average one-tolerance move cost the same as 0.2 volatility points of IV
    RMSE. Historical values are also injected into differential evolution when
    ``warm_start_from_history=True``.

    With ``use_historical=False`` the original optimizer behavior is retained.
    """
    prepared = prepare_market(market_data, r, q_curve, S0)
    if not np.isfinite(history_penalty_iv) or history_penalty_iv < 0.0:
        raise ValueError("history_penalty_iv must be finite and non-negative")

    history = None
    tolerances = None
    if use_historical:
        history = _coerce_historical_params(historical_params)
        tolerances = _resolve_history_tolerances(history_tolerances)
        active = [
            name
            for name in HESTON_PARAM_NAMES
            if not (fix_kappa and name == "kappa")
        ]
        _validate_history_within_global_bounds(history, active)

    rho_stage1 = float(stage1_fixed_rho)
    if history is not None and use_historical_rho_in_stage1:
        rho_stage1 = history["rho"]

    # --- Stage 1: fit {v0, kappa, theta, sigma} with rho fixed ---
    bounds_stage1 = [
        GLOBAL_PARAMETER_BOUNDS["v0"],
        GLOBAL_PARAMETER_BOUNDS["kappa"],
        GLOBAL_PARAMETER_BOUNDS["theta"],
        GLOBAL_PARAMETER_BOUNDS["sigma"],
    ]
    if fix_kappa:
        if not (
            GLOBAL_PARAMETER_BOUNDS["kappa"][0]
            <= kappa_fixed
            <= GLOBAL_PARAMETER_BOUNDS["kappa"][1]
        ):
            raise ValueError("kappa_fixed is outside the global kappa bounds")
        bounds_stage1[1] = (kappa_fixed - 1e-6, kappa_fixed + 1e-6)

    stage1_active_names = ["v0", "kappa", "theta", "sigma"]
    if fix_kappa:
        stage1_active_names.remove("kappa")

    def obj_stage1(params4, quad_n):
        v0, kappa, theta, sigma = params4
        full = [v0, kappa, theta, sigma, rho_stage1]
        return compute_iv_error(
            full,
            market_data,
            r,
            q_curve,
            S0,
            historical_params=history,
            history_penalty_iv=history_penalty_iv,
            history_tolerances=tolerances,
            history_active_names=stage1_active_names,
            prepared=prepared,
            quad_n=quad_n,
            max_u=max_u,
        )

    de1_extra = {}
    if history is not None and warm_start_from_history:
        x0_stage1 = [history[name] for name in HESTON_PARAM_NAMES[:4]]
        if fix_kappa:
            x0_stage1[1] = kappa_fixed
        de1_extra["x0"] = _clip_to_bounds(x0_stage1, bounds_stage1)

    result1 = differential_evolution(
        lambda p: obj_stage1(p, de_quad_n),
        bounds_stage1,
        maxiter=de_maxiter1,
        popsize=20,
        tol=1e-8,
        disp=False,
        seed=42,
        polish=False,
        workers=workers,
        **de1_extra,
    )
    result1 = minimize(
        lambda p: obj_stage1(p, polish_quad_n),
        result1.x,
        method="L-BFGS-B",
        bounds=bounds_stage1,
        options={"maxiter": 200},
    )

    v0_s1, kappa_s1, theta_s1, sigma_s1 = result1.x
    print(
        f"Stage 1 complete: v0={v0_s1:.4f}, kappa={kappa_s1:.4f}, "
        f"theta={theta_s1:.4f}, sigma={sigma_s1:.4f}, "
        f"fixed rho={rho_stage1:.4f}"
    )

    # --- Stage 2: fit full smile, free rho and sigma (soft anchor) ---
    bounds_stage2 = [
        (v0_s1 * 0.7, v0_s1 * 1.3),
        (kappa_s1 * 0.5, kappa_s1 * 2.0),
        (theta_s1 * 0.7, theta_s1 * 1.1),
        GLOBAL_PARAMETER_BOUNDS["sigma"],
        GLOBAL_PARAMETER_BOUNDS["rho"],
    ]
    if fix_kappa:
        bounds_stage2[1] = (kappa_s1 - 1e-6, kappa_s1 + 1e-6)

    stage2_active_names = list(HESTON_PARAM_NAMES)
    if fix_kappa:
        stage2_active_names.remove("kappa")

    # The historical point must remain feasible in Stage 2; otherwise the
    # Stage-1-centered bounds could make the prior and warm start unreachable.
    if history is not None:
        for index, name in enumerate(HESTON_PARAM_NAMES):
            if name in stage2_active_names:
                lower, upper = bounds_stage2[index]
                bounds_stage2[index] = (
                    min(lower, history[name]),
                    max(upper, history[name]),
                )

    def obj_stage2(params5, quad_n):
        return compute_iv_error(
            params5,
            market_data,
            r,
            q_curve,
            S0,
            penalty_weight=stage2_penalty,
            penalty_sigma=sigma_s1,
            historical_params=history,
            history_penalty_iv=history_penalty_iv,
            history_tolerances=tolerances,
            history_active_names=stage2_active_names,
            weight_by_T=weight_by_T,
            weight_T_power=weight_T_power,
            prepared=prepared,
            quad_n=quad_n,
            max_u=max_u,
        )

    de2_extra = {}
    if history is not None and warm_start_from_history:
        x0_stage2 = [history[name] for name in HESTON_PARAM_NAMES]
        if fix_kappa:
            x0_stage2[1] = kappa_s1
        de2_extra["x0"] = _clip_to_bounds(x0_stage2, bounds_stage2)

    result2 = differential_evolution(
        lambda p: obj_stage2(p, de_quad_n),
        bounds_stage2,
        maxiter=de_maxiter2,
        popsize=30,
        tol=1e-8,
        disp=False,
        seed=42,
        polish=False,
        workers=workers,
        **de2_extra,
    )
    result2 = minimize(
        lambda p: obj_stage2(p, polish_quad_n),
        result2.x,
        method="L-BFGS-B",
        bounds=bounds_stage2,
        options={"maxiter": 200},
    )

    v0, kappa, theta, sigma, rho = result2.x
    print(
        f"Stage 2 complete: v0={v0:.4f}, kappa={kappa:.4f}, "
        f"theta={theta:.4f}, sigma={sigma:.4f}, rho={rho:.4f}"
    )

    calibrated_params = {
        "v0": v0,
        "kappa": kappa,
        "theta": theta,
        "sigma": sigma,
        "rho": rho,
    }

    # Fitted IVs and curve diagnostics, aligned to original row order.
    fitted = np.full(len(market_data), np.nan)
    q_used = np.full(len(market_data), np.nan)
    Q_used = np.full(len(market_data), np.nan)
    forwards = np.full(len(market_data), np.nan)
    T_all = market_data["T"].to_numpy(dtype=float)

    for T, K, _, q_T in prepared:
        pr = heston_call_prices(
            S0, K, T, r, q_T, v0, kappa, theta, sigma, rho, n=polish_quad_n, max_u=max_u
        )
        iv = bs_implied_vol_vec(pr, S0, K, T, r, q_T, "call")
        m = T_all == T
        fitted[m] = iv
        q_used[m] = q_T
        Q_T = cumulative_q_at(q_curve, T)
        Q_used[m] = Q_T
        forwards[m] = S0 * np.exp(r * T - Q_T)

    result_df = market_data.copy()
    result_df["q_used"] = q_used
    result_df["Q_cumulative"] = Q_used
    result_df["Forward"] = forwards
    result_df["Fitted_IV"] = fitted
    if "IV" in market_data.columns:
        result_df["IV_error"] = result_df["Fitted_IV"] - result_df["IV"]

    # Preserve the existing two-value return signature and expose diagnostics
    # through DataFrame attrs for callers that want to audit regularization.
    unpenalized_mse = compute_iv_error(
        result2.x,
        market_data,
        r,
        q_curve,
        S0,
        weight_by_T=weight_by_T,
        weight_T_power=weight_T_power,
        prepared=prepared,
        quad_n=polish_quad_n,
        max_u=max_u,
    )
    diagnostics = {
        "unpenalized_iv_mse": float(unpenalized_mse),
        "unpenalized_iv_rmse": float(np.sqrt(unpenalized_mse)),
        "history_enabled": history is not None,
        "history_penalty_iv": (
            float(history_penalty_iv) if history is not None else 0.0
        ),
        "history_distance": 0.0,
        "history_penalty_contribution": 0.0,
    }
    if history is not None:
        distance = historical_parameter_distance(
            result2.x,
            history,
            history_tolerances=tolerances,
            active_names=stage2_active_names,
        )
        diagnostics["history_distance"] = distance
        diagnostics["history_penalty_contribution"] = (
            history_penalty_iv**2 * distance
        )
        print(f"Unpenalized weighted IV RMSE: {np.sqrt(unpenalized_mse):.6f}")
        print(f"Historical normalized distance: {distance:.6f}")
        print(
            "Historical penalty contribution: "
            f"{diagnostics['history_penalty_contribution']:.8g}"
        )
    result_df.attrs["calibration_diagnostics"] = diagnostics

    return calibrated_params, result_df


def load_vol_surface(file_name):
    """Load a wide Excel volatility surface into long-form market data."""
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


def heston_put_price(S0, K, T, r, q, v0, kappa, theta, sigma, rho, n=64, max_u=500):
    """Heston put via the same characteristic-function integrals as the call."""
    if T <= 0:
        return max(K - S0, 0.0)

    def i1(u):
        phi1 = heston_char_func(u, T, S0, r, q, v0, kappa, theta, sigma, rho, j=1)
        return np.real(np.exp(-1j * u * np.log(K)) * phi1 / (1j * u))

    def i2(u):
        phi2 = heston_char_func(u, T, S0, r, q, v0, kappa, theta, sigma, rho, j=2)
        return np.real(np.exp(-1j * u * np.log(K)) * phi2 / (1j * u))

    eps = 1e-12
    P1 = 0.5 + fixed_quad(i1, eps, max_u, n=n)[0] / np.pi  # P(S_T>K) under M1
    P2 = 0.5 + fixed_quad(i2, eps, max_u, n=n)[0] / np.pi  # under M2
    put = K * np.exp(-r * T) * (1.0 - P2) - S0 * np.exp(-q * T) * (1.0 - P1)
    return max(put, 0.0)


def heston_forward_cf(S0, T, r, q, v0, kappa, theta, sigma, rho, s=1.0 - 1e-9):
    """
    E[S_T] via the characteristic function (log MGF), evaluated at s -> 1.
    E[S_T] = S0 e^((r-q)T) * M_var(s),  M_var(s) = exp(C_v(s)+D_v(s)*v0),
    and the martingale property gives M_var(1) = 1. s is nudged just below 1
    to avoid the Little-Trap singularity of the char function exactly at s=1.
    """
    d = np.sqrt((kappa - rho * sigma * s) ** 2 + sigma**2 * s * (1.0 - s))
    g = (kappa - rho * sigma * s + d) / (kappa - rho * sigma * s - d)
    exp_dT = np.exp(d * T)
    D = (
        (kappa - rho * sigma * s + d)
        / sigma**2
        * (1.0 - np.exp(-d * T))
        / (1.0 - g * np.exp(-d * T))
    )
    C = (kappa * theta / sigma**2) * (
        (kappa - rho * sigma * s + d) * T - 2.0 * np.log((1.0 - g * exp_dT) / (1.0 - g))
    )
    Mv = np.exp(C + D * v0)
    return S0 * np.exp((r - q) * T) * Mv


def run_no_arbitrage_checks(
    md, P, S0, r, q_curve, label, tol_butt=1e-4, tol_cal=1e-4, fwd_ok=1e-6
):
    v0, kappa, theta, sigma, rho = P["v0"], P["kappa"], P["theta"], P["sigma"], P["rho"]
    print(f"\n{'=' * 64}\n{label}   (S0={S0}, r={r}, maturity-specific q)\n{'=' * 64}")
    tenors = np.sort(md["T"].unique())

    print("\nDividend curve used at quoted maturities")
    for T in tenors:
        q_T = effective_q_at(q_curve, T)
        Q_T = cumulative_q_at(q_curve, T)
        fwd = S0 * np.exp(r * T - Q_T)
        print(f"    T={T:.6f}: q_eff={q_T:.6%}  Q={Q_T:.8f}  F={fwd:.2f}")

    print("\n[0] true martingale check rho <= kappa/sigma")
    true_mtg_chck = rho - kappa / sigma
    print(f"    rho - kappa/sigma = {true_mtg_chck:.4f}")

    print("\n[1] Forward via characteristic function vs S0 exp(rT-Q(T))")
    fwd_err = 0.0
    for T in tenors:
        q_T = effective_q_at(q_curve, T)
        Q_T = cumulative_q_at(q_curve, T)
        f_th = S0 * np.exp(r * T - Q_T)
        f_cf = heston_forward_cf(S0, T, r, q_T, v0, kappa, theta, sigma, rho)
        rel = abs((f_cf - f_th) / f_th)
        fwd_err = max(fwd_err, rel)
        print(f"    T={T:.3f}:  CF={f_cf:10.2f}  theory={f_th:10.2f}  rel={rel:.2e}")
    print(
        f"    -> max rel fwd error = {fwd_err:.2e}   {'PASS' if fwd_err < fwd_ok else 'FAIL'}"
    )

    print("\n[2] Model put-call parity  |C - P - (S0 exp(-Q(T)) - K exp(-rT))|")
    max_diff, max_rel, npts = 0.0, 0.0, 0
    for T in tenors[:3]:
        q_T = effective_q_at(q_curve, T)
        Q_T = cumulative_q_at(q_curve, T)
        sub = md[np.isclose(md["T"], T)].sort_values("K")
        Ks = sub["K"].values
        for K in Ks[:: max(1, len(Ks) // 4)]:
            C = heston_call_price(S0, K, T, r, q_T, v0, kappa, theta, sigma, rho)
            Pp = heston_put_price(S0, K, T, r, q_T, v0, kappa, theta, sigma, rho)
            parity = S0 * np.exp(-Q_T) - K * np.exp(-r * T)
            d = abs(C - Pp - parity)
            max_diff = max(max_diff, d)
            max_rel = max(max_rel, d / (K * np.exp(-r * T)))
            npts += 1
    print(
        f"    ({npts} pts)  max |C-P-parity| = {max_diff:.2e}   max rel = {max_rel:.2e}"
    )

    def slope_viol(Cs, Ks, tol):
        slopes = np.diff(Cs) / np.diff(Ks)
        dv = np.diff(slopes)
        bad = dv < -tol
        return int(bad.sum()), (float(-dv[bad].min()) if bad.any() else 0.0)

    print(
        "\n[3] Butterfly (convexity): call-slope decreases -> violations (tol %.0e)"
        % tol_butt
    )
    print(
        f"    {'T':>7} | {'mkt viol':>8} {'mkt max':>10} | {'model viol':>10} {'model max':>10}"
    )
    for T in tenors:
        q_T = effective_q_at(q_curve, T)
        sub = md[np.isclose(md["T"], T)].sort_values("K").reset_index(drop=True)
        if len(sub) < 3:
            continue
        Ks = sub["K"].values
        Cs_m = np.array(
            [
                bs_call_price(S0, Ks[i], T, r, q_T, sub["IV"].values[i])
                for i in range(len(Ks))
            ]
        )
        Cs_M = np.array(
            [
                heston_call_price(S0, Ks[i], T, r, q_T, v0, kappa, theta, sigma, rho)
                for i in range(len(Ks))
            ]
        )
        nv_m, mx_m = slope_viol(Cs_m, Ks, tol_butt)
        nv_M, mx_M = slope_viol(Cs_M, Ks, tol_butt)
        print(f"    {T:.3f} | {nv_m:8d} {mx_m:10.2e} | {nv_M:10d} {mx_M:10.2e}")

    print(
        "\n[4] Calendar: total variance w=IV^2*T non-decreasing in T (per strike, tol %.0e)"
        % tol_cal
    )
    mdw = md.copy()
    mdw["w"] = mdw["IV"] ** 2 * mdw["T"]
    viol_m, viol_M, nchecks, maxdec_m, maxdec_M = 0, 0, 0, 0.0, 0.0
    for K in sorted(mdw["K"].unique()):
        sub = mdw[mdw["K"] == K].sort_values("T")
        if len(sub) < 2:
            continue
        Ts = sub["T"].values
        wm = sub["w"].values
        wM = np.array(
            [
                bs_implied_vol(
                    heston_call_price(
                        S0,
                        K,
                        T,
                        r,
                        effective_q_at(q_curve, T),
                        v0,
                        kappa,
                        theta,
                        sigma,
                        rho,
                    ),
                    S0,
                    K,
                    T,
                    r,
                    effective_q_at(q_curve, T),
                    "call",
                )
                ** 2
                * T
                for T in Ts
            ]
        )
        for i in range(len(Ts) - 1):
            nchecks += 1
            if wm[i + 1] < wm[i] - tol_cal:
                viol_m += 1
                maxdec_m = max(maxdec_m, wm[i] - wm[i + 1])
            if wM[i + 1] < wM[i] - tol_cal:
                viol_M += 1
                maxdec_M = max(maxdec_M, wM[i] - wM[i + 1])
    print(
        f"    {nchecks} adjacent-T checks: market viol={viol_m} "
        f"(max dec {maxdec_m:.2e}) | model viol={viol_M} "
        f"(max dec {maxdec_M:.2e})"
    )


# Backward-compatible names used by the legacy notebooks.
data_processer = load_vol_surface
no_arb_checks = run_no_arbitrage_checks
