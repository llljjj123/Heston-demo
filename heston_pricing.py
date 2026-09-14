"""Heston ADI structured-product pricing and finite-difference Greeks.

The public pricing functions return scalar prices. Dividend input may be a
constant yield, a DividendCarryCurve, or an effective-yield array accompanied
by matching maturities.
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


@njit(cache=False)
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


@njit(cache=False)
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


# ---------- ADI step (vectorized) ----------
@njit(cache=False)
def heston_adi_step(
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
):
    """
    One time step of the Douglas ADI scheme for the Heston PDE (vectorized).

    U : 2D array (Ns, Nv), current solution.
    lower_boundary : scalar Dirichlet value at S=0 at the end of this step.
    Returns updated U.

    Same scheme and boundary treatment as HST_PDE_funcs.heston_adi_step:
    upwind S-convection (forward when r>q, backward when q>r), central
    differences in v, explicit mixed-derivative term, product-specific
    Dirichlet value at S=0, zero gamma (linear extrapolation) at S_max, and
    linear extrapolation of the mixed term at the top v boundary.
    """
    Ns, Nv = U.shape

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
    a_S = -0.5 * dt * (Lm_c[:, None] + D).T  # (Nv, Nint)
    b_S = 1.0 - 0.5 * dt * (Li_c[:, None] - 2.0 * D).T
    c_S = -0.5 * dt * (Lp_c[:, None] + D).T
    rhs = (Y0[1:-1, :] - 0.5 * dt * L_S_U[1:-1, :]).T
    # The first interior equation contains a_S[:, 0] * U(S=0).  Move the
    # known Dirichlet value to the right-hand side before solving.
    rhs[:, 0] -= a_S[:, 0] * lower_boundary
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

    a_v = np.broadcast_to(-0.5 * dt * Lm_v[None, :], (Ns, Nv))
    b_v = np.broadcast_to(1.0 - 0.5 * dt * Li_v[None, :], (Ns, Nv))
    c_v = np.broadcast_to(-0.5 * dt * Lp_v[None, :], (Ns, Nv))
    rhs_v = Y1 - 0.5 * dt * L_v_U
    Y2 = thomas_algorithm_batch(a_v[:, 1:], b_v, c_v[:, :-1], rhs_v)

    # enforce S boundaries again
    Y2[0, :] = lower_boundary
    Y2[-1, :] = 2.0 * Y2[-2, :] - Y2[-3, :]
    return Y2


# Later, evaluate at any S with:
#     S, U_at_v0 = price_xxx_v0curve(...)
#     f = interp1d(S, U_at_v0, kind='linear', bounds_error=False, fill_value='extrapolate')
#     price_at_S = float(f(S_any))


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
    Nt_uniform=200,
    daily_ki=True,
    SPD=0.0,
    pay_times=None,
    mg=0.0,
    mg_ist=None,
    K=None,
    q_maturities=None,
):
    """Snowball price; q may be scalar, a curve, or an array with q_maturities."""
    q_input = prepare_dividend_input(q, q_maturities)
    obs_yr = np.asarray(obs_times, dtype=float) / 252.0
    if pay_times is None:
        pay_yr = obs_yr.copy()
    else:
        pay_yr = np.asarray(pay_times, dtype=float) / 360.0
    if mg_ist is None:
        mg_ist = r
    mgi = mg * mg_ist
    if K is None:
        K = S0

    order = np.argsort(obs_yr)
    obs_yr = obs_yr[order]
    pay_yr = pay_yr[order]
    knock_out = np.asarray(knock_out, dtype=float)
    coupon = np.asarray(coupon, dtype=float)
    if knock_out.ndim == 0:
        knock_out = np.full(obs_yr.size, float(knock_out))
    if coupon.ndim == 0:
        coupon = np.full(obs_yr.size, float(coupon))
    knock_out = knock_out[order]
    coupon = coupon[order]
    if not (knock_out.size == coupon.size == obs_yr.size == pay_yr.size):
        raise ValueError(
            "coupon, knock_out, obs_times and pay_times must all have the same length"
        )

    keep = obs_yr <= T
    obs_yr = obs_yr[keep]
    pay_yr = pay_yr[keep]
    knock_out = knock_out[keep]
    coupon = coupon[keep]
    _, uniq = np.unique(obs_yr, return_index=True)
    obs_yr = obs_yr[uniq]
    pay_yr = pay_yr[uniq]
    knock_out = knock_out[uniq]
    coupon = coupon[uniq]

    period_len = np.diff(np.concatenate(([0.0], pay_yr)))
    acc_coupon = np.cumsum(coupon * period_len)
    stub = coupon[-1] * max(T - pay_yr[-1], 0.0)
    full_coupon = acc_coupon[-1] + stub
    # maturity margin over the full coupon horizon (from issue), like the coupon
    full_pay = max(pay_yr[-1], T)

    S_max = max(S0 * 3.0, 2.0 * float(np.max(knock_out)))
    v_max = max(0.5, 4 * theta, 4 * v0)
    S = np.linspace(0, S_max, Ns)
    dS = S[1] - S[0]
    v = np.linspace(0, v_max, Nv)
    dv = v[1] - v[0]

    tau_obs = T - obs_yr
    tau_uniform = np.linspace(0, T, Nt_uniform)
    tau_points = np.unique(np.sort(np.concatenate(([0, T], tau_obs, tau_uniform))))
    tau_points = tau_points[tau_points >= 0]

    payoff_in = np.where(S >= K, 1.0, np.maximum(S / K, SPD))
    U_in = np.tile((payoff_in - 1.0)[:, None], (1, Nv)) - mgi * full_pay
    if daily_ki:
        U_noin = np.full((Ns, Nv), full_coupon - mgi * full_pay)
    else:
        payoff_noin = np.where(S <= H_in, payoff_in - 1.0, full_coupon) - mgi * full_pay
        U_noin = np.tile(payoff_noin[:, None], (1, Nv))

    # At S=0 the underlying cannot recover or knock out.  Both state values
    # therefore equal the knocked-in terminal loss, discounted while stepping
    # backward.  Initialize the no-in boundary as well for the first PDE step.
    lower_terminal = float(U_in[0, 0])
    U_noin[0, :] = lower_terminal

    for k in range(len(tau_points) - 1):
        dt = tau_points[k + 1] - tau_points[k]
        if dt <= 0:
            continue
        q_step = dividend_rate_for_tau_step(
            q_input, T, tau_points[k], tau_points[k + 1]
        )
        lower_boundary = lower_terminal * np.exp(-r * tau_points[k + 1])
        U_noin = heston_adi_step(
            U_noin,
            dt,
            S,
            dS,
            v,
            dv,
            r,
            q_step,
            kappa,
            theta,
            sigma_v,
            rho,
            lower_boundary,
        )
        if daily_ki:
            U_in = heston_adi_step(
                U_in,
                dt,
                S,
                dS,
                v,
                dv,
                r,
                q_step,
                kappa,
                theta,
                sigma_v,
                rho,
                lower_boundary,
            )
            mask_in = S <= H_in
            U_noin[mask_in, :] = U_in[mask_in, :]
        tau_now = tau_points[k + 1]
        hits = np.where(np.abs(tau_obs - tau_now) < 1e-12)[0]
        if hits.size > 0:
            idx = int(hits[0])
            redemption = acc_coupon[idx] - mgi * pay_yr[idx]
            mask_out = S >= knock_out[idx]
            U_noin[mask_out, :] = redemption
            if daily_ki:
                U_in[mask_out, :] = redemption

        # ---------- interpolate initial value ----------
    interp = RegularGridInterpolator(
        (S, v), U_noin, bounds_error=False, fill_value=None
    )
    price = float(interp((S0, v0)))
    return price


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
    Nt_uniform=200,
    H_coupon=None,
    daily_ki=True,
    SPD=0.0,
    mg=0.0,
    mg_ist=None,
    full_term=None,
    K=None,
    q_maturities=None,
):
    """Return the scalar net-to-client Phoenix price.
    obs_times in years (absolute coupons, like the original). Pass full_term (the
    trade's total term from issue) to charge the maturity margin over the full term.
    K: strike of the knocked-in put (initial level at issue for a seasoned trade);
    if None, uses S0 (fresh-trade case).
    q may be scalar, a curve, or an array with q_maturities."""
    q_input = prepare_dividend_input(q, q_maturities)
    obs_times = np.asarray(obs_times, dtype=float)
    order = np.argsort(obs_times)
    obs_times = obs_times[order]
    if mg_ist is None:
        mg_ist = r
    mgi = mg * mg_ist
    if K is None:
        K = S0

    knock_out = np.asarray(knock_out, dtype=float)
    coupon = np.asarray(coupon, dtype=float)
    if knock_out.ndim == 0:
        knock_out = np.full(obs_times.size, float(knock_out))
    if coupon.ndim == 0:
        coupon = np.full(obs_times.size, float(coupon))
    knock_out = knock_out[order]
    coupon = coupon[order]
    if not (knock_out.size == coupon.size == obs_times.size):
        raise ValueError(
            "coupon, knock_out and obs_times must all have the same length"
        )

    keep = obs_times <= T
    obs_times = obs_times[keep]
    knock_out = knock_out[keep]
    coupon = coupon[keep]
    _, uniq = np.unique(obs_times, return_index=True)
    obs_times = obs_times[uniq]
    knock_out = knock_out[uniq]
    coupon = coupon[uniq]
    if obs_times.size == 0:
        raise ValueError("obs_times contains no dates within [0, T]")
    if H_coupon is None:
        H_coupon = H_in

    # maturity margin over the full horizon (from issue); pass full_term (the trade's
    # total term from issue, e.g. 2.0) for a seasoned trade, else falls back to T
    if full_term is None:
        full_term = obs_times[-1]
    full_pay = max(full_term, T)

    S_max = max(S0 * 3.0, 2.0 * float(np.max(knock_out)))
    v_max = max(0.5, 4 * theta, 4 * v0)
    S = np.linspace(0, S_max, Ns)
    dS = S[1] - S[0]
    v = np.linspace(0, v_max, Nv)
    dv = v[1] - v[0]

    tau_obs = T - obs_times
    tau_uniform = np.linspace(0, T, Nt_uniform)
    tau_points = np.unique(np.sort(np.concatenate(([0, T], tau_obs, tau_uniform))))
    tau_points = tau_points[tau_points >= 0]

    payoff_in = np.where(S >= K, 1.0, np.maximum(S / K, SPD))
    U_in = np.tile((payoff_in - 1.0)[:, None], (1, Nv)) - mgi * full_pay
    if daily_ki:
        U_noin = np.full((Ns, Nv), -mgi * full_pay)
    else:
        payoff_noin = np.where(S <= H_in, payoff_in - 1.0, 0.0) - mgi * full_pay
        U_noin = np.tile(payoff_noin[:, None], (1, Nv))

    if np.isclose(obs_times[-1], T):
        coupon_last = float(coupon[-1])
        mask_c = S >= H_coupon
        U_noin[mask_c, :] += coupon_last
        U_in[mask_c, :] += coupon_last
        mask_out = S >= knock_out[-1]
        U_noin[mask_out, :] = coupon_last - mgi * full_pay
        U_in[mask_out, :] = coupon_last - mgi * full_pay

    # At S=0 the underlying cannot reach either the coupon or knock-out
    # barrier.  Both state values reduce to the knocked-in terminal loss.
    lower_terminal = float(U_in[0, 0])
    U_noin[0, :] = lower_terminal

    for k in range(len(tau_points) - 1):
        dt = tau_points[k + 1] - tau_points[k]
        if dt <= 0:
            continue
        q_step = dividend_rate_for_tau_step(
            q_input, T, tau_points[k], tau_points[k + 1]
        )
        lower_boundary = lower_terminal * np.exp(-r * tau_points[k + 1])
        U_noin = heston_adi_step(
            U_noin,
            dt,
            S,
            dS,
            v,
            dv,
            r,
            q_step,
            kappa,
            theta,
            sigma_v,
            rho,
            lower_boundary,
        )
        if daily_ki:
            U_in = heston_adi_step(
                U_in,
                dt,
                S,
                dS,
                v,
                dv,
                r,
                q_step,
                kappa,
                theta,
                sigma_v,
                rho,
                lower_boundary,
            )
            mask_in = S <= H_in
            U_noin[mask_in, :] = U_in[mask_in, :]
        tau_now = tau_points[k + 1]
        hits = np.where(np.abs(tau_obs - tau_now) < 1e-12)[0]
        if hits.size > 0:
            idx = int(hits[0])
            mask_c = S >= H_coupon
            U_noin[mask_c, :] += coupon[idx]
            U_in[mask_c, :] += coupon[idx]
            mask_out = S >= knock_out[idx]
            U_noin[mask_out, :] = coupon[idx] - mgi * obs_times[idx]
            U_in[mask_out, :] = coupon[idx] - mgi * obs_times[idx]

    # ---------- interpolate initial value ----------
    interp = RegularGridInterpolator(
        (S, v), U_noin, bounds_error=False, fill_value=None
    )
    price = float(interp((S0, v0)))
    return price


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
    Nt_uniform=200,
    pay_times=None,
    mg=0.0,
    mg_ist=None,
    q_maturities=None,
):
    """Autocallable (小雪球), but returns (S, U(S, v0)) instead of the scalar net price."""
    q_input = prepare_dividend_input(q, q_maturities)
    obs_yr = np.asarray(obs_times, dtype=float) / 252.0
    if pay_times is None:
        pay_yr = obs_yr.copy()
    else:
        pay_yr = np.asarray(pay_times, dtype=float) / 360.0
    if mg_ist is None:
        mg_ist = r
    mgi = mg * mg_ist

    order = np.argsort(obs_yr)
    obs_yr = obs_yr[order]
    pay_yr = pay_yr[order]
    knock_out = np.asarray(knock_out, dtype=float)
    coupon = np.asarray(coupon, dtype=float)
    if knock_out.ndim == 0:
        knock_out = np.full(obs_yr.size, float(knock_out))
    if coupon.ndim == 0:
        coupon = np.full(obs_yr.size, float(coupon))
    knock_out = knock_out[order]
    coupon = coupon[order]
    if not (knock_out.size == coupon.size == obs_yr.size == pay_yr.size):
        raise ValueError(
            "coupon, knock_out, obs_times and pay_times must all have the same length"
        )

    keep = obs_yr <= T
    obs_yr = obs_yr[keep]
    pay_yr = pay_yr[keep]
    knock_out = knock_out[keep]
    coupon = coupon[keep]
    _, uniq = np.unique(obs_yr, return_index=True)
    obs_yr = obs_yr[uniq]
    pay_yr = pay_yr[uniq]
    knock_out = knock_out[uniq]
    coupon = coupon[uniq]
    if obs_yr.size == 0:
        raise ValueError("obs_times contains no dates within [0, T]")

    period_len = np.diff(np.concatenate(([0.0], pay_yr)))
    acc_coupon = np.cumsum(coupon * period_len)
    stub = coupon[-1] * max(T - pay_yr[-1], 0.0)
    full_coupon = acc_coupon[-1] + stub
    # maturity premium/margin over the full coupon horizon (from issue), like the coupon
    full_pay = max(pay_yr[-1], T)

    S_max = max(S0 * 3.0, 2.0 * float(np.max(knock_out)))
    v_max = max(0.5, 4 * theta, 4 * v0)
    S = np.linspace(0, S_max, Ns)
    dS = S[1] - S[0]
    v = np.linspace(0, v_max, Nv)
    dv = v[1] - v[0]

    tau_obs = T - obs_yr
    tau_uniform = np.linspace(0, T, Nt_uniform)
    tau_points = np.unique(np.sort(np.concatenate(([0, T], tau_obs, tau_uniform))))
    tau_points = tau_points[tau_points >= 0]

    U = np.full((Ns, Nv), -premium * full_pay - mgi * full_pay)
    if np.isclose(obs_yr[-1], T):
        mask_out = S >= knock_out[-1]
        U[mask_out, :] = full_coupon - premium * full_pay - mgi * full_pay

    lower_terminal = float(U[0, 0])

    for k in range(len(tau_points) - 1):
        dt = tau_points[k + 1] - tau_points[k]
        if dt <= 0:
            continue
        q_step = dividend_rate_for_tau_step(
            q_input, T, tau_points[k], tau_points[k + 1]
        )
        lower_boundary = lower_terminal * np.exp(-r * tau_points[k + 1])
        U = heston_adi_step(
            U,
            dt,
            S,
            dS,
            v,
            dv,
            r,
            q_step,
            kappa,
            theta,
            sigma_v,
            rho,
            lower_boundary,
        )
        tau_now = tau_points[k + 1]
        hits = np.where(np.abs(tau_obs - tau_now) < 1e-12)[0]
        if hits.size > 0:
            idx = int(hits[0])
            redemption = acc_coupon[idx] - premium * pay_yr[idx] - mgi * pay_yr[idx]
            mask_out = S >= knock_out[idx]
            U[mask_out, :] = redemption

    # ---------- interpolate initial value ----------
    interp = RegularGridInterpolator((S, v), U, bounds_error=False, fill_value=None)
    price = float(interp((S0, v0)))
    return price


def calculate_greeks(pricing_function, parameters, spot_bump=None, variance_bump=None):
    """Calculate delta, gamma, and Heston vega by central finite differences.

    Parameters
    ----------
    pricing_function : callable
        Any scalar pricing function that accepts its inputs as keyword
        arguments, including ``heston_call_price``,
        ``price_snowball_heston``, ``price_phoenix_heston``, and
        ``price_autocallable_heston``.
    parameters : mapping
        All keyword arguments required by ``pricing_function``.  It must
        contain ``S0`` (spot) and ``v0`` (initial Heston variance).  The input
        mapping is not modified.
    spot_bump : float, optional
        Absolute spot bump.  The default is 1% of ``S0``.
    variance_bump : float, optional
        Absolute ``v0`` bump.  The default is 1% of ``v0`` (with a floor of
        1e-6).  Vega is therefore d(price)/d(v0), matching the project's
        existing Heston-grid convention.

    Returns
    -------
    dict
        ``{"delta": ..., "gamma": ..., "vega": ...}``, in price units.
    """
    if not callable(pricing_function):
        raise TypeError("pricing_function must be callable")

    base = dict(parameters)
    missing = {"S0", "v0"}.difference(base)
    if missing:
        raise KeyError(f"parameters is missing: {', '.join(sorted(missing))}")

    spot = float(base["S0"])
    variance = float(base["v0"])
    if spot <= 0.0:
        raise ValueError("S0 must be positive")
    if variance <= 0.0:
        raise ValueError("v0 must be positive")

    d_spot = abs(float(spot_bump)) if spot_bump is not None else 0.01 * spot
    d_variance = (
        abs(float(variance_bump))
        if variance_bump is not None
        else max(0.01 * variance, 1e-6)
    )
    if d_spot == 0.0 or d_spot >= spot:
        raise ValueError("spot_bump must satisfy 0 < abs(spot_bump) < S0")
    if d_variance == 0.0 or d_variance >= variance:
        raise ValueError("variance_bump must satisfy 0 < abs(variance_bump) < v0")

    spot_up = base.copy()
    spot_up["S0"] = spot + d_spot
    spot_down = base.copy()
    spot_down["S0"] = spot - d_spot
    variance_up = base.copy()
    variance_up["v0"] = variance + d_variance
    variance_down = base.copy()
    variance_down["v0"] = variance - d_variance

    values = (
        pricing_function(**base),
        pricing_function(**spot_up),
        pricing_function(**spot_down),
        pricing_function(**variance_up),
        pricing_function(**variance_down),
    )
    try:
        (
            price,
            price_spot_up,
            price_spot_down,
            price_variance_up,
            price_variance_down,
        ) = map(float, values)
    except (TypeError, ValueError) as exc:
        raise TypeError("pricing_function must return one scalar price") from exc

    return {
        "delta": (price_spot_up - price_spot_down) / (2.0 * d_spot),
        "gamma": (price_spot_up - 2.0 * price + price_spot_down) / d_spot**2,
        "vega": (price_variance_up - price_variance_down) / (2.0 * d_variance),
    }
