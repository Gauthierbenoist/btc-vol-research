"""Prix d'options européennes sous Heston (QuantLib — moteur analytique stable)."""

from __future__ import annotations

import numpy as np
import QuantLib as ql

from btc_vol_research.models.heston.params import HestonParams

_QL_INITIALIZED = False


def _ensure_ql_settings() -> ql.Date:
    global _QL_INITIALIZED
    today = ql.Date().todaysDate()
    if not _QL_INITIALIZED:
        ql.Settings.instance().evaluationDate = today
        _QL_INITIALIZED = True
    return today


def _maturity_date(T: float) -> ql.Date:
    today = _ensure_ql_settings()
    days = max(int(T * 365), 1)
    return today + ql.Period(days, ql.Days)


def _build_engine(
    S0: float,
    T: float,
    params: HestonParams,
    r: float,
    q: float,
) -> ql.AnalyticHestonEngine:
    today = _ensure_ql_settings()
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
    rate_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, r, ql.Actual365Fixed()))
    div_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, q, ql.Actual365Fixed()))
    process = ql.HestonProcess(
        rate_ts,
        div_ts,
        spot_handle,
        params.v0,
        params.kappa,
        params.theta,
        params.sigma,
        params.rho,
    )
    model = ql.HestonModel(process)
    return ql.AnalyticHestonEngine(model)


def _heston_price_with_engine(
    engine: ql.AnalyticHestonEngine,
    K: float,
    T: float,
    option_type: str,
) -> float:
    is_call = str(option_type).lower() == "call"
    opt_type = ql.Option.Call if is_call else ql.Option.Put
    payoff = ql.PlainVanillaPayoff(opt_type, float(K))
    exercise = ql.EuropeanExercise(_maturity_date(T))
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(engine)
    return float(option.NPV())


def heston_call_price(
    S0: float,
    K: float,
    T: float,
    params: HestonParams,
    r: float = 0.0,
    q: float = 0.0,
    *,
    engine: ql.AnalyticHestonEngine | None = None,
) -> float:
    if T <= 0:
        return max(S0 * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    eng = engine or _build_engine(S0, T, params, r, q)
    return _heston_price_with_engine(eng, K, T, "call")


def heston_put_price(
    S0: float,
    K: float,
    T: float,
    params: HestonParams,
    r: float = 0.0,
    q: float = 0.0,
    *,
    engine: ql.AnalyticHestonEngine | None = None,
) -> float:
    if T <= 0:
        return max(K * np.exp(-r * T) - S0 * np.exp(-q * T), 0.0)
    eng = engine or _build_engine(S0, T, params, r, q)
    return _heston_price_with_engine(eng, K, T, "put")


def heston_option_prices(
    S0: float,
    strikes: np.ndarray,
    T: float,
    params: HestonParams,
    r: float,
    q: float,
    option_types: np.ndarray,
) -> np.ndarray:
    """Prix Heston pour une grille de strikes (un moteur QuantLib par tranche)."""
    strikes = np.asarray(strikes, dtype=float)
    option_types = np.asarray(option_types)
    n = len(strikes)
    if T <= 0:
        is_call = np.char.lower(option_types.astype(str)) == "call"
        intrinsic_call = np.maximum(S0 * np.exp(-q * T) - strikes * np.exp(-r * T), 0.0)
        intrinsic_put = np.maximum(strikes * np.exp(-r * T) - S0 * np.exp(-q * T), 0.0)
        return np.where(is_call, intrinsic_call, intrinsic_put)

    engine = _build_engine(S0, T, params, r, q)
    prices = np.empty(n, dtype=float)
    for i, (k, opt) in enumerate(zip(strikes, option_types)):
        prices[i] = _heston_price_with_engine(engine, float(k), T, str(opt))
    return prices


def heston_iv_grid(
    S0: float,
    strikes: np.ndarray,
    T: float,
    params: HestonParams,
    r: float,
    q: float,
    option_types: np.ndarray | None = None,
) -> np.ndarray:
    from btc_vol_research.iv.black_scholes import implied_volatility

    if option_types is None:
        option_types = np.array(["call"] * len(strikes))
    prices = heston_option_prices(S0, strikes, T, params, r, q, option_types)
    return implied_volatility(
        prices,
        np.full(len(strikes), S0),
        strikes,
        np.full(len(strikes), T),
        r,
        q,
        option_types,
    )
