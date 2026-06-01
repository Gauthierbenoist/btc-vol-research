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


def heston_call_price(
    S0: float,
    K: float,
    T: float,
    params: HestonParams,
    r: float = 0.0,
    q: float = 0.0,
) -> float:
    if T <= 0:
        return max(S0 * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
    today = _ensure_ql_settings()
    days = max(int(T * 365), 1)
    maturity = today + ql.Period(days, ql.Days)
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, float(K))
    exercise = ql.EuropeanExercise(maturity)
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(_build_engine(S0, T, params, r, q))
    return float(option.NPV())


def heston_put_price(
    S0: float,
    K: float,
    T: float,
    params: HestonParams,
    r: float = 0.0,
    q: float = 0.0,
) -> float:
    if T <= 0:
        return max(K * np.exp(-r * T) - S0 * np.exp(-q * T), 0.0)
    today = _ensure_ql_settings()
    days = max(int(T * 365), 1)
    maturity = today + ql.Period(days, ql.Days)
    payoff = ql.PlainVanillaPayoff(ql.Option.Put, float(K))
    exercise = ql.EuropeanExercise(maturity)
    option = ql.VanillaOption(payoff, exercise)
    option.setPricingEngine(_build_engine(S0, T, params, r, q))
    return float(option.NPV())


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
    prices = []
    for k, opt in zip(strikes, option_types):
        if str(opt).lower() == "call":
            prices.append(heston_call_price(S0, float(k), T, params, r, q))
        else:
            prices.append(heston_put_price(S0, float(k), T, params, r, q))
    prices = np.array(prices)
    return implied_volatility(
        prices,
        np.full(len(strikes), S0),
        strikes,
        np.full(len(strikes), T),
        r,
        q,
        option_types,
    )
