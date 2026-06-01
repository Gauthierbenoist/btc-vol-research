"""Configuration (YAML + variables d'environnement)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _env_bool(name: str, default: bool = True) -> bool:
    return os.getenv(name, str(default).lower()).lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class PostgresConfig:
    database_url: str = os.getenv("DATABASE_URL", "")

    @property
    def is_configured(self) -> bool:
        return bool(self.database_url)


@dataclass(frozen=True)
class MarketConfig:
    risk_free_rate: float = float(os.getenv("RISK_FREE_RATE", "0.0"))
    dividend_yield: float = float(os.getenv("DIVIDEND_YIELD", "0.0"))
    min_open_interest: float = float(os.getenv("MIN_OPEN_INTEREST", "0.01"))
    min_time_to_expiry_days: float = float(os.getenv("MIN_TIME_TO_EXPIRY_DAYS", "1"))
    max_time_to_expiry_years: float = float(os.getenv("MAX_TIME_TO_EXPIRY_YEARS", "1.0"))
    max_relative_spread: float = float(os.getenv("MAX_REL_SPREAD", "0.5"))
    smile_convention: str = "otm"


@dataclass(frozen=True)
class CalibrationConfig:
    atm_log_moneyness_sigma: float = float(os.getenv("CALIB_ATM_SIGMA", "0.15"))
    use_vega_weight: bool = _env_bool("CALIB_USE_VEGA_WEIGHT", True)
    use_liquidity_weight: bool = _env_bool("CALIB_USE_LIQUIDITY_WEIGHT", True)
    min_strikes_per_slice: int = 5
    optimizer: str = "L-BFGS-B"
    feller_penalty: float = 1000.0


@dataclass(frozen=True)
class HestonBounds:
    v0: tuple[float, float] = (0.001, 4.0)
    kappa: tuple[float, float] = (0.05, 15.0)
    theta: tuple[float, float] = (0.001, 4.0)
    sigma: tuple[float, float] = (0.05, 3.0)
    rho: tuple[float, float] = (-0.99, 0.99)


@dataclass(frozen=True)
class AppConfig:
    snapshot_date: str | None = None
    postgres: PostgresConfig = field(default_factory=PostgresConfig)
    market: MarketConfig = field(default_factory=MarketConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    heston_bounds: HestonBounds = field(default_factory=HestonBounds)
    figures_dir: Path = PROJECT_ROOT / "outputs" / "figures"
    reports_dir: Path = PROJECT_ROOT / "outputs" / "reports"


def _merge_yaml(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_yaml(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Path | None = None) -> AppConfig:
    cfg_path = path or PROJECT_ROOT / "configs" / "default.yaml"
    raw: dict[str, Any] = {}
    if cfg_path.exists():
        with cfg_path.open(encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    market_raw = raw.get("market", {})
    calib_raw = raw.get("calibration", {})
    heston_raw = raw.get("heston", {})
    outputs_raw = raw.get("outputs", {})

    hb = heston_raw.get("v0", [0.001, 4.0])
    bounds = HestonBounds(
        v0=tuple(heston_raw.get("v0", [0.001, 4.0])),
        kappa=tuple(heston_raw.get("kappa", [0.05, 15.0])),
        theta=tuple(heston_raw.get("theta", [0.001, 4.0])),
        sigma=tuple(heston_raw.get("sigma", [0.05, 3.0])),
        rho=tuple(heston_raw.get("rho", [-0.99, 0.99])),
    )

    market = MarketConfig(
        risk_free_rate=float(market_raw.get("risk_free_rate", os.getenv("RISK_FREE_RATE", "0.0"))),
        dividend_yield=float(market_raw.get("dividend_yield", os.getenv("DIVIDEND_YIELD", "0.0"))),
        min_open_interest=float(market_raw.get("min_open_interest", 0.01)),
        min_time_to_expiry_days=float(market_raw.get("min_time_to_expiry_days", 1)),
        max_time_to_expiry_years=float(market_raw.get("max_time_to_expiry_years", 1.0)),
        max_relative_spread=float(market_raw.get("max_relative_spread", 0.5)),
        smile_convention=str(market_raw.get("smile_convention", "otm")),
    )

    calibration = CalibrationConfig(
        atm_log_moneyness_sigma=float(calib_raw.get("atm_log_moneyness_sigma", 0.15)),
        use_vega_weight=bool(calib_raw.get("use_vega_weight", True)),
        use_liquidity_weight=bool(calib_raw.get("use_liquidity_weight", True)),
        min_strikes_per_slice=int(calib_raw.get("min_strikes_per_slice", 5)),
        optimizer=str(calib_raw.get("optimizer", "L-BFGS-B")),
        feller_penalty=float(calib_raw.get("feller_penalty", 1000.0)),
    )

    return AppConfig(
        snapshot_date=raw.get("snapshot_date"),
        postgres=PostgresConfig(),
        market=market,
        calibration=calibration,
        heston_bounds=bounds,
        figures_dir=PROJECT_ROOT / outputs_raw.get("figures_dir", "outputs/figures"),
        reports_dir=PROJECT_ROOT / outputs_raw.get("reports_dir", "outputs/reports"),
    )
