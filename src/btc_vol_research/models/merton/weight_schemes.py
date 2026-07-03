"""Schémas de pondération Merton (extensible)."""

from __future__ import annotations

from dataclasses import dataclass

from btc_vol_research.models.calibration_weights import WeightFn, vega_only_weights


@dataclass(frozen=True)
class MertonWeightScheme:
    scheme_id: str
    label: str
    weight_fn: WeightFn | None


MERTON_WEIGHT_SCHEMES: tuple[MertonWeightScheme, ...] = (
    MertonWeightScheme("uniform", "sans ponderation", None),
    MertonWeightScheme("vega", "vega", vega_only_weights),
)
