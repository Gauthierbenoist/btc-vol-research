"""Schémas de pondération Merton (extensible)."""

from __future__ import annotations

from dataclasses import dataclass

from btc_vol_research.models.calibration_weights import WeightFn, vega_only_weights, volume_only_weights


@dataclass(frozen=True)
class MertonWeightScheme:
    scheme_id: str
    label: str
    weight_fn: WeightFn | None


MERTON_WEIGHT_SCHEMES: tuple[MertonWeightScheme, ...] = (
    MertonWeightScheme("uniform", "sans ponderation", None),
    MertonWeightScheme("vega", "vega", vega_only_weights),
    MertonWeightScheme("volume", "volume", volume_only_weights),
)

_SCHEME_BY_ID = {s.scheme_id: s for s in MERTON_WEIGHT_SCHEMES}


def get_merton_weight_scheme(scheme_id: str) -> MertonWeightScheme:
    key = scheme_id.strip().lower()
    if key not in _SCHEME_BY_ID:
        known = ", ".join(_SCHEME_BY_ID)
        raise ValueError(f"Ponderation Merton inconnue: {scheme_id!r} (attendu: {known})")
    return _SCHEME_BY_ID[key]
