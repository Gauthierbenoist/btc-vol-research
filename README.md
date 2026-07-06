# btc-vol-research

Recherche quantitative sur la **volatilité implicite** des options BTC (Deribit), à partir de la base **Neon** alimentée par le repo [Projet_Option_BTC](https://github.com/Gauthierbenoist/Projet_Option_BTC).

Trois modèles de smile/surface calibrés sur les mêmes données marché : **SVI** (baseline), **Heston** (volatilité stochastique) et **Merton** (jump-diffusion, calibration globale).

## Prérequis

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .              # ou : pip install -r requirements.txt
cp .env.example .env          # renseigner DATABASE_URL (Neon)
```

## Utilisation

```bash
# Baseline SVI (rapide, à lancer en premier)
python scripts/run_svi_calibration.py --date 2026-06-01

# Merton — jump-diffusion, calibration globale sur toute la surface
python scripts/run_merton_calibration.py --date 2026-06-01 --weight-scheme vega

# Heston — volatilité stochastique (QuantLib, plus lourd)
python scripts/run_heston_calibration.py --date 2026-06-01
```

Sans `--date`, le snapshot le plus récent de Neon est utilisé (ou `snapshot_date` de `configs/default.yaml`).

**Sorties** : `outputs/figures/` (fits PNG, surfaces 3D Plotly HTML) et `outputs/reports/` (CSV de calibration et de métriques).

## Modèles

### SVI (baseline, Gatheral)

Variance totale : `w(k) = a + b ( ρ(k−m) + √((k−m)² + σ²) )` avec `k = ln(K/F)`, `σ_IV = √(w/T)`.

- Calibration **par maturité**, contrainte **no-butterfly** (condition suffisante Gatheral)
- **Surface 3D + contour** : interpolation de `w(k)` entre tenors
- **Structure par terme du skew** : `ρ(T)`
- Génère aussi les figures avec la **pondération v2** (`svi_v2_*`)

### Merton (jump-diffusion, 1976)

Sauts log-normaux, un **seul jeu de paramètres** `(σ, λ, μ_J, σ_J)` sur toute la surface (calibration globale). Prix par série de Poisson × Black-Scholes. `--weight-scheme` : `uniform | vega | volume`.

### Heston (volatilité stochastique)

Pricing analytique via **QuantLib**, calibration **par maturité**, contrainte de **Feller** pénalisée.

## Structure

```
src/btc_vol_research/
  config.py            # config (YAML + env), dataclasses figées
  data/                # loader (Neon), panel marché, filtres qualité + OTM
  market/              # maths marché : forward, implied_vol (inversion BS), greeks
  models/              # pricing pur : black_scholes, svi, merton, heston
  calibration/         # errors, filters, weights, results, slices + svi/heston/merton
  surfaces/            # svi_surface, merton_surface, export CSV, plots
  analysis/            # tables récap, diagnostics Merton, rapports CSV
scripts/
  run_svi_calibration.py
  run_merton_calibration.py
  run_heston_calibration.py
```

Chaque couche a une responsabilité unique : `data` prépare, `market`/`models` calculent, `calibration` optimise, `surfaces`/`analysis` restituent.

## Choix de calibration (défauts)

| Élément | Choix |
|---------|-------|
| Smile | Options **OTM** systématiquement (calls K>F, puts K<F) — liquidité |
| IV source | `mark_iv` Deribit (`iv_used`) |
| Objectif | `Σ wᵢ (σ_model − σ_mkt)²` (fonction de coût `sse_objective`) |
| Poids v1 (défaut SVI/Heston) | `vega × √OI` — `calibration_weights()` |
| Poids v2 | `vega × √(1+OI) × √(1+volume)` — `calibration_weights_v2()` |
| Filtres qualité | bornes IV/maturité/spread depuis la config (source unique) |

Tous les paramètres (bornes des modèles, filtres marché, poids) sont dans `configs/default.yaml`.

## Tests

```bash
pytest tests/ -q
```

> `test_heston.py` nécessite **QuantLib** ; les autres tests s'exécutent sans cette dépendance.
