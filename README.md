# btc-vol-research

Recherche quantitative sur la **volatilité implicite** des options BTC (Deribit), à partir de la base **Neon** alimentée par le repo [Projet_Option_BTC](https://github.com/Gauthierbenoist/Projet_Option_BTC).

## Pipeline recommandé

1. **Smiles marché** — `run_smile_analysis.py`
2. **Baseline SVI** — `run_svi_calibration.py` (paramétrisation Gatheral, rapide)
3. **Heston** — `run_heston_calibration.py` (modèle stochastique, plus lourd)
4. **Comparaison** — `run_compare_models.py` (RMSE SVI vs Heston)

## Prérequis

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env     # DATABASE_URL Neon
```

## Utilisation

```bash
python scripts/run_smile_analysis.py --list-dates
python scripts/run_smile_analysis.py --date 2026-06-01

# Baseline (à lancer en premier)
python scripts/run_svi_calibration.py --date 2026-06-01
# rho(T) + surface : toutes les maturités éligibles ; --max-slices limite seulement les PNG smile

# Modèle stochastique
python scripts/run_heston_calibration.py --date 2026-06-01 --max-slices 4

# Tableau comparatif
python scripts/run_compare_models.py --date 2026-06-01 --max-slices 4
```

**Sorties** : `outputs/figures/` (`svi_fit_*.png`, `heston_fit_*.png`), `outputs/reports/` (`svi_calibration_*.csv`, …).

## SVI (baseline)

Variance totale :  
`w(k) = a + b ( ρ(k−m) + √((k−m)² + σ²) )`  
avec `k = ln(K/F)`, `σ_IV = √(w/T)`.

- Calibration par maturité, même **pondération** que Heston (vega × √OI × ATM)
- Contrainte **no butterfly** (condition suffisante Gatheral)
- Courbe lisse tracée sur une grille fine de `k`
- **Surface 3D + contour** : interpolation de `w(k)` entre tenors (`svi_surface_3d_*.png`, `svi_surface_contour_*.png`, CSV grille)
- **Structure terme du skew** : `ρ(T)` (`svi_rho_term_*.png`, `svi_rho_term_*.csv`)

## Structure

```
src/btc_vol_research/
  models/svi/       # baseline smile
  models/heston/    # QuantLib + calibration
  models/calibration_weights.py
  surfaces/         # plots
scripts/
  run_svi_calibration.py
  run_heston_calibration.py
  run_compare_models.py
```

## Calibration (défauts communs)

| Élément | Choix |
|---------|--------|
| Smile | Options **OTM** |
| IV | `mark_iv` si mid incohérent, sinon inversion BS |
| Poids | `vega × √OI × [(1−s) + s·exp(−½(k/σ)²)]` avec σ=0,25, s=0,35 |
| Objectif | `Σ w_i (σ_model − σ_mkt)²` |

Paramètres : `configs/default.yaml` (`svi:` et `heston:`).

## Tests

```bash
pytest tests/ -q
```
