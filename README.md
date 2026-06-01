# btc-vol-research

Recherche quantitative sur la **volatilité implicite** des options BTC (Deribit), à partir de la base **Neon** alimentée par le repo [Projet_Option_BTC](https://github.com/Gauthierbenoist/Projet_Option_BTC).

## Objectifs

- Extraction / contrôle des **volatilités implicites** (inversion BS sur mid + comparaison `mark_iv`)
- Construction et analyse des **smiles** et **surfaces** IV
- Effets de **moneyness** et **maturité**
- Modèle **Heston** : prix analytiques **QuantLib**, **calibration pondérée** (vega × liquidité × surpoids ATM)
- Comparaison marché vs modèle (RMSE IV)

## Prérequis

- Python 3.11+
- Table `btc_options` sur Neon (pipeline ETL phase 1)
- Fichier `.env` avec `DATABASE_URL` (copier depuis `.env.example`)

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
cp .env.example .env     # puis éditer DATABASE_URL
```

## Utilisation

```bash
# Dates disponibles
python scripts/run_smile_analysis.py --list-dates

# Smiles + surface + rapport
python scripts/run_smile_analysis.py
python scripts/run_smile_analysis.py --date 2026-06-01

# Calibration Heston (6 maturités les plus denses par défaut)
python scripts/run_heston_calibration.py
python scripts/run_heston_calibration.py --date 2026-06-01 --max-slices 4
```

**Sorties** : `outputs/figures/` (PNG), `outputs/reports/` (CSV).

## Structure

```
src/btc_vol_research/
  data/          # Neon → panel marché
  iv/            # Black-Scholes, conventions OTM
  surfaces/      # smiles, surface 3D, plots
  models/heston/ # pricer, calibration pondérée
  analysis/      # métriques et rapports
scripts/         # CLI
configs/         # paramètres par défaut
```

## Calibration (défauts)

| Élément | Choix |
|---------|--------|
| Smile | Options **OTM** |
| IV marché | `iv_mid` (inversion BS), sinon `mark_iv` |
| Poids | `vega × √OI × exp(-½(ln(K/F)/σ_ATM)²)` |
| Objectif | `Σ w_i (σ_heston - σ_mkt)²` |
| `r`, `q` | 0 (BTC) |

Ajustable dans `configs/default.yaml` et `.env`.

## Tests

```bash
pip install pytest
pytest tests/ -q
```

## Lien phase 1

Ce repo **lit uniquement** Neon ; il ne télécharge pas Deribit.  
Pipeline données : `Projet_Option_BTC`.
