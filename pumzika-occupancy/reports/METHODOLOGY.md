# Methodology — Occupancy & Demand Forecasting

This document covers the problem framing, the data-generating process, the
feature design (and how leakage is prevented), the model, the evaluation
protocol, and the limitations. The goal is that a reviewer can audit every
claim.

## 1. Problem framing

The challenge: *forecast occupancy rates so owners can plan ahead.* We model the
atomic event — **will listing _i_ be booked on night _t_?** — as a probability
`p_{i,t}`. Occupancy *rate* over any group (a listing-month, a market-week, the
whole portfolio) is then `mean(p_{i,t})` over that group. This single
formulation answers every level the business cares about, from one listing to a
whole country, with one model.

Horizon: **90 days**, daily resolution. That window is long enough to act on
(re-price, promote, staff) and is the natural planning cadence for hosts.

## 2. Data-generating process (self-sourced)

`generate_data.py` simulates 500 listings across 10 markets in TZ/KE/UG over two
years (2024-06-01 → 2026-06-01), then a 90-day forward window for forecasting.

Each night's booking is drawn from an explicit logistic model:

```
logit(p) =  base_occupancy(market)
          + 1.25·log(seasonal_index)         # archetype-specific annual curve
          +      log(weekday_index)           # leisure vs business weekly shape
          + 0.55·log(holiday_multiplier)      # Christmas / Easter / Eid / events
          +      listing_desirability         # latent per-listing quality
          + 0.45·(review_score − 4.3) + 0.20·superhost + amenities
          + platform_growth(t)                # young-marketplace uptrend
          − 1.30·(price_ratio − 1)            # price elasticity of demand
          + N(0, 0.45)                         # irreducible nightly noise
```

Design choices that make the task realistic — and hard in the right ways:
- **Prices move *with* demand** (peak season lifts both rate and occupancy). A
  naive elasticity reading would be wrong; the model must use season, not price.
- **Archetype-specific seasonality** (safari / beach / city) means no single
  global seasonal curve suffices — empirically, safari occupancy swings ~26
  points across the year, beach ~18, city ~4.
- **A genuine noise floor** (`N(0,0.45)`) caps achievable night-level AUC — as in
  reality, individual nights are partly random.

Because the DGP is known, we can verify the model recovers true structure.

## 3. Features — and the leakage guard

All features must be knowable at the **forecast origin**. Three families:

| Family | Examples | Known in advance? |
|---|---|---|
| Calendar / season | month, day-of-week, cyclical day-of-year, weekend | ✔ deterministic from the date |
| Holiday / event | is_holiday, holiday_multiplier, signed days-to-holiday | ✔ fixed calendar |
| Static listing | type, capacity, review_score, superhost, amenities, host tenure, lat/lon | ✔ |
| **Learned levels** | listing / market / market×month / archetype×weekday mean occupancy | ✔ **estimated only from pre-origin data** |

**The leakage guard.** "Learned level" features are empirical averages that could
trivially leak the future. They are fit with `features.fit_levels()` **only on
the training slice that precedes the test window**, and re-fit independently in
every back-test fold. No future night ever informs a level used to predict it.

**Realised future price is intentionally excluded** — it's a decision, not a
known input, and including it would both leak the outcome and make the forecast
unusable for planning (you'd need tomorrow's prices to forecast tomorrow). The
result is a clean *demand* forecast under expected pricing.

## 4. Model

A single global **LightGBM** gradient-boosted classifier
(`binary` objective, 600 trees, `lr=0.05`, `num_leaves=64`,
`min_child_samples=200`, feature/bagging fraction 0.8). Native categorical
handling for market / country / archetype / property_type. One global model
pools signal across all listings — far more stable than per-listing models,
especially for new or thin listings (cold-start falls back gracefully to
market × season levels).

## 5. Evaluation protocol

**Rolling-origin back-test, 3 folds.** Origins at T−270, T−180, T−90 days; each
fold trains on everything before its window and forecasts the next 90 days. This
mirrors production: always predict forward from a fixed "today".

Two metric axes:
- **Night-level discrimination** — ROC-AUC, log-loss, Brier.
- **Occupancy-rate accuracy** — MAE / RMSE of predicted vs actual occupancy
  *rate* at **market-week** and **listing-month** aggregation. This is the
  business metric.

**Baselines** (each encodes one piece of domain knowledge):
- Global average · Market average · Listing average · **Seasonal-naive
  (market × calendar-month)** — the last is strong and the one to beat.

### Headline results (mean over 3 folds)

| Model | AUC | MAE (market-week) | MAE (listing-month) |
|---|---|---|---|
| **LightGBM** | **0.643** | **0.0334** | **0.1309** |
| Seasonal-naive | 0.560 | 0.0415 | 0.1657 |
| Listing-average | 0.635 | 0.0560 | 0.1361 |
| Market-average | 0.542 | 0.0560 | 0.1691 |
| Global-average | 0.500 | 0.0625 | 0.1720 |

→ **19.6%** lower market-week error than the strongest baseline, winning every
column. Calibration (`figures/calibration.png`) is near-diagonal, so the
probabilities can be trusted as rates.

### Learned drivers
Top gain features: a listing's **own track record**, **market × month
seasonality**, the **annual season cycle**, **review score** and **host tenure**
— sensible, defensible, and matching the DGP.

## 6. Limitations & honest caveats

- **Synthetic data.** Real bookings carry messier effects (channel mix, lead-time
  curves, event shocks, supply growth). The architecture is built to ingest real
  Pumzika exports unchanged; numbers would be re-validated on real data.
- **Night-level AUC is modest by construction** — the DGP has a real noise floor.
  The *rate* forecast, which is the deliverable, is strong and calibrated.
- **No prediction intervals yet.** A natural next step is quantile LightGBM or
  conformal intervals to show forecast uncertainty per week.
- **Price held at "expected".** Joint demand-and-price optimisation is the
  hand-off to Track 01 (Dynamic Pricing).

## 7. Reproducibility
Fully seeded (`SEED = 42`). `./run.sh` regenerates data, model, forecasts and
figures from scratch in under a minute on a laptop.
