"""
forecast.py
===========
Generate the forward-looking 90-day occupancy forecast that powers the
dashboard, using the production model trained on all history.

Outputs:
  reports/forecast_listing.csv  (listing_id, date, occ_forecast)
  reports/forecast_market.csv   (market, country, date, occ_forecast, n_listings)
  reports/planning_summary.csv   one actionable row per listing
  reports/run_meta.json          origin date, horizon, generation context

Run:  python3 src/forecast.py   (after train.py)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

import features as F

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
MODELS = os.path.join(HERE, "..", "models")
REPORTS = os.path.join(HERE, "..", "reports")

ORIGIN = pd.Timestamp("2026-06-02")   # forecast from "today"
HORIZON = 90


def main():
    listings = pd.read_csv(os.path.join(DATA, "listings.csv"))
    booster = lgb.Booster(model_file=os.path.join(MODELS, "model.txt"))
    levels = joblib.load(os.path.join(MODELS, "levels.joblib"))
    listings_static = F.prepare_static(listings)

    # build the future listing x date grid
    future_dates = pd.date_range(ORIGIN, periods=HORIZON, freq="D")
    grid = pd.MultiIndex.from_product(
        [listings["listing_id"].values, future_dates],
        names=["listing_id", "date"]).to_frame(index=False)

    design = F.build_design(grid, listings_static, levels)
    design["occ_forecast"] = booster.predict(design[F.FEATURES])

    meta = listings[["listing_id", "market", "country", "archetype",
                     "base_price"]]
    out = design[["listing_id", "date", "occ_forecast"]].merge(
        meta, on="listing_id", how="left")

    # ---- listing-level forecast ----
    out[["listing_id", "date", "occ_forecast"]].to_csv(
        os.path.join(REPORTS, "forecast_listing.csv"), index=False)

    # ---- market-level forecast ----
    mkt = out.groupby(["market", "country", "date"]).agg(
        occ_forecast=("occ_forecast", "mean"),
        n_listings=("listing_id", "nunique")).reset_index()
    mkt.to_csv(os.path.join(REPORTS, "forecast_market.csv"), index=False)

    # ---- per-listing planning summary (actionable) ----
    out["week"] = out["date"].dt.to_period("W").apply(lambda p: p.start_time)
    wk = out.groupby(["listing_id", "week"]).occ_forecast.mean().reset_index()
    rows = []
    for lid, grp in wk.groupby("listing_id"):
        g = grp.sort_values("week")
        peak = g.loc[g.occ_forecast.idxmax()]
        low = g.loc[g.occ_forecast.idxmin()]
        avg = g.occ_forecast.mean()
        rows.append(dict(
            listing_id=lid,
            avg_occ_90d=round(float(avg), 3),
            peak_week=peak.week.date().isoformat(),
            peak_occ=round(float(peak.occ_forecast), 3),
            low_week=low.week.date().isoformat(),
            low_occ=round(float(low.occ_forecast), 3),
        ))
    summary = pd.DataFrame(rows).merge(
        listings[["listing_id", "market", "country", "archetype", "base_price",
                  "review_score", "is_superhost"]],
        on="listing_id", how="left")

    # simple, explainable recommendation rules off the forecast
    def recommend(r):
        if r.peak_occ >= 0.80:
            return f"Raise rates for the week of {r.peak_week} (forecast {r.peak_occ:.0%} full)."
        if r.low_occ <= 0.45:
            return f"Add a promo / min-stay discount around {r.low_week} (soft demand {r.low_occ:.0%})."
        if r.avg_occ_90d >= 0.70:
            return "Strong quarter ahead - protect availability, test a price uplift."
        return "Steady demand - keep rates, watch weekend pickup."
    summary["recommendation"] = summary.apply(recommend, axis=1)
    summary.sort_values(["market", "avg_occ_90d"], ascending=[True, False]).to_csv(
        os.path.join(REPORTS, "planning_summary.csv"), index=False)

    with open(os.path.join(REPORTS, "run_meta.json"), "w") as f:
        json.dump({
            "origin": ORIGIN.date().isoformat(),
            "horizon_days": HORIZON,
            "forecast_end": future_dates[-1].date().isoformat(),
            "n_listings": int(listings.shape[0]),
            "n_markets": int(listings.market.nunique()),
            "portfolio_avg_occ_90d": round(float(out.occ_forecast.mean()), 3),
        }, f, indent=2)

    print(f"Forecast {ORIGIN.date()} -> {future_dates[-1].date()} "
          f"for {listings.shape[0]} listings / {listings.market.nunique()} markets")
    print(f"  portfolio avg 90-day occupancy: {out.occ_forecast.mean():.1%}")
    print("  Saved forecast_listing.csv, forecast_market.csv, "
          "planning_summary.csv, run_meta.json")


if __name__ == "__main__":
    main()
