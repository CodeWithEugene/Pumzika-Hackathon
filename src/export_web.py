"""
export_web.py
=============
Serialise the model outputs into compact JSON that the static Next.js site
(deployed on Vercel) reads at runtime. Run after train.py + forecast.py.

Writes to web/public/data/:
  meta.json            headline metrics, baseline table, run context
  markets.json         per-market daily forecast (for line chart + heatmap)
  listings.json        per-listing planning summary (table + planner picker)
  listing_series.json  per-listing 90-day forecast + trailing weekly history
  importance.json      top demand drivers
  season.json          learned seasonality by archetype x month
"""

import os
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
DATA = os.path.join(ROOT, "data")
REPORTS = os.path.join(ROOT, "reports")
OUT = os.path.join(ROOT, "web", "public", "data")
os.makedirs(OUT, exist_ok=True)

NICE = {
    "listing_baseline_occ": "Listing's own track record",
    "market_month_occ": "Market seasonality (month)",
    "archetype_dow_occ": "Day-of-week pattern",
    "doy_sin": "Season (annual cycle)", "doy_cos": "Season (annual cycle)",
    "dow_sin": "Day-of-week pattern", "review_score": "Review score",
    "num_reviews": "Number of reviews", "host_tenure_days": "Host tenure",
    "dayofyear": "Time of year", "base_price": "Price tier",
    "is_weekend": "Weekend", "holiday_mult": "Holiday / event",
    "days_to_holiday": "Days to holiday", "capacity": "Capacity",
    "bedrooms": "Bedrooms", "market_baseline_occ": "Market baseline",
    "day": "Day of month", "lat": "Latitude", "lon": "Longitude",
    "month": "Month",
}


def w(name, obj):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, separators=(",", ":"))
    kb = os.path.getsize(os.path.join(OUT, name)) / 1024
    print(f"  {name:22s} {kb:7.1f} KB")


def main():
    listings = pd.read_csv(os.path.join(DATA, "listings.csv"))
    fc_m = pd.read_csv(os.path.join(REPORTS, "forecast_market.csv"),
                       parse_dates=["date"])
    fc_l = pd.read_csv(os.path.join(REPORTS, "forecast_listing.csv"),
                       parse_dates=["date"])
    plan = pd.read_csv(os.path.join(REPORTS, "planning_summary.csv"))
    metrics = json.load(open(os.path.join(REPORTS, "metrics.json")))
    meta = json.load(open(os.path.join(REPORTS, "run_meta.json")))
    imp = pd.read_csv(os.path.join(REPORTS, "feature_importance.csv"))
    hist = pd.read_csv(os.path.join(DATA, "calendar.csv"), parse_dates=["date"],
                       usecols=["listing_id", "date", "booked"])

    # ---- meta + baseline table ----
    s = metrics["summary"]
    order = ["LightGBM", "Seasonal-naive (mkt x month)", "Listing-average",
             "Market-average", "Global-average"]
    baselines = [{
        "model": k, "auc": round(s[k]["AUC"], 3),
        "mae_mw": round(s[k]["market_week_MAE"], 4),
        "mae_lm": round(s[k]["listing_month_MAE"], 4),
        "is_model": k == "LightGBM",
    } for k in order if k in s]
    w("meta.json", {
        **meta,
        "headline": {
            "improvement_pct": round(metrics["headline"]["improvement_pct"], 1),
            "model_mae": round(metrics["headline"]["model_market_week_MAE"], 4),
            "seasonal_mae": round(
                metrics["headline"]["seasonal_naive_market_week_MAE"], 4),
            "auc": round(metrics["headline"]["model_AUC"], 3),
        },
        "n_folds": metrics["n_folds"],
        "baselines": baselines,
    })

    # ---- per-market daily forecast ----
    dates = sorted(fc_m["date"].unique())
    date_str = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates]
    markets = []
    for mkt, g in fc_m.sort_values("date").groupby("market"):
        g = g.set_index("date").reindex(dates)
        markets.append({
            "market": mkt,
            "country": listings[listings.market == mkt].country.iloc[0],
            "archetype": listings[listings.market == mkt].archetype.iloc[0],
            "avg": round(float(g.occ_forecast.mean()), 3),
            "occ": [round(float(x), 3) for x in g.occ_forecast.values],
        })
    markets.sort(key=lambda m: -m["avg"])
    w("markets.json", {"dates": date_str, "markets": markets})

    # ---- per-listing planning summary ----
    cols = ["listing_id", "market", "country", "archetype", "base_price",
            "review_score", "is_superhost", "avg_occ_90d", "peak_week",
            "peak_occ", "low_week", "low_occ", "recommendation"]
    lj = plan[cols].copy()
    lj["base_price"] = lj["base_price"].round(0).astype(int)
    lj["is_superhost"] = lj["is_superhost"].astype(bool)
    nrev = listings.set_index("listing_id").num_reviews
    lj["num_reviews"] = lj.listing_id.map(nrev).astype(int)
    w("listings.json", lj.to_dict(orient="records"))

    # ---- per-listing 90-day forecast (shared date axis) ----
    fl = fc_l.pivot_table(index="listing_id", columns="date",
                          values="occ_forecast")
    fl = fl.reindex(columns=dates)
    series = {int(lid): [round(float(x), 3) for x in row]
              for lid, row in zip(fl.index, fl.values)}

    # ---- trailing weekly history per listing (last 16 weeks) ----
    hist["week"] = hist["date"].dt.to_period("W").apply(lambda p: p.start_time)
    hw = hist.groupby(["listing_id", "week"]).booked.mean().reset_index()
    last_weeks = sorted(hw["week"].unique())[-16:]
    hw = hw[hw["week"].isin(last_weeks)]
    hist_weeks = [pd.Timestamp(x).strftime("%Y-%m-%d") for x in last_weeks]
    hp = hw.pivot_table(index="listing_id", columns="week", values="booked")
    hp = hp.reindex(columns=last_weeks)
    history = {int(lid): [None if pd.isna(x) else round(float(x), 3)
                          for x in row]
               for lid, row in zip(hp.index, hp.values)}
    w("listing_series.json", {
        "forecast_dates": date_str, "forecast": series,
        "history_weeks": hist_weeks, "history": history,
    })

    # ---- demand drivers (grouped + nice names) ----
    imp2 = imp.copy()
    imp2["name"] = imp2.feature.map(NICE).fillna(imp2.feature)
    imp2 = imp2.groupby("name", as_index=False).gain.sum()
    imp2 = imp2.sort_values("gain", ascending=False).head(12)
    tot = imp2.gain.sum()
    w("importance.json", [{"name": r["name"],
                           "pct": round(float(r.gain / tot * 100), 1)}
                          for _, r in imp2.iterrows()])

    # ---- back-test detail for native (theme-aware) trust charts ----
    bt_path = os.path.join(REPORTS, "backtest_web.json")
    if os.path.exists(bt_path):
        w("backtest.json", json.load(open(bt_path)))

    # ---- learned seasonality by archetype x month ----
    hm = hist.merge(listings[["listing_id", "archetype"]], on="listing_id")
    hm["month"] = hm["date"].dt.month
    season = hm.groupby(["archetype", "month"]).booked.mean().reset_index()
    out = {}
    for a, g in season.groupby("archetype"):
        g = g.set_index("month").reindex(range(1, 13))
        out[a] = [round(float(x), 3) for x in g.booked.values]
    w("season.json", out)

    print("Exported web data to web/public/data/")


if __name__ == "__main__":
    main()
