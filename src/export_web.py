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


def export_kaggle():
    """Export web data from Kaggle forecast outputs — becomes primary data."""
    kf = pd.read_csv(os.path.join(REPORTS, "kaggle_forecast_hotel.csv"),
                     parse_dates=["date"])
    # shift dates forward 9 years (2017 → 2026) for a current-looking demo
    kf["date"] = kf["date"].apply(lambda d: d.replace(year=d.year + 9))
    plan = pd.read_csv(os.path.join(REPORTS, "kaggle_planning_summary.csv"))
    imp = pd.read_csv(os.path.join(REPORTS, "kaggle_importance.csv"))
    meta = json.load(open(os.path.join(REPORTS, "kaggle_run_meta.json")))
    # shift meta dates by 9 years
    meta["origin"] = str(pd.Timestamp(meta["origin"]) + pd.DateOffset(years=9)).split(" ")[0]
    meta["forecast_end"] = str(pd.Timestamp(meta["forecast_end"]) + pd.DateOffset(years=9)).split(" ")[0]
    metrics = json.load(open(os.path.join(REPORTS, "kaggle_fwd_metrics.json")))
    occ = pd.read_csv(os.path.join(DATA, "kaggle_occupancy.csv"),
                      parse_dates=["date"])

    # ---- meta.json ----
    s = metrics["summary"]
    order = ["LightGBM", "Seasonal-naive (hotel \u00d7 month)", "Hotel-average",
             "Global-average"]
    baselines = [{
        "model": k,
        "mae_mw": round(s[k]["hotel_week_MAE"], 4),
        "mae_lm": round(s[k]["MAE"], 4),
        "is_model": k == "LightGBM",
    } for k in order if k in s]
    w("meta.json", {
        **meta,
        "source": "kaggle",
        "headline": {
            "improvement_pct": round(metrics["headline"]["improvement_pct"], 1),
            "model_mae": round(metrics["headline"]["model_hotel_week_MAE"], 4),
            "seasonal_mae": round(
                metrics["headline"]["seasonal_naive_hotel_week_MAE"], 4),
        },
        "n_folds": metrics["n_folds"],
        "baselines": baselines,
    })

    # ---- per-hotel daily forecast (markets.json) ----
    dates = sorted(kf["date"].unique())
    date_str = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates]
    adrs = occ.groupby("hotel").avg_adr.mean()
    markets = []
    for h, g in kf.sort_values("date").groupby("hotel"):
        g = g.set_index("date").reindex(dates)
        markets.append({
            "market": h,
            "avg": round(float(g.occ_forecast.mean()), 3),
            "occ": [round(float(x), 3) for x in g.occ_forecast.values],
            "lower": [round(float(x), 3) for x in g.occ_lower.values],
            "upper": [round(float(x), 3) for x in g.occ_upper.values],
            "avg_adr": round(float(adrs[h]), 2),
        })
    markets.sort(key=lambda m: -m["avg"])
    w("markets.json", {"dates": date_str, "markets": markets, "has_intervals": True})

    # ---- per-hotel planning summary (listings.json) ----
    lj = []
    for _, r in plan.iterrows():
        lj.append({
            "listing_id": r["hotel"],
            "market": r["hotel"],
            "avg_occ_90d": round(float(r["avg_occ_90d"]), 3),
            "peak_week": r["peak_week"],
            "peak_occ": round(float(r["peak_occ"]), 3),
            "low_week": r["low_week"],
            "low_occ": round(float(r["low_occ"]), 3),
            "recommendation": r["recommendation"],
        })
    w("listings.json", lj)

    # ---- per-hotel forecast + trailing history (listing_series.json) ----
    fc = kf.pivot_table(index="hotel", columns="date", values="occ_forecast")
    fc = fc.reindex(columns=dates)
    series = {str(h): [round(float(x), 3) for x in row]
              for h, row in zip(fc.index, fc.values)}

    fc_lower = kf.pivot_table(index="hotel", columns="date", values="occ_lower")
    fc_lower = fc_lower.reindex(columns=dates)
    fc_upper = kf.pivot_table(index="hotel", columns="date", values="occ_upper")
    fc_upper = fc_upper.reindex(columns=dates)
    series_lower = {str(h): [round(float(x), 3) for x in row]
                    for h, row in zip(fc_lower.index, fc_lower.values)}
    series_upper = {str(h): [round(float(x), 3) for x in row]
                    for h, row in zip(fc_upper.index, fc_upper.values)}

    occ["week"] = occ["date"].dt.to_period("W").apply(lambda p: p.start_time)
    hw = occ.groupby(["hotel", "week"]).occupancy_rate.mean().reset_index()
    last_weeks = sorted(hw["week"].unique())[-16:]
    hist_weeks = [
        (pd.Timestamp(x) + pd.DateOffset(years=9)).strftime("%Y-%m-%d")
        for x in last_weeks
    ]
    hp = hw.pivot_table(index="hotel", columns="week", values="occupancy_rate")
    hp = hp.reindex(columns=last_weeks)
    history = {str(h): [None if pd.isna(x) else round(float(x), 3) for x in row]
               for h, row in zip(hp.index, hp.values)}
    w("listing_series.json", {
        "forecast_dates": date_str, "forecast": series,
        "forecast_lower": series_lower, "forecast_upper": series_upper,
        "history_weeks": hist_weeks, "history": history,
    })

    # ---- demand drivers ----
    imp2 = imp.sort_values("gain", ascending=False).head(12)
    tot = imp2.gain.sum()
    w("importance.json", [{"name": r["feature"],
                           "pct": round(float(r.gain / tot * 100), 1)}
                          for _, r in imp2.iterrows()])

    # ---- seasonality (per-hotel monthly avg) ----
    occ["month"] = occ["date"].dt.month
    season = occ.groupby(["hotel", "month"]).occupancy_rate.mean().reset_index()
    out = {}
    for h, g in season.groupby("hotel"):
        g = g.set_index("month").reindex(range(1, 13))
        out[h] = [round(float(x), 3) for x in g.occupancy_rate.values]
    w("season.json", out)

    # ---- back-test detail ----
    bt = os.path.join(REPORTS, "kaggle_fwd_backtest.json")
    if os.path.exists(bt):
        w("backtest.json", json.load(open(bt)))

    # ---- real-data (Inside Airbnb) validation, if present ----
    rm_path = os.path.join(REPORTS, "real_metrics.json")
    rb_path = os.path.join(REPORTS, "real_backtest_web.json")
    if os.path.exists(rm_path):
        rm = json.load(open(rm_path))
        s = rm["summary"]
        order = ["LightGBM", "Seasonal-naive (mkt x month)", "Listing-average",
                 "Market-average", "Global-average"]
        real = {
            "source": rm["source"], "n_listings": rm["n_listings"],
            "n_markets": rm["n_markets"], "n_rows": rm["n_rows"],
            "occupancy_rate": rm["occupancy_rate"], "n_folds": rm["n_folds"],
            "headline": {k: round(v, 4) if isinstance(v, float) else v
                         for k, v in rm["headline"].items()},
            "baselines": [{
                "model": k, "auc": round(s[k]["AUC"], 3),
                "mae_mw": round(s[k]["market_week_MAE"], 4),
                "mae_lm": round(s[k]["listing_month_MAE"], 4),
                "is_model": k == "LightGBM",
            } for k in order if k in s],
        }
        if os.path.exists(rb_path):
            real["detail"] = json.load(open(rb_path))
        w("real.json", real)

    # ---- Kaggle Hotel Booking Demand validation (lag-feature back-test) ----
    km_path = os.path.join(REPORTS, "kaggle_metrics.json")
    kb_path = os.path.join(REPORTS, "kaggle_backtest_web.json")
    if os.path.exists(km_path):
        km = json.load(open(km_path))
        s = km["summary"]
        order = ["LightGBM", "Seasonal-naive (hotel \u00d7 month)", "Hotel-average",
                 "Global-average"]
        kaggle = {
            "source": km["source"],
            "source_url": km.get("source_url", ""),
            "n_hotels": km["n_hotels"],
            "n_rows": km["n_rows"],
            "n_bookings": km.get("n_bookings", 0),
            "mean_occupancy": km["mean_occupancy"],
            "horizon_days": km.get("horizon_days", 90),
            "n_folds": km["n_folds"],
            "headline": {k: round(v, 4) if isinstance(v, float) else v
                         for k, v in km["headline"].items()},
            "baselines": [{
                "model": k,
                "mae_wk": round(s[k]["hotel_week_MAE"], 4),
                "mae_daily": round(s[k]["MAE"], 4),
                "is_model": k == "LightGBM",
            } for k in order if k in s],
            "importance": km.get("importance", {}),
        }
        if os.path.exists(kb_path):
            kaggle["detail"] = json.load(open(kb_path))
        w("kaggle.json", kaggle)

    # ---- East African synthetic data (same pipeline, different dataset) ----
    ea_metrics = os.path.join(REPORTS, "metrics.json")
    ea_backtest = os.path.join(REPORTS, "backtest_web.json")
    ea_meta = os.path.join(REPORTS, "run_meta.json")
    ea_imp = os.path.join(REPORTS, "feature_importance.csv")
    if os.path.exists(ea_metrics):
        em = json.load(open(ea_metrics))
        rm = json.load(open(ea_meta)) if os.path.exists(ea_meta) else {}
        s = em["summary"]
        order = ["LightGBM", "Seasonal-naive (mkt x month)", "Listing-average",
                 "Market-average", "Global-average"]
        imp_df = pd.read_csv(ea_imp) if os.path.exists(ea_imp) else None
        imp_list = []
        if imp_df is not None:
            imp2 = imp_df.copy()
            imp2["name"] = imp2.feature.map(NICE).fillna(imp2.feature)
            imp2 = imp2.groupby("name", as_index=False).gain.sum()
            imp2 = imp2.sort_values("gain", ascending=False).head(8)
            tot = imp2.gain.sum()
            imp_list = [{"name": r["name"], "pct": round(float(r.gain / tot * 100), 1)}
                        for _, r in imp2.iterrows()]
        east_africa = {
            "source": "Synthetic East African STR Data",
            "n_listings": rm.get("n_listings", 500),
            "n_markets": rm.get("n_markets", 10),
            "n_rows": rm.get("n_listings", 500) * 365,
            "occupancy_rate": rm.get("portfolio_avg_occ_90d", 0.657),
            "horizon_days": em.get("horizon_days", 90),
            "n_folds": em.get("n_folds", 3),
            "headline": {
                "improvement_pct": round(em["headline"]["improvement_pct"], 1),
                "model_market_week_MAE": round(em["headline"]["model_market_week_MAE"], 4),
                "model_AUC": round(em["headline"]["model_AUC"], 3),
            },
            "baselines": [{
                "model": k, "auc": round(s[k]["AUC"], 3),
                "mae_mw": round(s[k]["market_week_MAE"], 4),
                "mae_lm": round(s[k]["listing_month_MAE"], 4),
                "is_model": k == "LightGBM",
            } for k in order if k in s],
            "importance": imp_list,
        }
        if os.path.exists(ea_backtest):
            east_africa["detail"] = json.load(open(ea_backtest))
        w("east_africa.json", east_africa)

    print("Exported web data from Kaggle pipeline to web/public/data/")


def export_synthetic():
    """Legacy export from the synthetic East African STR pipeline."""
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

    # ---- real-data (Inside Airbnb Cape Town) validation, if present ----
    rm_path = os.path.join(REPORTS, "real_metrics.json")
    rb_path = os.path.join(REPORTS, "real_backtest_web.json")
    if os.path.exists(rm_path):
        rm = json.load(open(rm_path))
        s = rm["summary"]
        order = ["LightGBM", "Seasonal-naive (mkt x month)", "Listing-average",
                 "Market-average", "Global-average"]
        real = {
            "source": rm["source"], "n_listings": rm["n_listings"],
            "n_markets": rm["n_markets"], "n_rows": rm["n_rows"],
            "occupancy_rate": rm["occupancy_rate"], "n_folds": rm["n_folds"],
            "headline": {k: round(v, 4) if isinstance(v, float) else v
                         for k, v in rm["headline"].items()},
            "baselines": [{
                "model": k, "auc": round(s[k]["AUC"], 3),
                "mae_mw": round(s[k]["market_week_MAE"], 4),
                "mae_lm": round(s[k]["listing_month_MAE"], 4),
                "is_model": k == "LightGBM",
            } for k in order if k in s],
        }
        if os.path.exists(rb_path):
            real["detail"] = json.load(open(rb_path))
        w("real.json", real)

    # ---- Kaggle Hotel Booking Demand validation, if present ----
    km_path = os.path.join(REPORTS, "kaggle_metrics.json")
    kb_path = os.path.join(REPORTS, "kaggle_backtest_web.json")
    if os.path.exists(km_path):
        km = json.load(open(km_path))
        s = km["summary"]
        order = ["LightGBM", "Seasonal-naive (hotel \u00d7 month)", "Hotel-average",
                 "Global-average"]
        kaggle = {
            "source": km["source"],
            "source_url": km.get("source_url", ""),
            "n_hotels": km["n_hotels"],
            "n_rows": km["n_rows"],
            "n_bookings": km.get("n_bookings", 0),
            "mean_occupancy": km["mean_occupancy"],
            "horizon_days": km.get("horizon_days", 90),
            "n_folds": km["n_folds"],
            "headline": {k: round(v, 4) if isinstance(v, float) else v
                         for k, v in km["headline"].items()},
            "baselines": [{
                "model": k,
                "mae_wk": round(s[k]["hotel_week_MAE"], 4),
                "mae_daily": round(s[k]["MAE"], 4),
                "is_model": k == "LightGBM",
            } for k in order if k in s],
            "importance": km.get("importance", {}),
        }
        if os.path.exists(kb_path):
            kaggle["detail"] = json.load(open(kb_path))
        w("kaggle.json", kaggle)

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


def main():
    kaggle_fc = os.path.join(REPORTS, "kaggle_forecast_hotel.csv")
    if os.path.exists(kaggle_fc):
        print("Kaggle forecast found — using real hotel data as primary source.\n")
        export_kaggle()
    else:
        export_synthetic()


if __name__ == "__main__":
    main()
