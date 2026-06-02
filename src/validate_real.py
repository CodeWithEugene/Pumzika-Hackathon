"""
validate_real.py
================
Validate the SAME occupancy model + leakage-safe feature pipeline on REAL
Inside Airbnb (Cape Town) data — a true reality check that the approach works
beyond the synthetic East-African demo.

Reuses the exact functions from train.py (feature build, model, baselines,
metrics), so the comparison is apples-to-apples with the synthetic back-test.

Run:  python3 src/fetch_real_data.py  &&  python3 src/validate_real.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd

import features as F
import train as TR

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
REPORTS = os.path.join(HERE, "..", "reports")
HORIZON = TR.HORIZON


def load_real():
    listings = pd.read_csv(os.path.join(DATA, "real_listings.csv"))
    cal = pd.read_csv(os.path.join(DATA, "real_calendar.csv"), parse_dates=["date"])
    cal = cal.merge(listings[["listing_id", "market", "archetype"]],
                    on="listing_id", how="left")
    cal["month"] = cal["date"].dt.month
    cal["dayofweek"] = cal["date"].dt.dayofweek
    # forward-availability booking curve: days ahead of the scrape origin
    cal["lead_time"] = (cal["date"] - cal["date"].min()).dt.days
    return listings, cal


def main():
    listings, cal = load_real()
    listings_static = F.prepare_static(listings)
    max_date = cal["date"].max()
    baseline_names = {
        "b_seasonal": "Seasonal-naive (mkt x month)",
        "b_listing": "Listing-average",
        "b_market": "Market-average",
        "b_global": "Global-average",
    }
    fold_records = []
    last = None

    for k, off in enumerate([180, 90, 0]):
        test_end = max_date - pd.Timedelta(days=off)
        test_start = test_end - pd.Timedelta(days=HORIZON - 1)
        tr = cal[cal["date"] < test_start]
        te_cal = cal[(cal["date"] >= test_start) & (cal["date"] <= test_end)]
        if len(tr) < 50000 or len(te_cal) == 0:
            continue
        booster, levels = TR.train_one(tr, listings_static)
        te = TR.predict(booster, levels, te_cal, listings_static)
        te = TR.add_baselines(te, levels)
        y = te["booked"].values
        rec = {"window": f"{test_start.date()} -> {test_end.date()}",
               "LightGBM": {**TR.night_metrics(y, te["pred"].values),
                            **TR.occ_rate_metrics(te, "pred")}}
        for col, nice in baseline_names.items():
            rec[nice] = {**TR.night_metrics(y, te[col].values),
                         **TR.occ_rate_metrics(te, col)}
        fold_records.append(rec)
        last = te
        print(f"  fold {k+1} [{rec['window']}]  AUC={rec['LightGBM']['AUC']:.3f}  "
              f"mktwk-MAE={rec['LightGBM']['market_week_MAE']:.3f}  "
              f"(seasonal-naive {rec['Seasonal-naive (mkt x month)']['market_week_MAE']:.3f})")

    def avg(model, metric):
        return float(np.mean([r[model][metric] for r in fold_records]))

    models = ["LightGBM"] + list(baseline_names.values())
    summary = {m: {k: avg(m, k) for k in
                   ["AUC", "LogLoss", "Brier", "market_week_MAE",
                    "market_week_RMSE", "listing_month_MAE", "listing_month_RMSE"]}
               for m in models}
    base_mae = summary["Seasonal-naive (mkt x month)"]["market_week_MAE"]
    model_mae = summary["LightGBM"]["market_week_MAE"]
    improvement = (base_mae - model_mae) / base_mae * 100
    print(f"\n  REAL DATA — market-week MAE: LightGBM {model_mae:.3f} vs "
          f"seasonal-naive {base_mae:.3f}  => {improvement:.1f}% better")

    metrics = {
        "source": "Inside Airbnb — Cape Town (CC BY 4.0)",
        "n_listings": int(listings.shape[0]),
        "n_markets": int(listings.market.nunique()),
        "n_rows": int(cal.shape[0]),
        "occupancy_rate": round(float(cal.booked.mean()), 3),
        "horizon_days": HORIZON,
        "n_folds": len(fold_records),
        "summary": summary,
        "headline": {"model_market_week_MAE": model_mae,
                     "seasonal_naive_market_week_MAE": base_mae,
                     "improvement_pct": improvement,
                     "model_AUC": summary["LightGBM"]["AUC"]},
    }
    with open(os.path.join(REPORTS, "real_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- web detail: calibration + forecast-vs-actual on real data ----
    te = last.copy()
    bins = np.linspace(0, 1, 11)
    te["bin"] = pd.cut(te["pred"], bins, include_lowest=True)
    cb = te.groupby("bin").agg(p=("pred", "mean"), o=("booked", "mean")).dropna()
    calibration = [{"p": round(float(r.p), 3), "o": round(float(r.o), 3)}
                   for _, r in cb.iterrows()]
    te["week"] = te["date"].dt.to_period("W").apply(lambda p: p.start_time)
    big = te["market"].value_counts().head(4).index.tolist()
    mw = te[te.market.isin(big)].groupby(["market", "week"]).agg(
        actual=("booked", "mean"), pred=("pred", "mean"),
        seasonal=("b_seasonal", "mean")).reset_index()
    fva = {}
    for m in big:
        s = mw[mw.market == m].sort_values("week")
        fva[m] = [{"week": w.strftime("%Y-%m-%d"), "actual": round(float(a), 3),
                   "pred": round(float(p), 3), "seasonal": round(float(se), 3)}
                  for w, a, p, se in zip(s.week, s.actual, s.pred, s.seasonal)]
    with open(os.path.join(REPORTS, "real_backtest_web.json"), "w") as f:
        json.dump({"calibration": calibration, "fva": fva}, f, separators=(",", ":"))
    print("Saved reports/real_metrics.json and reports/real_backtest_web.json")


if __name__ == "__main__":
    main()
