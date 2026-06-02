"""
validate_kaggle.py
==================
Validate the occupancy forecaster on the official Kaggle Hotel Booking Demand
dataset (Antonio, Almeida, Nunes 2019).

What it does
------------
1. Loads the daily occupancy time series built by fetch_kaggle_hotel.py
2. Rolling-origin back-test (3 folds, 90-day horizon)
3. LightGBM regression vs seasonal-naive and historical-average baselines
4. Metric: MAE / RMSE of occupancy rate at hotel-week granularity
5. Model interpretability: feature importance
6. Exports metrics for the web dashboard
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
REPORTS = os.path.join(HERE, "..", "reports")
OCC_CSV = os.path.join(DATA, "kaggle_occupancy.csv")

HORIZON = 90
LGB_PARAMS = dict(
    objective="regression",
    metric="mae",
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=100,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    max_depth=-1,
    verbosity=-1,
    seed=42,
)
NUM_ROUNDS = 500

HOTEL_LABELS = {"City Hotel": "City Hotel", "Resort Hotel": "Resort Hotel"}

FEATURES = [
    "year", "month", "day", "dayofweek", "dayofyear", "weekofyear", "quarter",
    "is_weekend", "doy_sin", "doy_cos", "dow_sin", "dow_cos",
    "occ_lag_7", "occ_lag_14", "occ_lag_28",
    "occ_roll_7", "occ_roll_28",
    "avg_adr", "adr_lag_7", "adr_lag_14", "adr_lag_28",
    "avg_lead_time", "avg_adults",
    "seg_TA", "seg_TO", "seg_Direct", "seg_Corporate",
    "hotel_avg_occ", "hotel_month_occ",
]


def load():
    df = pd.read_csv(OCC_CSV, parse_dates=["date"])
    df = df.sort_values(["hotel", "date"]).reset_index(drop=True)
    return df


def add_hotel_levels(train_df, test_df, train_col="occupancy_rate"):
    """Compute hotel-level and hotel×month averages from training data."""
    avg = train_df.groupby("hotel")[train_col].mean()
    hm = train_df.groupby(["hotel", "month"])[train_col].mean()
    test_df = test_df.copy()
    test_df["hotel_avg_occ"] = test_df["hotel"].map(avg).fillna(train_df[train_col].mean())
    keys = list(zip(test_df["hotel"], test_df["month"]))
    test_df["hotel_month_occ"] = pd.Series(
        [hm.get(k, np.nan) for k in keys], index=test_df.index
    ).fillna(test_df["hotel_avg_occ"])
    return test_df


def make_baselines(train_df, test_df, train_col="occupancy_rate"):
    """Compute baseline predictions.

    Global average: flat rate across all hotels
    Hotel average: per-hotel historical rate
    Seasonal-naive: hotel × calendar-month average
    """
    test_df = test_df.copy()
    g = train_df[train_col].mean()
    test_df["b_global"] = g
    test_df["b_hotel"] = test_df["hotel"].map(
        train_df.groupby("hotel")[train_col].mean()
    ).fillna(g)
    hm = train_df.groupby(["hotel", "month"])[train_col].mean()
    keys = list(zip(test_df["hotel"], test_df["month"]))
    test_df["b_seasonal"] = pd.Series(
        [hm.get(k, np.nan) for k in keys], index=test_df.index
    ).fillna(test_df["b_hotel"])
    return test_df


def occ_rate_metrics(df, pred_col):
    """MAE/RMSE of occupancy rate at hotel-week granularity."""
    df = df.copy()
    df["week"] = df["date"].dt.to_period("W").astype(str)
    agg = df.groupby(["hotel", "week"]).agg(
        actual=("occupancy_rate", "mean"),
        pred=(pred_col, "mean")
    ).reset_index()
    err = agg["pred"] - agg["actual"]
    return {
        "hotel_week_MAE": float(np.mean(np.abs(err))),
        "hotel_week_RMSE": float(np.sqrt(np.mean(err ** 2))),
    }


def daily_metrics(y, p):
    err = p - y
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
    }


def main():
    df = load()
    print(f"Loaded {len(df):,} daily records from Kaggle hotel data")
    print(f"  Hotels: {df['hotel'].unique().tolist()}")
    print(f"  Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Mean occupancy: {df['occupancy_rate'].mean():.1%}")

    max_date = df["date"].max()
    fold_offsets = [180, 90, 0]  # origin = max_date - (offset + HORIZON)

    baseline_names = {
        "b_seasonal": "Seasonal-naive (hotel × month)",
        "b_hotel": "Hotel-average",
        "b_global": "Global-average",
    }

    fold_records = []
    importances = []

    for k, off in enumerate(fold_offsets):
        test_end = max_date - pd.Timedelta(days=off)
        test_start = test_end - pd.Timedelta(days=HORIZON - 1)
        train_df = df[df["date"] < test_start].copy()
        test_df = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()

        if len(train_df) < 30:
            continue

        # Feature engineering
        train_df = add_hotel_levels(train_df, train_df)
        test_df = add_hotel_levels(train_df, test_df)
        test_df = make_baselines(train_df, test_df)

        # Train LGB
        X_tr = train_df[FEATURES].fillna(0)
        y_tr = train_df["occupancy_rate"]
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        booster = lgb.train(LGB_PARAMS, dtrain, num_boost_round=NUM_ROUNDS)

        # Predict
        X_te = test_df[FEATURES].fillna(0)
        test_df["pred"] = booster.predict(X_te)

        # Metrics
        rec = {
            "fold": k + 1,
            "window": f"{test_start.date()} → {test_end.date()}",
            "train_size": len(train_df),
            "test_size": len(test_df),
        }
        dm = daily_metrics(test_df["occupancy_rate"].values, test_df["pred"].values)
        om = occ_rate_metrics(test_df, "pred")
        rec["LightGBM"] = {**dm, **om}

        for col, nice in baseline_names.items():
            bm = daily_metrics(test_df["occupancy_rate"].values, test_df[col].values)
            bo = occ_rate_metrics(test_df, col)
            rec[nice] = {**bm, **bo}

        fold_records.append(rec)

        imp = pd.DataFrame({
            "feature": booster.feature_name(),
            "gain": booster.feature_importance(importance_type="gain"),
            "fold": k + 1,
        }).sort_values("gain", ascending=False)
        importances.append(imp)

        base_mae = rec[baseline_names["b_seasonal"]]["hotel_week_MAE"]
        model_mae = om["hotel_week_MAE"]
        imp_pct = (base_mae - model_mae) / base_mae * 100 if base_mae > 0 else 0
        print(f"  fold {k+1} [{rec['window']}]  "
              f"daily-MAE={dm['MAE']:.3f}  wk-MAE={om['hotel_week_MAE']:.3f}  "
              f"(seasonal-naive wk-MAE={base_mae:.3f})  "
              f"improvement={imp_pct:+.1f}%")

    if not fold_records:
        print("No folds completed — not enough data.")
        return

    # Aggregate
    def avg(model_key, metric):
        return float(np.mean([r[model_key][metric] for r in fold_records]))

    summary = {}
    all_names = ["LightGBM"] + list(baseline_names.values())
    for model_key in all_names:
        summary[model_key] = {
            "MAE": avg(model_key, "MAE"),
            "RMSE": avg(model_key, "RMSE"),
            "hotel_week_MAE": avg(model_key, "hotel_week_MAE"),
            "hotel_week_RMSE": avg(model_key, "hotel_week_RMSE"),
        }

    base_mae = summary["Seasonal-naive (hotel × month)"]["hotel_week_MAE"]
    model_mae = summary["LightGBM"]["hotel_week_MAE"]
    improvement = (base_mae - model_mae) / base_mae * 100

    print(f"\n  === Kaggle Hotel Validation ===")
    print(f"  Occupancy-rate MAE (hotel-week): "
          f"LightGBM {model_mae:.3f} vs seasonal-naive {base_mae:.3f}  "
          f"=> {improvement:+.1f}%")
    headline = {
        "model_hotel_week_MAE": round(model_mae, 4),
        "seasonal_naive_hotel_week_MAE": round(base_mae, 4),
        "improvement_pct": round(improvement, 1),
        "model_MAE": round(summary["LightGBM"]["MAE"], 4),
    }

    # Feature importance across folds
    imp_all = pd.concat(importances)
    imp_agg = imp_all.groupby("feature")["gain"].mean().sort_values(ascending=False)
    imp_top = imp_agg.head(20).to_dict()

    # Export
    metrics = {
        "source": "Hotel Booking Demand (Antonio, Almeida, Nunes 2019)",
        "source_url": "https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand",
        "n_hotels": int(df["hotel"].nunique()),
        "n_rows": len(df),
        "n_bookings": int(pd.read_csv(os.path.join(DATA, "kaggle_bookings.csv")).shape[0]),
        "mean_occupancy": round(float(df["occupancy_rate"].mean()), 3),
        "horizon_days": HORIZON,
        "n_folds": len(fold_records),
        "headline": headline,
        "summary": {k: {mk: round(mv, 4) if isinstance(mv, float) else mv
                        for mk, mv in v.items()}
                    for k, v in summary.items()},
        "folds": fold_records,
        "importance": {k: round(v, 1) for k, v in imp_top.items()},
    }

    out = os.path.join(REPORTS, "kaggle_metrics.json")
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved {out}")

    # Backtest detail for web calibration + forecast-vs-actual
    last = fold_records[-1]
    test_end = max_date - pd.Timedelta(days=fold_offsets[-1])
    test_start = test_end - pd.Timedelta(days=HORIZON - 1)
    te = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()
    train_df = df[df["date"] < test_start].copy()
    train_df = add_hotel_levels(train_df, train_df)
    te = add_hotel_levels(train_df, te)
    X_te = te[FEATURES].fillna(0)
    # Re-train on full history up to test_start for the detail export
    full_train = df[df["date"] < test_start].copy()
    full_train = add_hotel_levels(full_train, full_train)
    X_full = full_train[FEATURES].fillna(0)
    y_full = full_train["occupancy_rate"]
    booster = lgb.train(LGB_PARAMS, lgb.Dataset(X_full, label=y_full), num_boost_round=NUM_ROUNDS)
    te["pred"] = booster.predict(X_te)
    te = make_baselines(full_train, te)

    # Calibration: predicted vs actual occupancy rate (binned)
    bins = np.linspace(0, 1, 11)
    te["bin"] = pd.cut(te["pred"], bins, include_lowest=True)
    cal = te.groupby("bin").agg(p=("pred", "mean"), o=("occupancy_rate", "mean")).dropna()
    calibration = [{"p": round(float(r.p), 3), "o": round(float(r.o), 3)}
                   for _, r in cal.iterrows()]

    # Forecast vs actual, weekly per hotel
    te["week"] = te["date"].dt.to_period("W").apply(lambda p: p.start_time)
    hw = te.groupby(["hotel", "week"]).agg(
        actual=("occupancy_rate", "mean"),
        pred=("pred", "mean"),
        seasonal=("b_seasonal", "mean"),
    ).reset_index()
    fva = {}
    for h in te["hotel"].unique():
        s = hw[hw.hotel == h].sort_values("week")
        fva[h] = [{"week": w.strftime("%Y-%m-%d"),
                   "actual": round(float(a), 3),
                   "pred": round(float(p), 3),
                   "seasonal": round(float(se), 3)}
                  for w, a, p, se in zip(s.week, s.actual, s.pred, s.seasonal)]

    detail = {"calibration": calibration, "fva": fva}
    det_out = os.path.join(REPORTS, "kaggle_backtest_web.json")
    with open(det_out, "w") as f:
        json.dump(detail, f, separators=(",", ":"))
    print(f"  Saved {det_out}")


if __name__ == "__main__":
    main()
