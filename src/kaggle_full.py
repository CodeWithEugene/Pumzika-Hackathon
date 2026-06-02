"""
kaggle_full.py
==============
Full pipeline for the official Kaggle Hotel Booking Demand dataset.
Trains a LightGBM regression model using only features that can be computed
for future dates (no lag features), so the same model powers both the
historical back-test and the forward forecast.

Run after:  python src/fetch_kaggle_hotel.py
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

LGB_PARAMS_LOWER = dict(LGB_PARAMS, objective="quantile", alpha=0.1, metric="quantile")
LGB_PARAMS_UPPER = dict(LGB_PARAMS, objective="quantile", alpha=0.9, metric="quantile")
QUANTILE_NAMES = ["pred", "lower", "upper"]
QUANTILE_PARAMS = [LGB_PARAMS, LGB_PARAMS_LOWER, LGB_PARAMS_UPPER]

FEATURES = [
    "year", "month", "day", "dayofweek", "dayofyear", "weekofyear", "quarter",
    "is_weekend", "doy_sin", "doy_cos", "dow_sin", "dow_cos",
    "avg_adr", "avg_lead_time", "avg_adults",
    "seg_TA", "seg_TO", "seg_Direct", "seg_Corporate",
    "hotel_avg_occ", "hotel_month_occ",
]


def load():
    df = pd.read_csv(OCC_CSV, parse_dates=["date"])
    df = df.sort_values(["hotel", "date"]).reset_index(drop=True)
    return df


def add_hotel_levels(train_df, test_df, target="occupancy_rate"):
    avg = train_df.groupby("hotel")[target].mean()
    hm = train_df.groupby(["hotel", "month"])[target].mean()
    test_df = test_df.copy()
    test_df["hotel_avg_occ"] = test_df["hotel"].map(avg).fillna(train_df[target].mean())
    keys = list(zip(test_df["hotel"], test_df["month"]))
    test_df["hotel_month_occ"] = pd.Series(
        [hm.get(k, np.nan) for k in keys], index=test_df.index
    ).fillna(test_df["hotel_avg_occ"])
    return test_df


def make_baselines(train_df, test_df, target="occupancy_rate"):
    test_df = test_df.copy()
    g = train_df[target].mean()
    test_df["b_global"] = g
    test_df["b_hotel"] = test_df["hotel"].map(
        train_df.groupby("hotel")[target].mean()
    ).fillna(g)
    hm = train_df.groupby(["hotel", "month"])[target].mean()
    keys = list(zip(test_df["hotel"], test_df["month"]))
    test_df["b_seasonal"] = pd.Series(
        [hm.get(k, np.nan) for k in keys], index=test_df.index
    ).fillna(test_df["b_hotel"])
    return test_df


def occ_metrics(df, pred_col):
    df = df.copy()
    df["week"] = df["date"].dt.to_period("W").astype(str)
    agg = df.groupby(["hotel", "week"]).agg(
        actual=("occupancy_rate", "mean"), pred=(pred_col, "mean")
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
    print(f"Loaded {len(df):,} daily records")
    print(f"  Hotels: {df['hotel'].unique().tolist()}")
    print(f"  Date range: {df['date'].min().date()} \u2192 {df['date'].max().date()}")
    print(f"  Mean occupancy: {df['occupancy_rate'].mean():.1%}")

    max_date = df["date"].max()
    fold_offsets = [180, 90, 0]
    baseline_names = {
        "b_seasonal": "Seasonal-naive (hotel \u00d7 month)",
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

        train_df = add_hotel_levels(train_df, train_df)
        test_df = add_hotel_levels(train_df, test_df)
        test_df = make_baselines(train_df, test_df)

        X_tr = train_df[FEATURES].fillna(0)
        y_tr = train_df["occupancy_rate"]
        boosters = {}
        for name, params in zip(QUANTILE_NAMES, QUANTILE_PARAMS):
            boosters[name] = lgb.train(
                params, lgb.Dataset(X_tr, label=y_tr), num_boost_round=NUM_ROUNDS
            )

        X_te = test_df[FEATURES].fillna(0)
        for name in QUANTILE_NAMES:
            test_df[name] = boosters[name].predict(X_te)
        test_df["lower"] = test_df["lower"].clip(lower=0)
        test_df["upper"] = test_df["upper"].clip(upper=1.0)
        test_df.loc[test_df["lower"] > test_df["pred"], "lower"] = (
            test_df["pred"] - 0.02
        )
        test_df.loc[test_df["upper"] < test_df["pred"], "upper"] = (
            test_df["pred"] + 0.02
        )

        interval_cov = float(
            ((test_df["lower"] <= test_df["occupancy_rate"])
             & (test_df["occupancy_rate"] <= test_df["upper"])).mean()
        )

        rec = {
            "fold": k + 1,
            "window": f"{test_start.date()} \u2192 {test_end.date()}",
            "train_size": len(train_df),
            "test_size": len(test_df),
            "interval_coverage": round(interval_cov, 3),
        }
        dm = daily_metrics(test_df["occupancy_rate"].values, test_df["pred"].values)
        om = occ_metrics(test_df, "pred")
        rec["LightGBM"] = {**dm, **om}

        for col, nice in baseline_names.items():
            bm = daily_metrics(test_df["occupancy_rate"].values, test_df[col].values)
            bo = occ_metrics(test_df, col)
            rec[nice] = {**bm, **bo}

        fold_records.append(rec)

        imp = pd.DataFrame({
            "feature": boosters["pred"].feature_name(),
            "gain": boosters["pred"].feature_importance(importance_type="gain"),
            "fold": k + 1,
        }).sort_values("gain", ascending=False)
        importances.append(imp)

        base_mae = rec[baseline_names["b_seasonal"]]["hotel_week_MAE"]
        model_mae = om["hotel_week_MAE"]
        imp_pct = (base_mae - model_mae) / base_mae * 100 if base_mae > 0 else 0
        print(
            f"  fold {k+1} [{rec['window']}]  "
            f"daily-MAE={dm['MAE']:.3f}  wk-MAE={model_mae:.3f}  "
            f"(seasonal wk-MAE={base_mae:.3f})  "
            f"improvement={imp_pct:+.1f}%"
        )

    if not fold_records:
        print("No folds completed.")
        return

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

    base_mae = summary["Seasonal-naive (hotel \u00d7 month)"]["hotel_week_MAE"]
    model_mae = summary["LightGBM"]["hotel_week_MAE"]
    improvement = (base_mae - model_mae) / base_mae * 100

    print(f"\n  === Kaggle Forward Model (no lag features) ===")
    print(
        f"  wk-MAE: LightGBM {model_mae:.3f} vs seasonal-naive {base_mae:.3f}  "
        f"=> {improvement:+.1f}%"
    )

    headline = {
        "model_hotel_week_MAE": round(model_mae, 4),
        "seasonal_naive_hotel_week_MAE": round(base_mae, 4),
        "improvement_pct": round(improvement, 1),
        "model_MAE": round(summary["LightGBM"]["MAE"], 4),
    }

    imp_all = pd.concat(importances)
    imp_agg = imp_all.groupby("feature")["gain"].mean().sort_values(ascending=False)
    imp_top = imp_agg.head(20).to_dict()

    # ---- Train production models on ALL history (mean + quantiles) ----
    full = add_hotel_levels(df, df)
    X_full = full[FEATURES].fillna(0)
    y_full = full["occupancy_rate"]
    prod_boosters = {}
    for name, params in zip(QUANTILE_NAMES, QUANTILE_PARAMS):
        prod_boosters[name] = lgb.train(
            params, lgb.Dataset(X_full, label=y_full), num_boost_round=NUM_ROUNDS
        )

    # ---- Forward 90-day forecast ----
    origin = max_date + pd.Timedelta(days=1)
    future_dates = pd.date_range(origin, periods=HORIZON, freq="D")
    grid = pd.MultiIndex.from_product(
        [df["hotel"].unique(), future_dates], names=["hotel", "date"]
    ).to_frame(index=False)

    grid["year"] = grid["date"].dt.year
    grid["month"] = grid["date"].dt.month
    grid["day"] = grid["date"].dt.day
    grid["dayofweek"] = grid["date"].dt.dayofweek
    grid["dayofyear"] = grid["date"].dt.dayofyear
    grid["weekofyear"] = grid["date"].dt.isocalendar().week.astype(int)
    grid["quarter"] = grid["date"].dt.quarter
    grid["is_weekend"] = (grid["date"].dt.dayofweek >= 4).astype(int)
    grid["doy_sin"] = np.sin(2 * np.pi * grid["dayofyear"] / 365.25)
    grid["doy_cos"] = np.cos(2 * np.pi * grid["dayofyear"] / 365.25)
    grid["dow_sin"] = np.sin(2 * np.pi * grid["dayofweek"] / 7)
    grid["dow_cos"] = np.cos(2 * np.pi * grid["dayofweek"] / 7)

    recent = df[df["date"] >= max_date - pd.Timedelta(days=90)]
    hotel_means = recent.groupby("hotel").agg(
        avg_adr=("avg_adr", "mean"),
        avg_lead_time=("avg_lead_time", "mean"),
        avg_adults=("avg_adults", "mean"),
        seg_TA=("seg_TA", "mean"),
        seg_TO=("seg_TO", "mean"),
        seg_Direct=("seg_Direct", "mean"),
        seg_Corporate=("seg_Corporate", "mean"),
    ).to_dict(orient="index")

    for h in df["hotel"].unique():
        mask = grid["hotel"] == h
        hm = hotel_means[h]
        for col in [
            "avg_adr", "avg_lead_time", "avg_adults",
            "seg_TA", "seg_TO", "seg_Direct", "seg_Corporate",
        ]:
            grid.loc[mask, col] = hm[col]

    grid = add_hotel_levels(df, grid)

    X_fwd = grid[FEATURES].fillna(0)
    grid["occ_forecast"] = prod_boosters["pred"].predict(X_fwd)
    grid["occ_lower"] = prod_boosters["lower"].predict(X_fwd).clip(min=0)
    grid["occ_upper"] = prod_boosters["upper"].predict(X_fwd).clip(max=1.0)
    mask_flip = grid["occ_lower"] > grid["occ_forecast"]
    grid.loc[mask_flip, "occ_lower"] = grid.loc[mask_flip, "occ_forecast"] - 0.02
    mask_flip2 = grid["occ_upper"] < grid["occ_forecast"]
    grid.loc[mask_flip2, "occ_upper"] = grid.loc[mask_flip2, "occ_forecast"] + 0.02

    fc_hotel = grid[["hotel", "date", "occ_forecast", "occ_lower", "occ_upper"]].copy()
    fc_hotel.to_csv(
        os.path.join(REPORTS, "kaggle_forecast_hotel.csv"), index=False
    )

    # ---- Planning summary ----
    grid["week"] = grid["date"].dt.to_period("W").apply(lambda p: p.start_time)
    wk = grid.groupby(["hotel", "week"]).occ_forecast.mean().reset_index()
    rows = []
    for h, grp in wk.groupby("hotel"):
        g = grp.sort_values("week")
        peak = g.loc[g.occ_forecast.idxmax()]
        low = g.loc[g.occ_forecast.idxmin()]
        avg_val = g.occ_forecast.mean()
        rows.append({
            "hotel": h,
            "avg_occ_90d": round(float(avg_val), 3),
            "peak_week": peak.week.date().isoformat(),
            "peak_occ": round(float(peak.occ_forecast), 3),
            "low_week": low.week.date().isoformat(),
            "low_occ": round(float(low.occ_forecast), 3),
        })
    plan = pd.DataFrame(rows)

    def recommend(r):
        if r.peak_occ >= 0.80:
            return (
                f"Raise rates for the week of {r.peak_week} "
                f"(forecast {r.peak_occ:.0%} full)."
            )
        if r.low_occ <= 0.45:
            return (
                f"Add a promo / min-stay discount around {r.low_week} "
                f"(soft demand {r.low_occ:.0%})."
            )
        if r.avg_occ_90d >= 0.70:
            return "Strong quarter ahead - protect availability, test a price uplift."
        return "Steady demand - keep rates, watch weekend pickup."

    plan["recommendation"] = plan.apply(recommend, axis=1)
    plan.sort_values("avg_occ_90d", ascending=False).to_csv(
        os.path.join(REPORTS, "kaggle_planning_summary.csv"), index=False
    )

    # ---- Run meta ----
    meta = {
        "origin": origin.date().isoformat(),
        "horizon_days": HORIZON,
        "forecast_end": future_dates[-1].date().isoformat(),
        "n_hotels": int(df["hotel"].nunique()),
        "portfolio_avg_occ_90d": round(float(grid.occ_forecast.mean()), 3),
    }
    with open(os.path.join(REPORTS, "kaggle_run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ---- Feature importance ----
    imp_df = pd.DataFrame(
        list(imp_top.items()), columns=["feature", "gain"]
    )
    imp_df.to_csv(
        os.path.join(REPORTS, "kaggle_importance.csv"), index=False
    )

    # ---- Back-test detail for web charts (last fold) ----
    test_end = max_date - pd.Timedelta(days=fold_offsets[-1])
    test_start = test_end - pd.Timedelta(days=HORIZON - 1)
    te = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()
    train_te = df[df["date"] < test_start].copy()
    train_te = add_hotel_levels(train_te, train_te)
    te = add_hotel_levels(train_te, te)
    X_te_last = te[FEATURES].fillna(0)

    full_train = train_te.copy()
    last_boosters = {}
    for name, params in zip(QUANTILE_NAMES, QUANTILE_PARAMS):
        last_boosters[name] = lgb.train(
            params,
            lgb.Dataset(
                full_train[FEATURES].fillna(0), label=full_train["occupancy_rate"]
            ),
            num_boost_round=NUM_ROUNDS,
        )
    for name in QUANTILE_NAMES:
        te[name] = last_boosters[name].predict(X_te_last)
    te["lower"] = te["lower"].clip(lower=0)
    te["upper"] = te["upper"].clip(upper=1.0)
    te.loc[te["lower"] > te["pred"], "lower"] = te["pred"] - 0.02
    te.loc[te["upper"] < te["pred"], "upper"] = te["pred"] + 0.02
    te = make_baselines(full_train, te)

    bins = np.linspace(0, 1, 11)
    te["bin"] = pd.cut(te["pred"], bins, include_lowest=True)
    cal = (
        te.groupby("bin")
        .agg(p=("pred", "mean"), o=("occupancy_rate", "mean"))
        .dropna()
    )
    calibration = [
        {"p": round(float(r.p), 3), "o": round(float(r.o), 3)}
        for _, r in cal.iterrows()
    ]

    te["week"] = te["date"].dt.to_period("W").apply(lambda p: p.start_time)
    hw = (
        te.groupby(["hotel", "week"])
        .agg(
            actual=("occupancy_rate", "mean"),
            pred=("pred", "mean"),
            lower=("lower", "mean"),
            upper=("upper", "mean"),
            seasonal=("b_seasonal", "mean"),
        )
        .reset_index()
    )
    fva = {}
    for h in te["hotel"].unique():
        s = hw[hw.hotel == h].sort_values("week")
        fva[h] = [
            {
                "week": w.strftime("%Y-%m-%d"),
                "actual": round(float(a), 3),
                "pred": round(float(p), 3),
                "lower": round(float(lw), 3),
                "upper": round(float(up), 3),
                "seasonal": round(float(se), 3),
            }
            for w, a, p, lw, up, se in zip(
                s.week, s.actual, s.pred, s.lower, s.upper, s.seasonal
            )
        ]

    with open(os.path.join(REPORTS, "kaggle_fwd_backtest.json"), "w") as f:
        json.dump(
            {"calibration": calibration, "fva": fva}, f, separators=(",", ":")
        )

    # ---- Export metrics ----
    metrics = {
        "source": "Hotel Booking Demand (Antonio, Almeida, Nunes 2019)",
        "source_url": "https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand",
        "n_hotels": int(df["hotel"].nunique()),
        "n_rows": len(df),
        "n_bookings": int(
            pd.read_csv(os.path.join(DATA, "kaggle_bookings.csv")).shape[0]
        ),
        "mean_occupancy": round(float(df["occupancy_rate"].mean()), 3),
        "horizon_days": HORIZON,
        "n_folds": len(fold_records),
        "headline": headline,
        "summary": {
            k: {
                mk: round(mv, 4) if isinstance(mv, float) else mv
                for mk, mv in v.items()
            }
            for k, v in summary.items()
        },
        "folds": fold_records,
        "importance": {k: round(v, 1) for k, v in imp_top.items()},
    }
    with open(os.path.join(REPORTS, "kaggle_fwd_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(
        f"\n  Forward forecast: {origin.date()} \u2192 {future_dates[-1].date()}"
    )
    for h in df["hotel"].unique():
        mean_fc = grid[grid.hotel == h].occ_forecast.mean()
        print(f"    {h}: 90-day avg {mean_fc:.1%}")
    print(
        "  Saved kaggle_forecast_hotel.csv, kaggle_planning_summary.csv, "
        "kaggle_importance.csv, kaggle_run_meta.json"
    )
    print(
        "  Saved kaggle_fwd_metrics.json, kaggle_fwd_backtest.json"
    )


if __name__ == "__main__":
    main()
