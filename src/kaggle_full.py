"""
kaggle_full.py
==============
Full pipeline for the Kaggle Hotel Booking Demand dataset.
Per-hotel ensemble (LGB+XGB+CatBoost) on logit-transformed target with
weighted blend, extended features, LGB hyperparam search, quantile intervals.
Uses only features computable for future dates (no lag features), so the
same model powers back-test and forward forecast.

Run after:  python src/fetch_kaggle_hotel.py
"""

import os, json, warnings, itertools
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
REPORTS = os.path.join(HERE, "..", "reports")
OCC_CSV = os.path.join(DATA, "kaggle_occupancy.csv")

HORIZON = 90
FWD_HORIZON = 214
NUM_ROUNDS = 1500
ES_ROUNDS = 80

LGB_BASE = dict(
    objective="regression", metric="mae", verbosity=-1,
    min_child_samples=20, feature_fraction=0.85,
    bagging_fraction=0.85, bagging_freq=1,
    lambda_l1=0.5, lambda_l2=0.5, max_depth=-1, seed=42,
)
LGB_LOWER = dict(objective="quantile", alpha=0.1, metric="quantile", verbosity=-1,
                  num_leaves=31, learning_rate=0.03, seed=42)
LGB_UPPER = dict(objective="quantile", alpha=0.9, metric="quantile", verbosity=-1,
                  num_leaves=31, learning_rate=0.03, seed=42)
XGB_BASE = dict(
    objective="reg:absoluteerror", eval_metric="mae",
    learning_rate=0.03, max_depth=7, subsample=0.85, colsample_bytree=0.85,
    reg_alpha=0.5, reg_lambda=0.5, seed=42, verbosity=0,
)
CB_BASE = dict(
    loss_function="MAE", learning_rate=0.03, depth=7,
    l2_leaf_reg=5, subsample=0.85, random_seed=42, verbose=False,
)

FEATURES = [
    "year", "day", "dayofyear", "weekofyear", "quarter",
    "doy_sin", "doy_cos", "dow_sin", "dow_cos",
    "month_cos",
    "week_sin", "week_cos",
    "quarter_sin", "quarter_cos",
    "day_sin", "day_cos",
    "avg_adr", "avg_lead_time", "avg_adults",
    "seg_Direct", "seg_Corporate",
    "hotel_avg_occ", "hotel_month_occ", "hotel_dow_occ",
    "adr_sq", "adr_log", "lead_time_log",
    "week_of_month", "is_month_start", "is_month_end",
    "is_peak", "hotel_month_cos_x_avg",
]

LGB_GRID = [
    dict(learning_rate=lr, num_leaves=nl)
    for lr, nl in itertools.product([0.02, 0.04], [31, 63, 127])
]


def logit(x, eps=1e-4):
    x = np.clip(x, eps, 1 - eps)
    return np.log(x / (1 - x))


def inv_logit(x):
    return 1 / (1 + np.exp(-np.clip(x, -20, 20)))


def add_features(df_):
    df_["month_cos"] = np.cos(2 * np.pi * df_["month"] / 12)
    df_["week_sin"] = np.sin(2 * np.pi * df_["weekofyear"] / 52)
    df_["week_cos"] = np.cos(2 * np.pi * df_["weekofyear"] / 52)
    df_["quarter_sin"] = np.sin(2 * np.pi * df_["quarter"] / 4)
    df_["quarter_cos"] = np.cos(2 * np.pi * df_["quarter"] / 4)
    df_["day_sin"] = np.sin(2 * np.pi * df_["day"] / 31)
    df_["day_cos"] = np.cos(2 * np.pi * df_["day"] / 31)
    df_["week_of_month"] = (df_["day"] - 1) // 7 + 1
    df_["is_month_start"] = (df_["day"] <= 3).astype(int)
    df_["is_month_end"] = (df_["day"] >= 28).astype(int)
    df_["adr_sq"] = df_["avg_adr"] ** 2 / 10000
    df_["adr_log"] = np.log(df_["avg_adr"] + 1)
    df_["lead_time_log"] = np.log(df_["avg_lead_time"] + 1)
    if "month" in df_.columns:
        df_["is_peak"] = (df_["month"] >= 6) & (df_["month"] <= 8)
    if "month_cos" in df_.columns and "hotel_avg_occ" in df_.columns:
        df_["hotel_month_cos_x_avg"] = df_["month_cos"] * df_["hotel_avg_occ"]
    return df_


def load():
    df = pd.read_csv(OCC_CSV, parse_dates=["date"])
    df = df.sort_values(["hotel", "date"]).reset_index(drop=True)
    return df


def add_hotel_levels(train_df, test_df, target="occupancy_rate"):
    avg = train_df.groupby("hotel")[target].mean()
    hm = train_df.groupby(["hotel", "month"])[target].mean()
    hdow = train_df.groupby(["hotel", "dayofweek"])[target].mean()
    test_df = test_df.copy()
    test_df["hotel_avg_occ"] = test_df["hotel"].map(avg).fillna(train_df[target].mean())
    keys_m = list(zip(test_df["hotel"], test_df["month"]))
    test_df["hotel_month_occ"] = pd.Series(
        [hm.get(k, np.nan) for k in keys_m], index=test_df.index
    ).fillna(test_df["hotel_avg_occ"])
    keys_dow = list(zip(test_df["hotel"], test_df["dayofweek"]))
    test_df["hotel_dow_occ"] = pd.Series(
        [hdow.get(k, np.nan) for k in keys_dow], index=test_df.index
    ).fillna(test_df["hotel_avg_occ"])
    return test_df


def make_baselines(train_df, test_df, target="occupancy_rate"):
    test_df = test_df.copy()
    g = train_df[target].mean()
    test_df["b_global"] = g
    test_df["b_hotel"] = test_df["hotel"].map(train_df.groupby("hotel")[target].mean()).fillna(g)
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


def tune_lgb(X_tr, y_tr, X_val, y_val, grid=LGB_GRID):
    """Quick grid search for LGB mean params. Returns best params dict."""
    y_tr_l = logit(y_tr.values)
    y_val_l = logit(y_val.values)
    best = None
    best_score = float("inf")
    for extra in grid:
        params = dict(LGB_BASE, **extra)
        m = lgb.train(params, lgb.Dataset(X_tr, label=y_tr_l),
                       num_boost_round=NUM_ROUNDS,
                       valid_sets=[lgb.Dataset(X_val, label=y_val_l)],
                       callbacks=[lgb.early_stopping(ES_ROUNDS), lgb.log_evaluation(0)])
        score = list(m.best_score["valid_0"].values())[0]
        if score < best_score:
            best_score = score
            best = extra
    return best


def train_hotel_ensemble(X_tr, y_tr, X_val, y_val, lgb_params=None):
    """Per-hotel ensemble: LGB + XGB + CatBoost, logit target, weighted blend."""
    y_tr_l = logit(y_tr.values)
    y_val_l = logit(y_val.values)

    lgb_p = dict(LGB_BASE, **(lgb_params or dict(learning_rate=0.03, num_leaves=63)))
    lgb_m = lgb.train(lgb_p, lgb.Dataset(X_tr, label=y_tr_l),
                       num_boost_round=NUM_ROUNDS,
                       valid_sets=[lgb.Dataset(X_val, label=y_val_l)],
                       callbacks=[lgb.early_stopping(ES_ROUNDS), lgb.log_evaluation(0)])

    xgb_m = xgb.train(XGB_BASE, xgb.DMatrix(X_tr, label=y_tr_l), num_boost_round=NUM_ROUNDS,
                       evals=[(xgb.DMatrix(X_val, label=y_val_l), "valid")],
                       early_stopping_rounds=ES_ROUNDS, verbose_eval=False)

    cb_m = cb.CatBoost(CB_BASE)
    cb_m.fit(X_tr, y_tr_l, eval_set=(X_val, y_val_l),
             early_stopping_rounds=ES_ROUNDS, verbose=False, plot=False)

    p_lgb_v = inv_logit(lgb_m.predict(X_val, num_iteration=lgb_m.best_iteration))
    p_xgb_v = inv_logit(xgb_m.predict(xgb.DMatrix(X_val), iteration_range=(0, xgb_m.best_iteration)))
    p_cb_v = inv_logit(cb_m.predict(X_val))
    y_v = y_val.values

    best_w = None
    best_err = float("inf")
    for w1 in np.linspace(0, 1, 6):
        for w2 in np.linspace(0, 1 - w1, 6):
            w3 = 1 - w1 - w2
            if w3 < 0:
                continue
            err = np.mean(np.abs(w1 * p_lgb_v + w2 * p_xgb_v + w3 * p_cb_v - y_v))
            if err < best_err:
                best_err = err
                best_w = (w1, w2, w3)

    def predict(X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=FEATURES).astype(float)
        w1, w2, w3 = best_w
        p = w1 * inv_logit(lgb_m.predict(X, num_iteration=lgb_m.best_iteration))
        p += w2 * inv_logit(xgb_m.predict(xgb.DMatrix(X), iteration_range=(0, xgb_m.best_iteration)))
        p += w3 * inv_logit(cb_m.predict(X))
        return p

    return predict


def train_hotel_quantile(X_tr, y_tr, X_val, y_val, params):
    m = lgb.train(params, lgb.Dataset(X_tr, label=y_tr.values),
                   num_boost_round=int(NUM_ROUNDS * 0.5),
                   valid_sets=[lgb.Dataset(X_val, label=y_val.values)],
                   callbacks=[lgb.early_stopping(ES_ROUNDS), lgb.log_evaluation(0)])

    def predict(X):
        return m.predict(X, num_iteration=m.best_iteration)

    return predict


def main():
    df = load()
    print(f"Loaded {len(df):,} daily records")
    hotels = sorted(df["hotel"].unique())
    print(f"  Hotels: {hotels}")
    print(f"  Date range: {df['date'].min().date()} \u2192 {df['date'].max().date()}")
    print(f"  Mean occupancy: {df['occupancy_rate'].mean():.1%}")

    max_date = df["date"].max()
    fold_offsets = [180, 90, 0]
    baseline_names = {
        "b_seasonal": "Seasonal-naive (hotel \u00d7 month)",
        "b_hotel": "Hotel-average",
        "b_global": "Global-average",
    }

    # ---- Tune LGB hyperparams on the largest training window (fold 3) ----
    tune_end = max_date - pd.Timedelta(days=fold_offsets[-1])
    tune_start = tune_end - pd.Timedelta(days=HORIZON - 1)
    tune_train = df[df["date"] < tune_start].copy()
    tune_train = add_hotel_levels(tune_train, tune_train)
    tune_train = add_features(tune_train)
    tu_sorted = tune_train.sort_values("date")
    tu_split = int(len(tu_sorted) * 0.8)
    tu_tr = tu_sorted.iloc[:tu_split]
    tu_val = tu_sorted.iloc[tu_split:]
    best_lgb = {}
    for h in hotels:
        mask_tr = tu_tr["hotel"] == h
        mask_val = tu_val["hotel"] == h
        if mask_tr.sum() < 30:
            best_lgb[h] = dict(learning_rate=0.03, num_leaves=63)
            continue
        X_tr = tu_tr.loc[mask_tr, FEATURES].fillna(0)
        y_tr = tu_tr.loc[mask_tr, "occupancy_rate"]
        X_val = tu_val.loc[mask_val, FEATURES].fillna(0)
        y_val = tu_val.loc[mask_val, "occupancy_rate"]
        print(f"  Tuning LGB for {h}...")
        best_lgb[h] = tune_lgb(X_tr, y_tr, X_val, y_val)

    fold_records = []

    for k, off in enumerate(fold_offsets):
        test_end = max_date - pd.Timedelta(days=off)
        test_start = test_end - pd.Timedelta(days=HORIZON - 1)
        train_df = df[df["date"] < test_start].copy()
        test_df = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()
        if len(train_df) < 30:
            continue

        train_df = add_hotel_levels(train_df, train_df)
        test_df = add_hotel_levels(train_df, test_df)
        train_df = add_features(train_df)
        test_df = add_features(test_df)
        test_df = make_baselines(train_df, test_df)

        tr_sorted = train_df.sort_values("date")
        split_at = int(len(tr_sorted) * 0.8)
        tr_fit = tr_sorted.iloc[:split_at]
        val_fit = tr_sorted.iloc[split_at:]

        test_df["pred"] = 0.0
        test_df["lower"] = 0.0
        test_df["upper"] = 0.0
        for h in hotels:
            mask_tr = tr_fit["hotel"] == h
            mask_val = val_fit["hotel"] == h
            mask_te = test_df["hotel"] == h
            if mask_tr.sum() < 30 or mask_te.sum() < 5:
                test_df.loc[mask_te, "pred"] = test_df.loc[mask_te, "b_hotel"]
                test_df.loc[mask_te, "lower"] = (test_df.loc[mask_te, "pred"] - 0.05).clip(lower=0)
                test_df.loc[mask_te, "upper"] = (test_df.loc[mask_te, "pred"] + 0.05).clip(upper=1)
                continue
            X_tr = tr_fit.loc[mask_tr, FEATURES].fillna(0)
            y_tr = tr_fit.loc[mask_tr, "occupancy_rate"]
            X_val = val_fit.loc[mask_val, FEATURES].fillna(0)
            y_val = val_fit.loc[mask_val, "occupancy_rate"]
            X_te = test_df.loc[mask_te, FEATURES].fillna(0)

            mean_fn = train_hotel_ensemble(X_tr, y_tr, X_val, y_val, lgb_params=best_lgb[h])
            test_df.loc[mask_te, "pred"] = mean_fn(X_te.values)

            lower_fn = train_hotel_quantile(X_tr, y_tr, X_val, y_val, LGB_LOWER)
            upper_fn = train_hotel_quantile(X_tr, y_tr, X_val, y_val, LGB_UPPER)
            test_df.loc[mask_te, "lower"] = lower_fn(X_te.values).clip(min=0)
            test_df.loc[mask_te, "upper"] = upper_fn(X_te.values).clip(max=1.0)

        mask_flip = test_df["lower"] > test_df["pred"]
        test_df.loc[mask_flip, "lower"] = test_df.loc[mask_flip, "pred"] - 0.02
        test_df.loc[mask_flip, "lower"] = test_df["lower"].clip(lower=0)
        mask_flip2 = test_df["upper"] < test_df["pred"]
        test_df.loc[mask_flip2, "upper"] = test_df.loc[mask_flip2, "pred"] + 0.02
        test_df["upper"] = test_df["upper"].clip(upper=1.0)

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
        rec["Ensemble"] = {**dm, **om}

        for col, nice in baseline_names.items():
            bm = daily_metrics(test_df["occupancy_rate"].values, test_df[col].values)
            bo = occ_metrics(test_df, col)
            rec[nice] = {**bm, **bo}

        fold_records.append(rec)

        base_mae = rec[baseline_names["b_seasonal"]]["hotel_week_MAE"]
        model_mae = om["hotel_week_MAE"]
        imp_pct = (base_mae - model_mae) / base_mae * 100 if base_mae > 0 else 0
        print(f"  fold {k+1} [{rec['window']}]  "
              f"daily-MAE={dm['MAE']:.4f}  wk-MAE={model_mae:.4f}  "
              f"(seasonal wk-MAE={base_mae:.4f})  "
              f"improvement={imp_pct:+.1f}%  cov={interval_cov:.1%}")

    if not fold_records:
        print("No folds completed.")
        return

    def avg(model_key, metric):
        return float(np.mean([r[model_key][metric] for r in fold_records]))

    summary = {}
    all_keys = ["Ensemble"] + list(baseline_names.values())
    for mk in all_keys:
        summary[mk] = {
            "MAE": avg(mk, "MAE"), "RMSE": avg(mk, "RMSE"),
            "hotel_week_MAE": avg(mk, "hotel_week_MAE"),
            "hotel_week_RMSE": avg(mk, "hotel_week_RMSE"),
        }

    base_mae = summary["Seasonal-naive (hotel \u00d7 month)"]["hotel_week_MAE"]
    model_mae = summary["Ensemble"]["hotel_week_MAE"]
    improvement = (base_mae - model_mae) / base_mae * 100

    print(f"\n  === Ensemble (per-hotel LGB+XGB+CB, logit, weighted, tuned) ===")
    print(f"  wk-MAE: Ensemble {model_mae:.4f} vs seasonal-naive {base_mae:.4f}  "
          f"=> {improvement:+.1f}%")

    headline = {
        "model_hotel_week_MAE": round(model_mae, 4),
        "seasonal_naive_hotel_week_MAE": round(base_mae, 4),
        "improvement_pct": round(improvement, 1),
        "model_MAE": round(summary["Ensemble"]["MAE"], 4),
    }

    # ---- Train production models ----
    origin = (max_date + pd.Timedelta(days=1)) - pd.Timedelta(days=105)
    train_prod = df[df["date"] < origin].copy()
    train_prod = add_hotel_levels(train_prod, train_prod)
    train_prod = add_features(train_prod)
    prod_sorted = train_prod.sort_values("date")
    split_p = int(len(prod_sorted) * 0.8)
    tr_p = prod_sorted.iloc[:split_p]
    val_p = prod_sorted.iloc[split_p:]

    prod_mean = {}
    prod_lower = {}
    prod_upper = {}
    for h in hotels:
        mask_tr = tr_p["hotel"] == h
        mask_val = val_p["hotel"] == h
        if mask_tr.sum() < 30:
            fallback = train_prod.loc[train_prod["hotel"] == h, "occupancy_rate"].mean()
            prod_mean[h] = lambda x, v=fallback: np.full(len(x), v)
            prod_lower[h] = lambda x, v=fallback - 0.05: np.full(len(x), max(v, 0))
            prod_upper[h] = lambda x, v=fallback + 0.05: np.full(len(x), min(v, 1))
            continue
        X_tr = tr_p.loc[mask_tr, FEATURES].fillna(0)
        y_tr = tr_p.loc[mask_tr, "occupancy_rate"]
        X_val = val_p.loc[mask_val, FEATURES].fillna(0)
        y_val = val_p.loc[mask_val, "occupancy_rate"]
        print(f"  Training production models for {h}...")
        prod_mean[h] = train_hotel_ensemble(X_tr, y_tr, X_val, y_val, lgb_params=best_lgb[h])
        prod_lower[h] = train_hotel_quantile(X_tr, y_tr, X_val, y_val, LGB_LOWER)
        prod_upper[h] = train_hotel_quantile(X_tr, y_tr, X_val, y_val, LGB_UPPER)

    # ---- Forward forecast ----
    future_dates = pd.date_range(origin, periods=FWD_HORIZON, freq="D")
    grid = pd.MultiIndex.from_product(
        [hotels, future_dates], names=["hotel", "date"]
    ).to_frame(index=False)

    grid["year"] = grid["date"].dt.year
    grid["month"] = grid["date"].dt.month
    grid["day"] = grid["date"].dt.day
    grid["dayofweek"] = grid["date"].dt.dayofweek
    grid["dayofyear"] = grid["date"].dt.dayofyear
    grid["weekofyear"] = grid["date"].dt.isocalendar().week.astype(int)
    grid["quarter"] = grid["date"].dt.quarter
    grid["doy_sin"] = np.sin(2 * np.pi * grid["dayofyear"] / 365.25)
    grid["doy_cos"] = np.cos(2 * np.pi * grid["dayofyear"] / 365.25)
    grid["dow_sin"] = np.sin(2 * np.pi * grid["dayofweek"] / 7)
    grid["dow_cos"] = np.cos(2 * np.pi * grid["dayofweek"] / 7)

    recent = train_prod[train_prod["date"] >= train_prod["date"].max() - pd.Timedelta(days=90)]
    hotel_means = recent.groupby("hotel").agg(
        avg_adr=("avg_adr", "mean"), avg_lead_time=("avg_lead_time", "mean"),
        avg_adults=("avg_adults", "mean"),
        seg_Direct=("seg_Direct", "mean"), seg_Corporate=("seg_Corporate", "mean"),
    ).to_dict(orient="index")

    for h in hotels:
        mask = grid["hotel"] == h
        hm = hotel_means[h]
        for col in ["avg_adr", "avg_lead_time", "avg_adults", "seg_Direct", "seg_Corporate"]:
            grid.loc[mask, col] = hm[col]

    grid = add_hotel_levels(train_prod, grid)
    grid = add_features(grid)

    grid["occ_forecast"] = 0.0
    grid["occ_lower"] = 0.0
    grid["occ_upper"] = 0.0
    for h in hotels:
        mask = grid["hotel"] == h
        X_fwd = grid.loc[mask, FEATURES].fillna(0)
        grid.loc[mask, "occ_forecast"] = prod_mean[h](X_fwd.values)
        grid.loc[mask, "occ_lower"] = prod_lower[h](X_fwd.values).clip(min=0)
        grid.loc[mask, "occ_upper"] = prod_upper[h](X_fwd.values).clip(max=1.0)

    mask_flip = grid["occ_lower"] > grid["occ_forecast"]
    grid.loc[mask_flip, "occ_lower"] = grid.loc[mask_flip, "occ_forecast"] - 0.02
    grid["occ_lower"] = grid["occ_lower"].clip(lower=0)
    mask_flip2 = grid["occ_upper"] < grid["occ_forecast"]
    grid.loc[mask_flip2, "occ_upper"] = grid.loc[mask_flip2, "occ_forecast"] + 0.02
    grid["occ_upper"] = grid["occ_upper"].clip(upper=1.0)

    fc_hotel = grid[["hotel", "date", "occ_forecast", "occ_lower", "occ_upper"]].copy()
    fc_hotel.to_csv(os.path.join(REPORTS, "kaggle_forecast_hotel.csv"), index=False)

    # ---- Planning summary ----
    grid["week"] = grid["date"].dt.to_period("W").apply(lambda p: p.start_time)
    wk = grid.groupby(["hotel", "week"]).occ_forecast.mean().reset_index()
    plan_rows = []
    for h, grp in wk.groupby("hotel"):
        g = grp.sort_values("week")
        peak = g.loc[g.occ_forecast.idxmax()]
        low = g.loc[g.occ_forecast.idxmin()]
        avg_val = g.occ_forecast.mean()
        plan_rows.append({
            "hotel": h,
            "avg_occ_90d": round(float(avg_val), 3),
            "peak_week": peak.week.date().isoformat(),
            "peak_occ": round(float(peak.occ_forecast), 3),
            "low_week": low.week.date().isoformat(),
            "low_occ": round(float(low.occ_forecast), 3),
        })
    plan = pd.DataFrame(plan_rows)

    def recommend(r):
        if r.peak_occ >= 0.80:
            return f"Raise rates the week of {r.peak_week} (forecast {r.peak_occ:.0%} full)."
        if r.low_occ <= 0.45:
            return f"Add a promo around {r.low_week} (soft demand {r.low_occ:.0%})."
        return "Demand stable \u2014 maintain current strategy."

    plan["recommendation"] = plan.apply(recommend, axis=1)
    plan.sort_values("avg_occ_90d", ascending=False).to_csv(
        os.path.join(REPORTS, "kaggle_planning_summary.csv"), index=False
    )

    # ---- Run meta ----
    meta = {
        "origin": origin.date().isoformat(),
        "horizon_days": FWD_HORIZON,
        "forecast_end": future_dates[-1].date().isoformat(),
        "n_hotels": int(df["hotel"].nunique()),
        "portfolio_avg_occ_90d": round(float(grid.occ_forecast.mean()), 3),
        "model": "per-hotel ensemble (LGB+XGB+CB, logit, weighted blend, tuned)",
    }
    with open(os.path.join(REPORTS, "kaggle_run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ---- Feature importance ----
    tr_imp = train_prod.sort_values("date")
    tri_split = int(len(tr_imp) * 0.8)
    tri_tr = tr_imp.iloc[:tri_split]
    tri_val = tr_imp.iloc[tri_split:]
    lgb_imp = lgb.train(dict(LGB_BASE, learning_rate=0.03, num_leaves=63),
                         lgb.Dataset(tri_tr[FEATURES].fillna(0), label=tri_tr["occupancy_rate"]),
                         num_boost_round=NUM_ROUNDS,
                         valid_sets=[lgb.Dataset(tri_val[FEATURES].fillna(0), label=tri_val["occupancy_rate"])],
                         callbacks=[lgb.early_stopping(ES_ROUNDS), lgb.log_evaluation(0)])
    imp_df = pd.DataFrame({
        "feature": lgb_imp.feature_name(),
        "gain": lgb_imp.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=False)
    imp_df.to_csv(os.path.join(REPORTS, "kaggle_importance.csv"), index=False)

    # ---- Back-test detail for web charts (last fold) ----
    test_end = max_date - pd.Timedelta(days=fold_offsets[-1])
    test_start_bk = test_end - pd.Timedelta(days=HORIZON - 1)
    te = df[(df["date"] >= test_start_bk) & (df["date"] <= test_end)].copy()
    train_te = df[df["date"] < test_start_bk].copy()
    train_te = add_hotel_levels(train_te, train_te)
    te = add_hotel_levels(train_te, te)
    train_te = add_features(train_te)
    te = add_features(te)

    te["pred"] = 0.0
    te["lower"] = 0.0
    te["upper"] = 0.0
    for h in hotels:
        mask_tr_bt = train_te["hotel"] == h
        mask_te_bt = te["hotel"] == h
        if mask_tr_bt.sum() < 30 or mask_te_bt.sum() < 5:
            te.loc[mask_te_bt, "pred"] = te.loc[mask_te_bt, "b_hotel"]
            te.loc[mask_te_bt, "lower"] = (te.loc[mask_te_bt, "pred"] - 0.05).clip(lower=0)
            te.loc[mask_te_bt, "upper"] = (te.loc[mask_te_bt, "pred"] + 0.05).clip(upper=1)
            continue
        bdt = train_te.loc[mask_tr_bt].sort_values("date")
        bd_split = int(len(bdt) * 0.8)
        bd_tr = bdt.iloc[:bd_split]
        bd_val = bdt.iloc[bd_split:]
        X_tr_bt = bd_tr[FEATURES].fillna(0)
        y_tr_bt = bd_tr["occupancy_rate"]
        X_val_bt = bd_val[FEATURES].fillna(0)
        y_val_bt = bd_val["occupancy_rate"]
        X_te_bt = te.loc[mask_te_bt, FEATURES].fillna(0)

        mean_fn_bt = train_hotel_ensemble(X_tr_bt, y_tr_bt, X_val_bt, y_val_bt, lgb_params=best_lgb[h])
        te.loc[mask_te_bt, "pred"] = mean_fn_bt(X_te_bt.values)
        lower_fn_bt = train_hotel_quantile(X_tr_bt, y_tr_bt, X_val_bt, y_val_bt, LGB_LOWER)
        upper_fn_bt = train_hotel_quantile(X_tr_bt, y_tr_bt, X_val_bt, y_val_bt, LGB_UPPER)
        te.loc[mask_te_bt, "lower"] = lower_fn_bt(X_te_bt.values).clip(min=0)
        te.loc[mask_te_bt, "upper"] = upper_fn_bt(X_te_bt.values).clip(max=1.0)

    mask_flip = te["lower"] > te["pred"]
    te.loc[mask_flip, "lower"] = te.loc[mask_flip, "pred"] - 0.02
    te["lower"] = te["lower"].clip(lower=0)
    mask_flip2 = te["upper"] < te["pred"]
    te.loc[mask_flip2, "upper"] = te.loc[mask_flip2, "pred"] + 0.02
    te["upper"] = te["upper"].clip(upper=1.0)
    te = make_baselines(train_te, te)

    n_bins = min(10, max(3, int(len(te) / 15)))
    bins = np.linspace(te["pred"].min(), te["pred"].max(), n_bins + 1)
    te["bin"] = pd.cut(te["pred"], bins, include_lowest=True)
    cal = te.groupby("bin").agg(p=("pred", "mean"), o=("occupancy_rate", "mean")).dropna()
    calibration = [
        {"p": round(float(r.p), 3), "o": round(float(r.o), 3)}
        for _, r in cal.iterrows()
    ]

    te["week"] = te["date"].dt.to_period("W").apply(lambda p: p.start_time)
    hw = te.groupby(["hotel", "week"]).agg(
        actual=("occupancy_rate", "mean"), pred=("pred", "mean"),
        lower=("lower", "mean"), upper=("upper", "mean"),
        seasonal=("b_seasonal", "mean"),
    ).reset_index()
    fva = {}
    for h in hotels:
        s = hw[hw.hotel == h].sort_values("week")
        fva[h] = [
            {"week": w.strftime("%Y-%m-%d"), "actual": round(float(a), 3),
             "pred": round(float(p), 3), "lower": round(float(lw), 3),
             "upper": round(float(up), 3), "seasonal": round(float(se), 3)}
            for w, a, p, lw, up, se in zip(
                s.week, s.actual, s.pred, s.lower, s.upper, s.seasonal
            )
        ]

    with open(os.path.join(REPORTS, "kaggle_fwd_backtest.json"), "w") as f:
        json.dump({"calibration": calibration, "fva": fva}, f, separators=(",", ":"))

    # ---- Export metrics ----
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
        "summary": {
            k: {mk: round(mv, 4) if isinstance(mv, float) else mv
                for mk, mv in v.items()}
            for k, v in summary.items()
        },
        "folds": fold_records,
    }
    with open(os.path.join(REPORTS, "kaggle_fwd_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  Forward forecast: {origin.date()} \u2192 {future_dates[-1].date()}")
    for h in hotels:
        avg_fc = grid[grid.hotel == h].occ_forecast.mean()
        print(f"    {h}: {FWD_HORIZON}-day avg {avg_fc:.1%}")
    print(f"  Saved all outputs to reports/")
    for h in hotels:
        print(f"  Best LGB params for {h}: lr={best_lgb[h].get('learning_rate')}, "
              f"nl={best_lgb[h].get('num_leaves')}")
    print(f"\n  === FINAL ===")
    print(f"  Ensemble wk-MAE: {model_mae:.4f} vs seasonal {base_mae:.4f} = {improvement:+.1f}%")


if __name__ == "__main__":
    main()
