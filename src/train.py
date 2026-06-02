"""
train.py
========
Train and back-test the occupancy / demand forecaster.

What it does
------------
1. Rolling-origin back-test (3 folds): at each origin we train only on the
   past and forecast the next 90 days -- a true "plan ahead" simulation.
2. Benchmarks the LightGBM model against four baselines, including the strong
   seasonal-naive (market x calendar-month) baseline.
3. Scores on two axes:
     * per-night discrimination  : ROC-AUC, log-loss, Brier
     * occupancy-RATE accuracy   : MAE / RMSE of the predicted occupancy rate
                                    at market-week and listing-month level
                                    (this is what the challenge actually asks).
4. Trains a production model on ALL history and saves every artefact the
   forecaster and dashboard need.

Run:  python3 src/train.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

import features as F

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
MODELS = os.path.join(HERE, "..", "models")
REPORTS = os.path.join(HERE, "..", "reports")
os.makedirs(MODELS, exist_ok=True)
os.makedirs(REPORTS, exist_ok=True)

HORIZON = 90  # forecast 90 days ahead
LGB_PARAMS = dict(
    objective="binary",
    metric="binary_logloss",
    learning_rate=0.05,
    num_leaves=64,
    min_child_samples=200,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    max_depth=-1,
    verbosity=-1,
    seed=42,
)
NUM_ROUNDS = 600


# ---------------------------------------------------------------------------
def load():
    listings = pd.read_csv(os.path.join(DATA, "listings.csv"))
    cal = pd.read_csv(os.path.join(DATA, "calendar.csv"), parse_dates=["date"])
    # attach the columns needed to fit level features
    meta = listings[["listing_id", "market", "archetype"]]
    cal = cal.merge(meta, on="listing_id", how="left")
    cal["month"] = cal["date"].dt.month
    cal["dayofweek"] = cal["date"].dt.dayofweek
    return listings, cal


def occ_rate_metrics(df, pred_col):
    """MAE/RMSE of occupancy RATE at market-week and listing-month level."""
    out = {}
    d = df.copy()
    d["week"] = d["date"].dt.to_period("W").astype(str)
    d["ym"] = d["date"].dt.to_period("M").astype(str)
    for name, keys in [("market_week", ["market", "week"]),
                       ("listing_month", ["listing_id", "ym"])]:
        agg = d.groupby(keys).agg(actual=("booked", "mean"),
                                  pred=(pred_col, "mean")).reset_index()
        err = agg["pred"] - agg["actual"]
        out[f"{name}_MAE"] = float(np.mean(np.abs(err)))
        out[f"{name}_RMSE"] = float(np.sqrt(np.mean(err ** 2)))
    return out


def night_metrics(y, p):
    return dict(
        AUC=float(roc_auc_score(y, p)),
        LogLoss=float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
        Brier=float(brier_score_loss(y, p)),
    )


def train_one(train_cal, listings_static):
    levels = F.fit_levels(train_cal)
    tr = F.build_design(train_cal, listings_static, levels)
    dtrain = lgb.Dataset(tr[F.FEATURES], label=tr["booked"].values,
                         categorical_feature=F.CATEGORICAL,
                         free_raw_data=False)
    booster = lgb.train(LGB_PARAMS, dtrain, num_boost_round=NUM_ROUNDS)
    return booster, levels


def predict(booster, levels, test_cal, listings_static):
    te = F.build_design(test_cal, listings_static, levels)
    te["pred"] = booster.predict(te[F.FEATURES])
    return te


def add_baselines(test_df, levels):
    g = levels["global"]
    mkt = test_df["market"].astype(str)
    test_df["b_global"] = g
    test_df["b_market"] = mkt.map(levels["market"]).astype(float).fillna(g)
    mm = levels["market_month"]
    keys = list(zip(mkt, test_df["month"]))
    test_df["b_seasonal"] = pd.Series(
        [mm.get(k, np.nan) for k in keys], index=test_df.index).astype(float)
    test_df["b_seasonal"] = test_df["b_seasonal"].fillna(test_df["b_market"])
    test_df["b_listing"] = (
        test_df["listing_id"].map(levels["listing"]).astype(float).fillna(g))
    return test_df


# ---------------------------------------------------------------------------
def main():
    listings, cal = load()
    listings_static = F.prepare_static(listings)
    max_date = cal["date"].max()

    # ---- rolling-origin back-test --------------------------------------
    fold_offsets = [180, 90, 0]   # origin = max_date - (offset + HORIZON)
    models_for = {"model": ["LightGBM"], "AUC": [], "LogLoss": [], "Brier": [],
                  "mw_MAE": [], "lm_MAE": []}
    baseline_names = {
        "b_seasonal": "Seasonal-naive (mkt x month)",
        "b_listing": "Listing-average",
        "b_market": "Market-average",
        "b_global": "Global-average",
    }
    fold_records = []
    last_fold_pred = None
    last_fold_levels = None

    for k, off in enumerate(fold_offsets):
        test_end = max_date - pd.Timedelta(days=off)
        test_start = test_end - pd.Timedelta(days=HORIZON - 1)
        train_cal = cal[cal["date"] < test_start]
        test_cal = cal[(cal["date"] >= test_start) & (cal["date"] <= test_end)]
        booster, levels = train_one(train_cal, listings_static)
        te = predict(booster, levels, test_cal, listings_static)
        te = add_baselines(te, levels)

        y = te["booked"].values
        rec = {"fold": k + 1,
               "window": f"{test_start.date()} -> {test_end.date()}"}
        # model
        nm = night_metrics(y, te["pred"].values)
        om = occ_rate_metrics(te, "pred")
        rec["LightGBM"] = {**nm, **om}
        # baselines
        for col, nice in baseline_names.items():
            bm = night_metrics(y, te[col].values)
            bo = occ_rate_metrics(te, col)
            rec[nice] = {**bm, **bo}
        fold_records.append(rec)
        last_fold_pred, last_fold_levels = te, levels
        print(f"  fold {k+1} [{rec['window']}]  "
              f"AUC={nm['AUC']:.3f}  mktwk-MAE={om['market_week_MAE']:.3f}  "
              f"(seasonal-naive MAE={rec[baseline_names['b_seasonal']]['market_week_MAE']:.3f})")

    # ---- aggregate back-test metrics -----------------------------------
    def avg(model_key, metric):
        return float(np.mean([r[model_key][metric] for r in fold_records]))

    summary = {}
    for model_key in ["LightGBM"] + list(baseline_names.values()):
        summary[model_key] = {
            "AUC": avg(model_key, "AUC"),
            "LogLoss": avg(model_key, "LogLoss"),
            "Brier": avg(model_key, "Brier"),
            "market_week_MAE": avg(model_key, "market_week_MAE"),
            "market_week_RMSE": avg(model_key, "market_week_RMSE"),
            "listing_month_MAE": avg(model_key, "listing_month_MAE"),
            "listing_month_RMSE": avg(model_key, "listing_month_RMSE"),
        }

    base_mae = summary["Seasonal-naive (mkt x month)"]["market_week_MAE"]
    model_mae = summary["LightGBM"]["market_week_MAE"]
    improvement = (base_mae - model_mae) / base_mae * 100
    print(f"\n  Occupancy-rate MAE (market-week): "
          f"LightGBM {model_mae:.3f} vs seasonal-naive {base_mae:.3f}  "
          f"=> {improvement:.1f}% better")

    # ---- production model on ALL history -------------------------------
    print("\nTraining production model on all history ...")
    booster, levels = train_one(cal, listings_static)
    booster.save_model(os.path.join(MODELS, "model.txt"))
    import joblib
    joblib.dump(levels, os.path.join(MODELS, "levels.joblib"))
    joblib.dump({"features": F.FEATURES, "categorical": F.CATEGORICAL},
                os.path.join(MODELS, "feature_meta.joblib"))

    # feature importance
    imp = pd.DataFrame({
        "feature": booster.feature_name(),
        "gain": booster.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=False)
    imp.to_csv(os.path.join(REPORTS, "feature_importance.csv"), index=False)

    metrics = {
        "horizon_days": HORIZON,
        "n_folds": len(fold_offsets),
        "summary": summary,
        "headline": {
            "model_market_week_MAE": model_mae,
            "seasonal_naive_market_week_MAE": base_mae,
            "improvement_pct": improvement,
            "model_AUC": summary["LightGBM"]["AUC"],
        },
        "folds": fold_records,
    }
    with open(os.path.join(REPORTS, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved models/ and reports/metrics.json")

    # ---- figures from the most-recent fold -----------------------------
    make_figures(last_fold_pred, imp)
    print("Saved reports/figures/*.png")


# ---------------------------------------------------------------------------
def make_figures(te, imp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = os.path.join(REPORTS, "figures")
    os.makedirs(figdir, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 120, "axes.grid": True,
                         "grid.alpha": 0.25, "font.size": 10})
    TEAL = "#0E8388"

    # 1. Forecast vs actual occupancy rate, market-week, a few markets
    te = te.copy()
    te["week"] = te["date"].dt.to_period("W").apply(lambda p: p.start_time)
    mw = te.groupby(["market", "week"]).agg(
        actual=("booked", "mean"), pred=("pred", "mean"),
        seasonal=("b_seasonal", "mean")).reset_index()
    show = ["Maasai Mara", "Zanzibar", "Nairobi", "Serengeti"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=False)
    for ax, mkt in zip(axes.ravel(), show):
        s = mw[mw.market == mkt]
        ax.plot(s.week, s.actual, "o-", color="#2b2b2b", ms=3, label="Actual")
        ax.plot(s.week, s.pred, "s-", color=TEAL, ms=3, label="Forecast")
        ax.plot(s.week, s.seasonal, "--", color="#E08E45", lw=1,
                label="Seasonal-naive")
        ax.set_title(mkt, fontsize=10)
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=30, labelsize=7)
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle("90-day occupancy forecast vs actual (held-out back-test)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "forecast_vs_actual.png"))
    plt.close(fig)

    # 2. Calibration
    fig, ax = plt.subplots(figsize=(5, 5))
    bins = np.linspace(0, 1, 11)
    te["bin"] = pd.cut(te["pred"], bins, include_lowest=True)
    cal = te.groupby("bin").agg(p=("pred", "mean"),
                                o=("booked", "mean")).dropna()
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
    ax.plot(cal.p, cal.o, "o-", color=TEAL, label="LightGBM")
    ax.set_xlabel("Predicted booking probability")
    ax.set_ylabel("Observed booking frequency")
    ax.set_title("Calibration (held-out)", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "calibration.png"))
    plt.close(fig)

    # 3. Feature importance (top 15)
    top = imp.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top.feature, top.gain, color=TEAL)
    ax.set_title("Top demand drivers (LightGBM gain)", fontweight="bold")
    ax.set_xlabel("Total gain")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "feature_importance.png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
