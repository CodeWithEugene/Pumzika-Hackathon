"""
features.py
===========
Leakage-safe feature engineering shared by training, back-testing and forward
forecasting.

Design principle: every feature must be knowable at the *forecast origin*.

  * Calendar / seasonality / holiday features depend only on the target date,
    which is always known in advance.
  * "Level" features (how busy a listing / market / season usually is) are
    EMPIRICAL averages estimated strictly from data *before* the origin and
    then looked up for future dates. They are recomputed per back-test fold so
    no future information ever leaks into the past.
  * Realised nightly price is deliberately EXCLUDED -- future price is a
    decision, not a known input, and including the realised price would leak
    the booking outcome. The model therefore produces a clean demand forecast
    under expected pricing (and hands off to the Dynamic-Pricing track).
"""

import numpy as np
import pandas as pd

from generate_data import holiday_multipliers

HOLIDAYS = holiday_multipliers()
_HOLIDAY_DATES = np.array(sorted(HOLIDAYS.keys()))

# ---------------------------------------------------------------------------
# Static listing columns carried straight into the model
# ---------------------------------------------------------------------------
STATIC_NUM = [
    "bedrooms", "capacity", "base_price", "review_score", "num_reviews",
    "host_tenure_days", "lat", "lon",
]
STATIC_BOOL = [
    "is_superhost", "instant_book", "wifi", "air_conditioning",
    "pool", "parking", "kitchen",
]
STATIC_CAT = ["market", "country", "archetype", "property_type"]

CAL_NUM = [
    "year", "month", "day", "dayofweek", "dayofyear", "weekofyear", "quarter",
    "is_weekend", "doy_sin", "doy_cos", "dow_sin", "dow_cos",
    "is_holiday", "holiday_mult", "days_to_holiday",
]
LEVEL_NUM = [
    "listing_baseline_occ", "market_baseline_occ",
    "market_month_occ", "archetype_dow_occ", "market_lead_dummy",
]

FEATURES = STATIC_NUM + STATIC_BOOL + STATIC_CAT + CAL_NUM + LEVEL_NUM
CATEGORICAL = STATIC_CAT


# ---------------------------------------------------------------------------
# Calendar features (always known in advance)
# ---------------------------------------------------------------------------
def add_calendar_features(df):
    d = df["date"]
    df["year"] = d.dt.year
    df["month"] = d.dt.month
    df["day"] = d.dt.day
    df["dayofweek"] = d.dt.dayofweek
    df["dayofyear"] = d.dt.dayofyear
    df["weekofyear"] = d.dt.isocalendar().week.astype(int)
    df["quarter"] = d.dt.quarter
    df["is_weekend"] = (d.dt.dayofweek >= 4).astype(int)  # Fri-Sun leisure
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    nd = d.dt.normalize()
    df["holiday_mult"] = nd.map(HOLIDAYS).fillna(1.0).astype(float)
    df["is_holiday"] = (df["holiday_mult"] > 1.0).astype(int)
    # signed-magnitude days to the nearest holiday (clipped to +/-30)
    tvals = nd.values.astype("datetime64[D]")
    hol = _HOLIDAY_DATES.astype("datetime64[D]")
    idx = np.searchsorted(hol, tvals)
    idx = np.clip(idx, 1, len(hol) - 1)
    nearest = np.minimum(
        np.abs((tvals - hol[idx]).astype(int)),
        np.abs((tvals - hol[idx - 1]).astype(int)),
    )
    df["days_to_holiday"] = np.clip(nearest, 0, 30)
    return df


# ---------------------------------------------------------------------------
# Empirical "level" features -- fit on the PAST only, then looked up
# ---------------------------------------------------------------------------
def fit_levels(train_df):
    """Estimate occupancy levels from training rows (must predate the origin)."""
    g = train_df
    return {
        "global": float(g["booked"].mean()),
        "listing": g.groupby("listing_id")["booked"].mean(),
        "market": g.groupby("market")["booked"].mean(),
        "market_month": g.groupby(["market", "month"])["booked"].mean(),
        "archetype_dow": g.groupby(["archetype", "dayofweek"])["booked"].mean(),
    }


def apply_levels(df, levels):
    gmean = levels["global"]
    df["listing_baseline_occ"] = (
        df["listing_id"].map(levels["listing"]).fillna(gmean))
    df["market_baseline_occ"] = df["market"].map(levels["market"]).fillna(gmean)
    mm = levels["market_month"].rename("market_month_occ").reset_index()
    df = df.merge(mm, on=["market", "month"], how="left")
    df["market_month_occ"] = df["market_month_occ"].fillna(df["market_baseline_occ"])
    ad = levels["archetype_dow"].rename("archetype_dow_occ").reset_index()
    df = df.merge(ad, on=["archetype", "dayofweek"], how="left")
    df["archetype_dow_occ"] = df["archetype_dow_occ"].fillna(df["market_baseline_occ"])
    # placeholder kept for schema stability across versions
    df["market_lead_dummy"] = 0.0
    return df


# ---------------------------------------------------------------------------
# Assemble a model-ready frame
# ---------------------------------------------------------------------------
def prepare_static(listings):
    s = listings.copy()
    s["host_since"] = pd.to_datetime(s["host_since"])
    ref = pd.Timestamp("2026-06-02")
    s["host_tenure_days"] = (ref - s["host_since"]).dt.days
    for c in STATIC_BOOL:
        s[c] = s[c].astype(int)
    # categoricals are left as plain strings here; build_design casts them to
    # the `category` dtype only after the level lookups/merges are done.
    return s


def build_design(calendar, listings_static, levels):
    """Join calendar+listings, add calendar & level features, return frame.

    `calendar` may or may not contain the `booked` column (future frames don't).
    """
    # the calendar may already carry helper columns (market/archetype/month/...)
    # used to fit levels; drop any that the listings table will supply to avoid
    # _x/_y suffix collisions on the join.
    static = listings_static.drop(columns=["host_since"], errors="ignore")
    dup = [c for c in static.columns if c in calendar.columns and c != "listing_id"]
    df = calendar.drop(columns=dup).merge(static, on="listing_id", how="left")
    df = add_calendar_features(df)
    df = apply_levels(df, levels)
    for c in CATEGORICAL:
        df[c] = df[c].astype("category")
    return df
