"""
fetch_kaggle_hotel.py
====================
Download the official Hotel Booking Demand dataset (Antonio, Almeida, Nunes
2019) and transform booking records into a daily occupancy time series per
hotel, ready for the forecasting pipeline.

Out: data/kaggle_hotel.csv  —  daily occupancy, ADR, lead time, segment mix
     data/kaggle_bookings.csv  —  cleaned booking records for reference

Dataset: https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand
License: CC0 (public domain)
"""

import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

URL = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2020/2020-02-11/hotels.csv"

HOTELS_CSV = os.path.join(DATA, "kaggle_bookings.csv")
OCC_CSV = os.path.join(DATA, "kaggle_occupancy.csv")


def download():
    if os.path.exists(HOTELS_CSV):
        print(f"Using cached {HOTELS_CSV}")
        return pd.read_csv(HOTELS_CSV)
    print(f"Downloading from {URL} …")
    df = pd.read_csv(URL)
    df.to_csv(HOTELS_CSV, index=False)
    print(f"Saved {len(df):,} rows → {HOTELS_CSV}")
    return df


def clean(df):
    """Parse dates, filter usable bookings, add derived columns."""
    month_map = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12,
    }
    df = df.copy()
    df["arrival_month"] = df["arrival_date_month"].map(month_map)
    df["arrival_date"] = pd.to_datetime(
        df["arrival_date_year"].astype(str) + "-" +
        df["arrival_month"].astype(str) + "-" +
        df["arrival_date_day_of_month"].astype(str)
    )
    df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]
    df["checkout_date"] = df["arrival_date"] + pd.to_timedelta(df["total_nights"], unit="D")
    df["adr"] = df["adr"].clip(lower=0)  # drop negative ADR
    return df


def build_daily_occupancy(df):
    """Expand non-canceled bookings into daily room-nights per hotel.

    Returns a DataFrame with columns:
        hotel, date, occupied_rooms, avg_adr, avg_lead_time,
        market_segment_TA, market_segment_TO, etc.
    """
    active = df[df["is_canceled"] == 0].copy()
    active = active[active["total_nights"] > 0].copy()
    print(f"Active bookings: {len(active):,}")

    rows = []
    for _, r in active.iterrows():
        d = r["arrival_date"]
        end = r["checkout_date"]
        while d < end:
            rows.append({
                "hotel": r["hotel"],
                "date": d,
                "adr": r["adr"],
                "lead_time": r["lead_time"],
                "market_segment": r["market_segment"],
                "customer_type": r["customer_type"],
                "adults": r["adults"],
            })
            d += pd.Timedelta(days=1)

    detail = pd.DataFrame(rows)
    print(f"Expanded room-nights: {len(detail):,}")

    # Aggregate to daily per hotel
    daily = detail.groupby(["hotel", "date"]).agg(
        occupied_rooms=("adr", "count"),
        avg_adr=("adr", "mean"),
        avg_lead_time=("lead_time", "mean"),
        avg_adults=("adults", "mean"),
    ).reset_index()

    # Market segment mix: fraction from each top segment
    seg_pivot = detail.groupby(["hotel", "date", "market_segment"]).size().unstack(fill_value=0)
    for seg in ["TA", "TO", "Direct", "Corporate"]:
        if seg not in seg_pivot.columns:
            daily[f"seg_{seg}"] = 0.0
        else:
            aligned = seg_pivot[seg].reindex(pd.MultiIndex.from_frame(daily[["hotel", "date"]])).fillna(0).values
            daily[f"seg_{seg}"] = aligned / daily["occupied_rooms"].values

    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values(["hotel", "date"]).reset_index(drop=True)

    # Estimate capacity per hotel (95th percentile of occupied rooms)
    for h in daily["hotel"].unique():
        mask = daily["hotel"] == h
        cap = daily.loc[mask, "occupied_rooms"].quantile(0.95)
        daily.loc[mask, "capacity"] = cap

    daily["occupancy_rate"] = daily["occupied_rooms"] / daily["capacity"]
    daily["occupancy_rate"] = daily["occupancy_rate"].clip(upper=1.0)

    print(f"\nDaily occupancy records: {len(daily):,}")
    for h in daily["hotel"].unique():
        sub = daily[daily["hotel"] == h]
        print(f"  {h}: {len(sub)} days, "
              f"capacity={int(sub['capacity'].iloc[0])}, "
              f"mean occ={sub['occupancy_rate'].mean():.1%}")

    return daily


def add_time_features(df):
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["dayofweek"] = df["date"].dt.dayofweek
    df["dayofyear"] = df["date"].dt.dayofyear
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["date"].dt.quarter
    df["is_weekend"] = (df["date"].dt.dayofweek >= 4).astype(int)
    # Fourier features
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    return df


def add_lag_features(df):
    """Add lagged occupancy rates and rolling means per hotel."""
    df = df.sort_values(["hotel", "date"]).copy()
    for h in df["hotel"].unique():
        mask = df["hotel"] == h
        o = df.loc[mask, "occupancy_rate"].copy()
        adr = df.loc[mask, "avg_adr"].copy()
        for lag in [7, 14, 28]:
            df.loc[mask, f"occ_lag_{lag}"] = o.shift(lag)
            if lag <= 28:
                df.loc[mask, f"adr_lag_{lag}"] = adr.shift(lag)
        for win in [7, 28]:
            df.loc[mask, f"occ_roll_{win}"] = o.shift(1).rolling(win, min_periods=1).mean()
    return df


if __name__ == "__main__":
    raw = download()
    cleaned = clean(raw)
    daily = build_daily_occupancy(cleaned)
    daily = add_time_features(daily)
    daily = add_lag_features(daily)
    daily.to_csv(OCC_CSV, index=False)
    print(f"\nWrote {len(daily):,} rows → {OCC_CSV}")
