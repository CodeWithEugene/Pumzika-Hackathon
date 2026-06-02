"""
fetch_real_data.py
===================
Download REAL short-term-rental data from Inside Airbnb (Cape Town, the only
African city they publish) and map it onto the same schema our pipeline uses,
so the occupancy model can be validated on genuine market data.

Source : Inside Airbnb — http://insideairbnb.com  (Cape Town snapshot)
License: Creative Commons Attribution 4.0 International (CC BY 4.0)

Occupancy proxy: Inside Airbnb publishes a forward *availability* calendar.
A night marked `available = f` is taken (booked or host-blocked) — the standard
real-world occupancy proxy used in STR research. We carry that as `booked`.

Outputs:
  data/real_listings.csv   (same columns as the synthetic listings.csv)
  data/real_calendar.csv   (listing_id, date, booked, price)
"""

import os
import io
import gzip
import urllib.request
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
os.makedirs(DATA, exist_ok=True)

BASE = "https://data.insideairbnb.com/south-africa/wc/cape-town/2025-09-28/data"
N_LISTINGS = 5000           # deterministic subsample to keep the backtest snappy
SEED = 42


def fetch_gz_csv(name, **kw):
    url = f"{BASE}/{name}"
    print(f"  downloading {name} …")
    with urllib.request.urlopen(url) as r:
        raw = r.read()
    return pd.read_csv(io.BytesIO(gzip.decompress(raw)), **kw)


def parse_price(s):
    return (s.astype(str).str.replace(r"[$,]", "", regex=True)
            .replace({"nan": np.nan, "": np.nan}).astype(float))


def has(amen, token):
    return amen.str.contains(token, case=False, na=False)


def main():
    rng = np.random.default_rng(SEED)
    print("Fetching Inside Airbnb Cape Town …")
    L = fetch_gz_csv("listings.csv.gz", low_memory=False)

    # deterministic subsample of listings
    keep = rng.choice(L["id"].values, size=min(N_LISTINGS, len(L)), replace=False)
    L = L[L["id"].isin(keep)].copy()

    amen = L["amenities"].fillna("")
    price = parse_price(L["price"])
    price = price.fillna(price.median())
    review = pd.to_numeric(L["review_scores_rating"], errors="coerce")
    review = review.fillna(review.median())
    bedrooms = pd.to_numeric(L["bedrooms"], errors="coerce").fillna(1).clip(1, 10)
    cap = pd.to_numeric(L["accommodates"], errors="coerce").fillna(2).clip(1, 20)
    host_since = pd.to_datetime(L["host_since"], errors="coerce")
    host_since = host_since.fillna(pd.Timestamp("2018-01-01"))

    room = L["room_type"].fillna("Entire home/apt")
    archetype = room.map({
        "Entire home/apt": "entire_home", "Private room": "private_room",
        "Hotel room": "hotel_room", "Shared room": "shared_room",
    }).fillna("entire_home")

    # keep property_type cardinality sane
    ptype = L["property_type"].fillna("Other")
    top = ptype.value_counts().head(18).index
    ptype = ptype.where(ptype.isin(top), "Other")

    listings = pd.DataFrame({
        "listing_id": L["id"].values,
        "market": L["neighbourhood_cleansed"].fillna("Cape Town").values,
        "country": "ZA",
        "archetype": archetype.values,
        "property_type": ptype.values,
        "bedrooms": bedrooms.astype(int).values,
        "capacity": cap.astype(int).values,
        "base_price": price.round(2).values,
        "is_superhost": (L["host_is_superhost"] == "t").values,
        "review_score": review.round(2).clip(0, 5).values,
        "num_reviews": pd.to_numeric(L["number_of_reviews"], errors="coerce")
                         .fillna(0).astype(int).values,
        "instant_book": (L["instant_bookable"] == "t").values,
        "wifi": has(amen, "wifi").values,
        "air_conditioning": has(amen, "air condition").values,
        "pool": has(amen, "pool").values,
        "parking": has(amen, "parking").values,
        "kitchen": has(amen, "kitchen").values,
        "lat": L["latitude"].values,
        "lon": L["longitude"].values,
        "host_since": host_since.dt.date.astype(str).values,
    })

    print(f"  {len(listings)} listings across "
          f"{listings.market.nunique()} neighbourhoods")

    # ---- calendar ----
    cal = fetch_gz_csv("calendar.csv.gz",
                       usecols=["listing_id", "date", "available", "price"],
                       parse_dates=["date"], low_memory=False)
    cal = cal[cal["listing_id"].isin(keep)].copy()
    cal["booked"] = (cal["available"] == "f").astype("int8")
    cal["price"] = parse_price(cal["price"])
    # fill missing calendar price with the listing base price
    base = listings.set_index("listing_id")["base_price"]
    cal["price"] = cal["price"].fillna(cal["listing_id"].map(base))
    cal = cal[["listing_id", "date", "booked", "price"]].dropna(subset=["price"])

    listings.to_csv(os.path.join(DATA, "real_listings.csv"), index=False)
    cal.to_csv(os.path.join(DATA, "real_calendar.csv"), index=False)
    print(f"  {len(cal):,} listing-day rows "
          f"({cal.date.min().date()} -> {cal.date.max().date()})")
    print(f"  occupancy (taken) rate: {cal.booked.mean():.1%}")
    print("Saved data/real_listings.csv and data/real_calendar.csv")


if __name__ == "__main__":
    main()
