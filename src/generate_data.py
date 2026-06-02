"""
generate_data.py
=================
Realistic synthetic data generator for the Pumzika Occupancy & Demand
Forecasting challenge (Track 02).

WHY SYNTHETIC?  Pumzika's real booking history is private and no public API
exists, so we *source our own data* (as the challenge permits). Rather than
draw random noise, we simulate a short-term-rental (STR) marketplace whose
behaviour is grounded in documented East-African tourism dynamics:

  * Safari markets (Serengeti, Maasai Mara) peak Jul-Oct with the Great
    Migration, plus a Dec-Feb green-season bump.
  * Coastal markets (Zanzibar, Diani, Mombasa) peak over Dec-Mar and again
    Jul-Aug; they crater during the Apr-May "long rains".
  * City / business markets (Dar es Salaam, Nairobi, Kampala, Entebbe) are far
    flatter, driven by weekday business travel.
  * Demand spikes around Christmas / New Year, Easter and the two Eids.
  * Prices move WITH demand (high season => higher nightly rate AND higher
    occupancy), which is exactly the confound a naive model gets wrong.
  * A gentle platform-growth trend reflects a young marketplace gaining traction.

The generative process is an explicit logistic booking model, so every driver
is documented and the ground truth is known -- which lets us prove the
forecaster recovers real structure rather than memorising noise.

Outputs:
  data/listings.csv   one row per property
  data/calendar.csv   one row per (listing, date): booked flag + nightly price
"""

import os
import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# History window: 2 years of daily calendar ending the day before "today".
TODAY = pd.Timestamp("2026-06-02")          # matches challenge "current date"
HISTORY_START = pd.Timestamp("2024-06-01")
HISTORY_END = TODAY - pd.Timedelta(days=1)  # 2026-06-01 inclusive
N_LISTINGS = 500

# ---------------------------------------------------------------------------
# 1. Market definitions
# ---------------------------------------------------------------------------
# archetype drives the seasonal & weekly shape; base_occ is the long-run mean
# occupancy the market gravitates to; price_usd is a typical nightly base rate.
MARKETS = [
    # name,            country, archetype, base_occ, base_price, lat,     lon,    weight
    ("Zanzibar",       "TZ", "beach",   0.62, 85,  -6.165, 39.199, 0.16),
    ("Serengeti",      "TZ", "safari",  0.55, 220, -2.333, 34.833, 0.06),
    ("Arusha",         "TZ", "safari",  0.58, 70,  -3.387, 36.683, 0.08),
    ("Dar es Salaam",  "TZ", "city",    0.60, 55,  -6.792, 39.208, 0.14),
    ("Nairobi",        "KE", "city",    0.64, 60,  -1.286, 36.817, 0.16),
    ("Diani Beach",    "KE", "beach",   0.60, 95,  -4.278, 39.591, 0.07),
    ("Mombasa",        "KE", "beach",   0.58, 70,  -4.043, 39.668, 0.08),
    ("Maasai Mara",    "KE", "safari",  0.54, 240, -1.500, 35.143, 0.05),
    ("Kampala",        "UG", "city",    0.57, 45,   0.347, 32.582, 0.09),
    ("Entebbe",        "UG", "city",    0.59, 50,   0.051, 32.463, 0.05),
]
MARKETS = pd.DataFrame(
    MARKETS,
    columns=["market", "country", "archetype", "base_occ", "base_price",
             "lat", "lon", "weight"],
)
MARKETS["weight"] /= MARKETS["weight"].sum()

# Monthly seasonal multipliers (Jan..Dec) per archetype, smoothed to daily later.
SEASON = {
    "beach":  [1.35, 1.30, 1.05, 0.55, 0.50, 0.75, 1.10, 1.20, 0.95, 0.90, 0.80, 1.45],
    "safari": [1.05, 1.10, 0.70, 0.40, 0.45, 0.95, 1.45, 1.50, 1.40, 1.20, 0.75, 1.10],
    "city":   [0.95, 1.00, 1.02, 0.90, 0.95, 1.00, 1.05, 1.05, 1.06, 1.05, 1.00, 1.00],
}

# Weekly multipliers (Mon=0 .. Sun=6).
WEEKDAY = {
    "beach":  [0.85, 0.85, 0.90, 1.00, 1.25, 1.30, 1.05],
    "safari": [0.95, 0.95, 0.98, 1.02, 1.10, 1.12, 1.00],
    "city":   [1.05, 1.08, 1.10, 1.10, 1.05, 0.85, 0.80],
}

PROPERTY_TYPES = {
    "beach":  ["beach_house", "villa", "bungalow", "apartment", "cottage"],
    "safari": ["safari_tent", "lodge_room", "cottage", "villa"],
    "city":   ["apartment", "studio", "townhouse", "villa"],
}

# Public-holiday / event windows -> demand multiplier. Dates approximate the
# real calendar (Easter & Eid shift yearly).
def holiday_multipliers():
    spans = []
    for y in (2024, 2025, 2026, 2027):
        spans.append((f"{y}-12-20", f"{y+1}-01-03", 1.45))   # Christmas / New Year
    spans += [
        ("2024-03-29", "2024-04-01", 1.25),  # Easter 2024
        ("2025-04-18", "2025-04-21", 1.25),  # Easter 2025
        ("2026-04-03", "2026-04-06", 1.25),  # Easter 2026
        ("2024-04-09", "2024-04-11", 1.20),  # Eid al-Fitr 2024
        ("2025-03-30", "2025-04-01", 1.20),  # Eid al-Fitr 2025
        ("2026-03-20", "2026-03-22", 1.20),  # Eid al-Fitr 2026
        ("2024-06-15", "2024-06-17", 1.18),  # Eid al-Adha 2024
        ("2025-06-06", "2025-06-08", 1.18),  # Eid al-Adha 2025
        ("2026-05-26", "2026-05-28", 1.18),  # Eid al-Adha 2026
        ("2026-07-10", "2026-07-14", 1.15),  # Sauti za Busara / festival proxy
    ]
    idx = {}
    for start, end, mult in spans:
        for d in pd.date_range(start, end):
            idx[d.normalize()] = max(idx.get(d.normalize(), 1.0), mult)
    return idx

HOLIDAYS = holiday_multipliers()


def daily_season_curve(archetype, dates):
    """Smoothly interpolate monthly multipliers onto each calendar date."""
    month_mid = np.array([15, 45, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349])
    vals = np.array(SEASON[archetype])
    # wrap for smooth Dec->Jan transition
    xp = np.concatenate(([month_mid[-1] - 365], month_mid, [month_mid[0] + 365]))
    fp = np.concatenate(([vals[-1]], vals, [vals[0]]))
    doy = dates.dayofyear.to_numpy()
    return np.interp(doy, xp, fp)


# ---------------------------------------------------------------------------
# 2. Build listings
# ---------------------------------------------------------------------------
def build_listings():
    n_per = rng.multinomial(N_LISTINGS, MARKETS["weight"].to_numpy())
    rows = []
    lid = 1000
    for (_, m), n in zip(MARKETS.iterrows(), n_per):
        for _ in range(n):
            ptype = rng.choice(PROPERTY_TYPES[m.archetype])
            bedrooms = int(np.clip(rng.poisson(1.6) + 1, 1, 6))
            capacity = bedrooms * 2 + int(rng.integers(0, 3))
            # quality
            is_superhost = rng.random() < 0.28
            review_score = float(np.clip(
                rng.normal(4.55 if is_superhost else 4.25, 0.28), 3.2, 5.0))
            num_reviews = int(np.clip(rng.gamma(2.0, 22), 0, 600))
            instant_book = rng.random() < 0.55
            # amenities
            wifi = rng.random() < 0.9
            ac = rng.random() < (0.85 if m.archetype == "beach" else 0.5)
            pool = rng.random() < (0.45 if m.archetype != "city" else 0.18)
            parking = rng.random() < 0.7
            kitchen = rng.random() < 0.8
            # listing-level base price varies around market base
            base_price = float(round(
                m.base_price * (0.55 + 0.5 * bedrooms / 2)
                * rng.lognormal(0, 0.18), 2))
            # static popularity multiplier (latent desirability of this listing)
            desirability = float(rng.normal(0, 0.45))
            host_since = HISTORY_START - pd.Timedelta(days=int(rng.integers(60, 1500)))
            rows.append(dict(
                listing_id=lid, market=m.market, country=m.country,
                archetype=m.archetype, property_type=ptype, bedrooms=bedrooms,
                capacity=capacity, base_price=base_price,
                is_superhost=is_superhost, review_score=round(review_score, 2),
                num_reviews=num_reviews, instant_book=instant_book,
                wifi=wifi, air_conditioning=ac, pool=pool, parking=parking,
                kitchen=kitchen, lat=round(m.lat + rng.normal(0, 0.05), 4),
                lon=round(m.lon + rng.normal(0, 0.05), 4),
                host_since=host_since.date().isoformat(),
                _base_occ=m.base_occ, _desirability=desirability,
            ))
            lid += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Build daily calendar via an explicit logistic booking model
# ---------------------------------------------------------------------------
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def build_calendar(listings):
    dates = pd.date_range(HISTORY_START, HISTORY_END, freq="D")
    n_days = len(dates)
    doy = dates  # for season curve
    weekday = dates.weekday.to_numpy()
    hol_mult = np.array([HOLIDAYS.get(d.normalize(), 1.0) for d in dates])

    # platform growth: occupancy logit drifts up ~ +0.35 over 2 years
    t = np.linspace(0, 1, n_days)
    growth = 0.35 * t

    frames = []
    for _, L in listings.iterrows():
        season = daily_season_curve(L.archetype, doy)
        wd = np.array(WEEKDAY[L.archetype])[weekday]

        # ---- dynamic nightly price: moves with season / weekend / holiday ----
        # Owners raise rates in peak season, but demand outpaces the hike, so
        # occupancy still climbs (true of destination markets that sell out).
        price = (L.base_price
                 * (0.90 + 0.20 * season)         # seasonal pricing power
                 * (1 + 0.10 * (wd - 1))          # weekend premium
                 * hol_mult                        # holiday premium
                 * rng.lognormal(0, 0.05, n_days)) # day noise
        price = np.round(price, 2)
        price_ratio = price / L.base_price          # relative to own baseline

        # ---- latent occupancy probability (the ground-truth DGP) ----
        z = logit(L._base_occ)
        z = z + 1.25 * np.log(np.clip(season, 0.2, None))   # seasonality
        z = z + np.log(np.clip(wd, 0.2, None))              # weekday
        z = z + 0.55 * np.log(hol_mult)                     # holidays
        z = z + L._desirability                             # listing desirability
        z = z + 0.45 * (L.review_score - 4.3)               # quality
        z = z + 0.20 * L.is_superhost
        z = z + 0.10 * L.instant_book
        z = z + 0.06 * L.pool + 0.05 * L.air_conditioning
        z = z + growth                                      # platform growth
        # price elasticity: demand falls when price runs above own baseline
        z = z - 1.30 * (price_ratio - 1.0)
        # idiosyncratic day-to-day noise (unpredictable component)
        z = z + rng.normal(0, 0.45, n_days)

        p = 1 / (1 + np.exp(-z))
        booked = (rng.random(n_days) < p).astype(np.int8)

        frames.append(pd.DataFrame(dict(
            listing_id=L.listing_id,
            date=dates,
            booked=booked,
            price=price,
        )))
    cal = pd.concat(frames, ignore_index=True)
    return cal


def main():
    print("Generating listings ...")
    listings = build_listings()
    print(f"  {len(listings)} listings across {listings.market.nunique()} markets")

    print("Generating daily calendar (this builds the booking ground truth) ...")
    cal = build_calendar(listings)
    print(f"  {len(cal):,} listing-day rows "
          f"({cal.date.min().date()} -> {cal.date.max().date()})")
    print(f"  overall historical occupancy: {cal.booked.mean():.1%}")

    # drop the internal latent columns before saving listings
    listings_out = listings.drop(columns=["_base_occ", "_desirability"])
    listings_out.to_csv(os.path.join(DATA_DIR, "listings.csv"), index=False)
    cal.to_csv(os.path.join(DATA_DIR, "calendar.csv"), index=False)
    print("Saved data/listings.csv and data/calendar.csv")


if __name__ == "__main__":
    main()
