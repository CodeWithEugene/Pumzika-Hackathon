"""
Pumzika - Occupancy & Demand Forecasting dashboard
==================================================
A host-facing "revenue planner": forecast occupancy 90 days out, spot peaks
and soft spots early, and get plain-language planning actions.

Run:  streamlit run app/dashboard.py
"""

import os
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
REPORTS = os.path.join(ROOT, "reports")
DATA = os.path.join(ROOT, "data")
FIG = os.path.join(REPORTS, "figures")

TEAL = "#0E8388"
DARK = "#13293D"
CORAL = "#FF6B57"
SAND = "#E8B84B"
GREEN = "#2BA84A"
PALETTE = [TEAL, CORAL, SAND, "#5B8C5A", "#9B5DE5", "#00A6A6",
           "#E07A5F", "#3D5A80", "#8338EC", "#F15BB5"]

st.set_page_config(page_title="Pumzika · Demand Forecasting",
                   page_icon="🏝️", layout="wide",
                   initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
  .stApp {{ background: #FBFCFC; }}
  #MainMenu, footer {{ visibility: hidden; }}
  .block-container {{ padding-top: 1.6rem; max-width: 1300px; }}
  h1,h2,h3,h4 {{ color: {DARK}; font-weight: 700; letter-spacing:-.01em; }}
  .hero {{
     background: linear-gradient(120deg,{DARK} 0%, #0E5C60 55%, {TEAL} 100%);
     color:#fff; padding: 26px 30px; border-radius: 18px; margin-bottom: 18px;
     box-shadow: 0 10px 30px rgba(14,131,136,.18); }}
  .hero h1 {{ color:#fff; margin:0; font-size: 2.0rem; }}
  .hero p {{ color:#D6EDED; margin:.35rem 0 0; font-size:1.02rem; }}
  .pill {{ display:inline-block; background:rgba(255,255,255,.16);
     padding:3px 12px; border-radius:999px; font-size:.78rem; margin-right:6px;}}
  .kpi {{ background:#fff; border:1px solid #E6ECEC; border-radius:14px;
     padding:16px 18px; box-shadow:0 2px 10px rgba(19,41,61,.04); height:100%;}}
  .kpi .lab {{ color:#6B7B7B; font-size:.80rem; text-transform:uppercase;
     letter-spacing:.04em; font-weight:600; }}
  .kpi .val {{ color:{DARK}; font-size:1.85rem; font-weight:800; line-height:1.1;}}
  .kpi .sub {{ color:{TEAL}; font-size:.82rem; font-weight:600; }}
  .reccard {{ background:linear-gradient(135deg,#FFF6F2,#FFFFFF);
     border-left:5px solid {CORAL}; border-radius:12px; padding:14px 18px;
     font-size:1.02rem; color:{DARK}; }}
  .src {{ color:#8A9A9A; font-size:.78rem; }}
  div[data-testid="stMetricValue"] {{ font-size:1.6rem; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@st.cache_data
def load():
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
    return listings, fc_m, fc_l, plan, metrics, meta, imp, hist


listings, fc_m, fc_l, plan, metrics, meta, imp, hist = load()
hist_lookup = listings[["listing_id", "market"]]
hist_m = (hist.merge(hist_lookup, on="listing_id")
          .groupby(["market", "date"]).booked.mean().reset_index())

NICE_NAME = {
    "listing_baseline_occ": "Listing's own track record",
    "market_month_occ": "Market seasonality (month)",
    "archetype_dow_occ": "Day-of-week pattern",
    "doy_sin": "Season (annual cycle)", "doy_cos": "Season (annual cycle)",
    "review_score": "Review score", "num_reviews": "Number of reviews",
    "host_tenure_days": "Host tenure", "dayofyear": "Time of year",
    "base_price": "Price tier", "is_weekend": "Weekend",
    "holiday_mult": "Holiday / event", "days_to_holiday": "Days to holiday",
    "capacity": "Capacity", "bedrooms": "Bedrooms",
    "market_baseline_occ": "Market baseline", "lat": "Latitude",
    "lon": "Longitude", "day": "Day of month", "dow_sin": "Day-of-week pattern",
}

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
imp_pct = metrics["headline"]["improvement_pct"]
st.markdown(f"""
<div class="hero">
  <span class="pill">🇹🇿 Tanzania</span><span class="pill">🇰🇪 Kenya</span>
  <span class="pill">🇺🇬 Uganda</span><span class="pill">Track 02 · Occupancy & Demand</span>
  <h1>Pumzika Demand Radar</h1>
  <p>A 90-day occupancy forecast for every host &mdash; know your peaks and soft
  spots before they arrive. Beats the seasonal-average baseline by
  <b>{imp_pct:.0f}%</b> in held-out back-testing.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
def kpi(col, label, value, sub):
    col.markdown(f'<div class="kpi"><div class="lab">{label}</div>'
                 f'<div class="val">{value}</div><div class="sub">{sub}</div></div>',
                 unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
kpi(c1, "Portfolio occupancy · next 90d",
    f'{meta["portfolio_avg_occ_90d"]*100:.0f}%', "forecast mean")
kpi(c2, "Listings covered", f'{meta["n_listings"]:,}',
    f'{meta["n_markets"]} markets')
kpi(c3, "Forecast error (MAE)",
    f'{metrics["headline"]["model_market_week_MAE"]*100:.1f} pts',
    "market-week occupancy")
kpi(c4, "Lift vs seasonal-naive", f'{imp_pct:.0f}%', "lower error")
kpi(c5, "Horizon", f'{meta["horizon_days"]} days',
    f'from {meta["origin"]}')
st.write("")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_outlook, tab_listing, tab_drivers, tab_trust = st.tabs(
    ["📈  Market outlook", "🏠  Listing planner",
     "🧭  Demand drivers", "✅  Model & trust"])

# ===== TAB 1: market outlook ===============================================
with tab_outlook:
    st.subheader("Where demand is heading")
    markets = sorted(fc_m["market"].unique())
    default = ["Serengeti", "Zanzibar", "Nairobi", "Maasai Mara"]
    sel = st.multiselect("Markets", markets,
                         default=[m for m in default if m in markets])
    if not sel:
        sel = markets[:4]
    smooth = st.toggle("Smooth (7-day rolling)", value=True)

    fig = go.Figure()
    for i, m in enumerate(sel):
        s = fc_m[fc_m.market == m].sort_values("date")
        y = s.occ_forecast.rolling(7, min_periods=1, center=True).mean() if smooth \
            else s.occ_forecast
        fig.add_trace(go.Scatter(
            x=s.date, y=y, name=m, mode="lines",
            line=dict(color=PALETTE[i % len(PALETTE)], width=3),
            hovertemplate=f"<b>{m}</b><br>%{{x|%b %d}}<br>"
                          "Occupancy %{y:.0%}<extra></extra>"))
    fig.update_layout(
        height=430, margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(title="Forecast occupancy", tickformat=".0%",
                   range=[0, 1], gridcolor="#ECECEC"),
        xaxis=dict(gridcolor="#F3F3F3"),
        legend=dict(orientation="h", y=1.08), plot_bgcolor="white",
        hovermode="x unified", font=dict(color=DARK))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Occupancy heatmap · market × week")
    hm = fc_m.copy()
    hm["week"] = hm["date"].dt.to_period("W").apply(lambda p: p.start_time)
    pivot = (hm.groupby(["market", "week"]).occ_forecast.mean()
             .unstack().sort_index())
    order = pivot.mean(axis=1).sort_values(ascending=False).index
    pivot = pivot.loc[order]
    heat = go.Figure(go.Heatmap(
        z=pivot.values, x=[d.strftime("%b %d") for d in pivot.columns],
        y=pivot.index, colorscale="Teal", zmin=0.4, zmax=0.9,
        colorbar=dict(title="Occ", tickformat=".0%"),
        hovertemplate="%{y}<br>week of %{x}<br>%{z:.0%}<extra></extra>"))
    heat.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10),
                       font=dict(color=DARK), plot_bgcolor="white")
    st.plotly_chart(heat, width="stretch")
    st.caption("Darker = busier. Safari markets light up into the Jul–Aug "
               "Great Migration peak; coastal demand softens after the long rains.")

# ===== TAB 2: listing planner ==============================================
with tab_listing:
    left, right = st.columns([1, 2.4])
    with left:
        st.subheader("Pick a listing")
        mk = st.selectbox("Market", sorted(listings.market.unique()))
        sub = plan[plan.market == mk].sort_values("avg_occ_90d", ascending=False)
        def label(r):
            return (f"#{r.listing_id} · {r.archetype} · "
                    f"{r.avg_occ_90d:.0%} occ")
        choice = st.selectbox("Listing", sub.listing_id.tolist(),
                              format_func=lambda i: label(
                                  sub[sub.listing_id == i].iloc[0]))
        row = sub[sub.listing_id == choice].iloc[0]
        st.metric("Avg occupancy · next 90 days", f"{row.avg_occ_90d:.0%}")
        m1, m2 = st.columns(2)
        m1.metric("Peak week", row.peak_week, f"{row.peak_occ:.0%}")
        m2.metric("Soft week", row.low_week, f"{row.low_occ:.0%}",
                  delta_color="inverse")
        st.markdown(f'<div class="reccard">💡 <b>Plan:</b> {row.recommendation}'
                    f'</div>', unsafe_allow_html=True)
        info = listings[listings.listing_id == choice].iloc[0]
        st.caption(f"⭐ {info.review_score} · {int(info.num_reviews)} reviews · "
                   f"{'Superhost · ' if info.is_superhost else ''}"
                   f"base ${info.base_price:.0f}/night · {info.bedrooms} bdr")

    with right:
        st.subheader(f"90-day forecast · listing #{choice}")
        s = fc_l[fc_l.listing_id == choice].sort_values("date")
        h = hist[hist.listing_id == choice].sort_values("date")
        h = h[h.date >= pd.Timestamp(meta["origin"]) - pd.Timedelta(days=120)]
        hroll = h.set_index("date").booked.rolling(14, min_periods=3).mean()
        sroll = s.set_index("date").occ_forecast.rolling(7, min_periods=1,
                                                         center=True).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hroll.index, y=hroll.values,
                      name="Actual (trailing)", mode="lines",
                      line=dict(color="#9AA8A8", width=2, dash="dot")))
        fig.add_trace(go.Scatter(x=sroll.index, y=sroll.values,
                      name="Forecast", mode="lines",
                      line=dict(color=TEAL, width=3.5)))
        origin_ts = pd.Timestamp(meta["origin"])
        fig.add_shape(type="line", x0=origin_ts, x1=origin_ts, y0=0, y1=1,
                      yref="paper", line=dict(color=CORAL, dash="dash"))
        fig.add_annotation(x=origin_ts, y=1.0, yref="paper", text="today",
                           showarrow=False, yshift=10,
                           font=dict(color=CORAL, size=11))
        pk = pd.Timestamp(row.peak_week)
        fig.add_trace(go.Scatter(x=[pk], y=[row.peak_occ], mode="markers+text",
                      marker=dict(color=GREEN, size=12, symbol="star"),
                      text=["peak"], textposition="top center",
                      showlegend=False))
        fig.update_layout(height=460, margin=dict(l=10, r=10, t=10, b=10),
                          yaxis=dict(title="Occupancy", tickformat=".0%",
                                     range=[0, 1], gridcolor="#ECECEC"),
                          xaxis=dict(gridcolor="#F6F6F6"),
                          legend=dict(orientation="h", y=1.1),
                          plot_bgcolor="white", hovermode="x unified",
                          font=dict(color=DARK))
        st.plotly_chart(fig, width="stretch")

    st.subheader("Action list · busiest & softest weeks across the portfolio")
    colA, colB = st.columns(2)
    busy = plan.sort_values("peak_occ", ascending=False).head(8)[
        ["listing_id", "market", "peak_week", "peak_occ", "recommendation"]]
    soft = plan.sort_values("low_occ").head(8)[
        ["listing_id", "market", "low_week", "low_occ", "recommendation"]]
    colA.markdown("**🔥 Raise rates here**")
    colA.dataframe(busy.assign(peak_occ=(busy.peak_occ*100).round(0)),
                   hide_index=True, width="stretch")
    colB.markdown("**🧊 Fill these gaps**")
    colB.dataframe(soft.assign(low_occ=(soft.low_occ*100).round(0)),
                   hide_index=True, width="stretch")

# ===== TAB 3: demand drivers ===============================================
with tab_drivers:
    st.subheader("What the model watches")
    imp2 = imp.copy()
    imp2["name"] = imp2.feature.map(NICE_NAME).fillna(imp2.feature)
    imp2 = imp2.groupby("name", as_index=False).gain.sum()
    imp2 = imp2.sort_values("gain", ascending=True).tail(12)
    fig = go.Figure(go.Bar(
        x=imp2.gain, y=imp2.name, orientation="h",
        marker=dict(color=imp2.gain, colorscale="Teal")))
    fig.update_layout(height=440, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis=dict(title="Importance (model gain)",
                                 gridcolor="#ECECEC"),
                      plot_bgcolor="white", font=dict(color=DARK))
    st.plotly_chart(fig, width="stretch")
    st.markdown(
        "The forecaster leans on **a listing's own track record**, **market "
        "seasonality**, the **annual season cycle**, and **review quality** — "
        "the same signals an experienced host uses, quantified and projected "
        "forward. Price is deliberately left out so the output is a clean "
        "*demand* forecast that hands off to the Dynamic-Pricing track.")

    st.subheader("Seasonality, learned from history")
    hm = hist_m.copy()
    hm["month"] = hm["date"].dt.month
    arche = listings[["market", "archetype"]].drop_duplicates()
    hm = hm.merge(arche, on="market")
    season = hm.groupby(["archetype", "month"]).booked.mean().reset_index()
    figs = go.Figure()
    for i, a in enumerate(sorted(season.archetype.unique())):
        s = season[season.archetype == a]
        figs.add_trace(go.Scatter(x=s.month, y=s.booked, name=a.title(),
                       mode="lines+markers",
                       line=dict(color=PALETTE[i], width=3)))
    figs.update_layout(
        height=360, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(title="Month", tickmode="array", tickvals=list(range(1, 13)),
                   ticktext=["J","F","M","A","M","J","J","A","S","O","N","D"]),
        yaxis=dict(title="Occupancy", tickformat=".0%", gridcolor="#ECECEC"),
        legend=dict(orientation="h", y=1.1), plot_bgcolor="white",
        font=dict(color=DARK))
    st.plotly_chart(figs, width="stretch")
    st.caption("Safari peaks Jul–Oct (Great Migration) + a Dec–Feb bump; "
               "coastal peaks Dec–Mar and dips in the Apr–May long rains; "
               "city demand stays flat — driven by business travel.")

# ===== TAB 4: model & trust ================================================
with tab_trust:
    st.subheader("Honest, held-out back-testing")
    st.markdown(
        f"Validated with **{metrics['n_folds']} rolling-origin folds**: at each "
        "origin the model sees only the past and forecasts the next "
        f"**{metrics['horizon_days']} days** — a true *plan-ahead* test, no leakage.")
    rows = []
    for name, v in metrics["summary"].items():
        rows.append(dict(Model=name, AUC=round(v["AUC"], 3),
                         **{"Occ-rate MAE (mkt-wk)": round(v["market_week_MAE"], 4),
                            "Occ-rate MAE (listing-mo)": round(v["listing_month_MAE"], 4)}))
    tbl = pd.DataFrame(rows)
    best = tbl["Occ-rate MAE (mkt-wk)"].min()
    st.dataframe(
        tbl.style.format(precision=4).apply(
            lambda s: ["background-color:#E6F4F1;font-weight:700"
                       if s["Model"] == "LightGBM" else "" for _ in s], axis=1),
        hide_index=True, width="stretch")
    st.caption("Lower MAE = better. LightGBM wins every metric — it fuses each "
               "listing's track record *and* seasonality *and* quality, which no "
               "single baseline does.")

    c1, c2 = st.columns(2)
    fva = os.path.join(FIG, "forecast_vs_actual.png")
    cal = os.path.join(FIG, "calibration.png")
    if os.path.exists(fva):
        c1.image(fva, caption="Forecast tracks actual across market types.")
    if os.path.exists(cal):
        c2.image(cal, caption="Well-calibrated probabilities (held-out).")

    st.markdown("---")
    st.markdown(
        "<span class='src'><b>Data note.</b> Pumzika's live booking history is "
        "private, so this entry sources its own data: a simulator grounded in "
        "documented East-African tourism seasonality (Great Migration, coastal "
        "long-rains, Eid/Christmas demand). The pipeline is dataset-agnostic — "
        "point it at real Pumzika exports (same schema) and it runs unchanged."
        "</span>", unsafe_allow_html=True)

st.markdown(
    f"<div class='src' style='text-align:center;margin-top:14px'>"
    f"Pumzika Hackathon 2026 · Track 02 · Occupancy & Demand Forecasting · "
    f"forecast origin {meta['origin']}</div>", unsafe_allow_html=True)
