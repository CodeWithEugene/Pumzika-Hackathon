"use client";

import { useEffect, useMemo, useState } from "react";
import {
  ResponsiveContainer, LineChart, Line, AreaChart, Area, XAxis, YAxis,
  CartesianGrid, Tooltip, ReferenceLine, Legend,
} from "recharts";
import {
  fmtPct, fmtDate, fmtDateLong, smooth, heatColor, heatText, weekly,
} from "./lib";

const LINE_COLORS = [
  "#0e7c80", "#c2603c", "#d39a3c", "#6f8a64", "#8a5a9e",
  "#0a4a4d", "#b87a1d", "#3d6b8e", "#a23b3b", "#557a55",
];

async function getJSON(name) {
  const res = await fetch(`./data/${name}`);
  return res.json();
}

const TAB_SLUGS = ["outlook", "planner", "drivers", "trust"];

export default function Page() {
  const [d, setD] = useState(null);
  const [tab, setTab] = useState(0);

  // deep-linkable tabs via URL hash (#planner, #drivers, …)
  useEffect(() => {
    const apply = () => {
      const i = TAB_SLUGS.indexOf(window.location.hash.replace("#", ""));
      if (i >= 0) setTab(i);
    };
    apply();
    window.addEventListener("hashchange", apply);
    return () => window.removeEventListener("hashchange", apply);
  }, []);

  const goTab = (i) => {
    setTab(i);
    if (typeof window !== "undefined") window.location.hash = TAB_SLUGS[i];
  };

  useEffect(() => {
    Promise.all([
      getJSON("meta.json"), getJSON("markets.json"), getJSON("listings.json"),
      getJSON("listing_series.json"), getJSON("importance.json"),
      getJSON("season.json"),
    ]).then(([meta, markets, listings, series, importance, season]) =>
      setD({ meta, markets, listings, series, importance, season })
    );
  }, []);

  if (!d) return <Loading />;

  return (
    <div className="wrap">
      <TopBar />
      <Hero meta={d.meta} />
      <Kpis meta={d.meta} />

      <nav className="tabs">
        {["Market outlook", "Listing planner", "Demand drivers", "Model & trust"]
          .map((t, i) => (
            <button key={t} className={`tab ${tab === i ? "active" : ""}`}
              onClick={() => goTab(i)}>
              {["📈", "🏠", "🧭", "✅"][i]} {t}
            </button>
          ))}
      </nav>

      {tab === 0 && <MarketOutlook markets={d.markets} />}
      {tab === 1 && <ListingPlanner data={d} />}
      {tab === 2 && <DemandDrivers importance={d.importance} season={d.season} />}
      {tab === 3 && <ModelTrust meta={d.meta} />}

      <Footer meta={d.meta} />
    </div>
  );
}

/* ----------------------------------------------------------------- */
function Loading() {
  return (
    <div className="wrap" style={{ paddingTop: 120, textAlign: "center" }}>
      <div className="brand" style={{ justifyContent: "center" }}>
        <div className="mark">P</div>
      </div>
      <p style={{ color: "var(--muted)", marginTop: 18 }}>Loading forecast…</p>
    </div>
  );
}

function TopBar() {
  return (
    <header className="topbar rise">
      <div className="brand">
        <div className="mark">P</div>
        <div>
          <b>Pumzika</b>
          <span>Demand Radar</span>
        </div>
      </div>
      <div className="flags">
        <span className="chip">🇹🇿 TZ</span>
        <span className="chip">🇰🇪 KE</span>
        <span className="chip">🇺🇬 UG</span>
      </div>
    </header>
  );
}

function Hero({ meta }) {
  return (
    <section className="hero rise d1">
      <div className="eyebrow">Pumzika Hackathon 2026 · Track 02</div>
      <h1>Know your <em>peaks</em> before they arrive.</h1>
      <p>
        A 90-day occupancy forecast for every host across East Africa — so owners
        can price, staff and promote ahead of demand instead of reacting after
        the fact.
      </p>
      <div className="liftbadge">
        <b>{meta.headline.improvement_pct}%</b>
        <span>more accurate than the seasonal-average baseline,<br />
          in held-out rolling back-testing</span>
      </div>
    </section>
  );
}

function Kpis({ meta }) {
  const items = [
    ["Portfolio occupancy · 90d", fmtPct(meta.portfolio_avg_occ_90d), "forecast mean", ""],
    ["Listings covered", meta.n_listings.toLocaleString(), `${meta.n_markets} markets`, "gold"],
    ["Forecast error (MAE)", `${(meta.headline.model_mae * 100).toFixed(1)} pts`, "market-week occupancy", ""],
    ["Lift vs seasonal-naive", `${meta.headline.improvement_pct}%`, "lower error", "terra"],
    ["Horizon", `${meta.horizon_days} days`, `from ${fmtDate(meta.origin)}`, "gold"],
  ];
  return (
    <div className="kpis rise d2">
      {items.map(([lab, val, sub, cls]) => (
        <div key={lab} className={`kpi ${cls}`}>
          <div className="lab">{lab}</div>
          <div className="val">{val}</div>
          <div className="sub">{sub}</div>
        </div>
      ))}
    </div>
  );
}

/* ===== TAB 1: market outlook ===================================== */
function MarketOutlook({ markets }) {
  const all = markets.markets;
  const def = ["Serengeti", "Zanzibar", "Nairobi", "Maasai Mara"]
    .filter((m) => all.some((x) => x.market === m));
  const [sel, setSel] = useState(def.length ? def : all.slice(0, 4).map((m) => m.market));
  const [smoothOn, setSmoothOn] = useState(true);

  const toggle = (m) =>
    setSel((s) => (s.includes(m) ? s.filter((x) => x !== m) : [...s, m]));

  const chosen = all.filter((m) => sel.includes(m.market));
  const chartData = useMemo(() => {
    return markets.dates.map((dt, i) => {
      const row = { date: dt };
      chosen.forEach((m) => {
        const arr = smoothOn ? smooth(m.occ, 7) : m.occ;
        row[m.market] = arr[i];
      });
      return row;
    });
  }, [sel, smoothOn, markets]);

  // weekly heatmap across all markets
  const weeks = useMemo(() => {
    const w = [];
    for (let i = 0; i < markets.dates.length; i += 7) w.push(markets.dates[i]);
    return w;
  }, [markets]);
  const heatRows = all.map((m) => ({
    market: m.market,
    cells: weekly(markets.dates, m.occ).map((x) => x.value),
  }));

  return (
    <>
      <section className="panel rise">
        <div className="section-head">
          <div>
            <div className="kicker">Forward view</div>
            <h3>Where demand is heading</h3>
          </div>
          <button className="mkt-pill on" style={{ cursor: "pointer" }}
            onClick={() => setSmoothOn((v) => !v)}>
            {smoothOn ? "Smoothed (7-day)" : "Daily"}
          </button>
        </div>

        <div className="mkt-pills" style={{ marginBottom: 18 }}>
          {all.map((m) => (
            <button key={m.market}
              className={`mkt-pill ${sel.includes(m.market) ? "on" : ""}`}
              onClick={() => toggle(m.market)}>
              {m.market}
            </button>
          ))}
        </div>

        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={chartData} margin={{ top: 6, right: 12, left: -8, bottom: 0 }}>
            <CartesianGrid stroke="#e8ddc9" vertical={false} />
            <XAxis dataKey="date" tickFormatter={fmtDate} minTickGap={42}
              tick={{ fill: "#8a7c66", fontSize: 12 }} stroke="#d8ccb6" />
            <YAxis domain={[0, 1]} tickFormatter={(v) => fmtPct(v)}
              tick={{ fill: "#8a7c66", fontSize: 12 }} stroke="#d8ccb6" width={46} />
            <Tooltip formatter={(v) => fmtPct(v, 1)} labelFormatter={fmtDateLong} />
            <Legend wrapperStyle={{ fontSize: 13, paddingTop: 8 }} />
            {chosen.map((m, i) => (
              <Line key={m.market} type="monotone" dataKey={m.market}
                stroke={LINE_COLORS[all.indexOf(m) % LINE_COLORS.length]}
                strokeWidth={2.6} dot={false} activeDot={{ r: 4 }}
                isAnimationActive={false} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </section>

      <section className="panel rise" style={{ marginTop: 16 }}>
        <div className="section-head">
          <div>
            <div className="kicker">Market × week</div>
            <h3>Occupancy heatmap</h3>
          </div>
        </div>
        <div className="heat">
          <table>
            <thead>
              <tr>
                <th></th>
                {weeks.map((w) => <th key={w}>{fmtDate(w)}</th>)}
              </tr>
            </thead>
            <tbody>
              {heatRows.map((r) => (
                <tr key={r.market}>
                  <td className="lbl">{r.market}</td>
                  {r.cells.map((v, i) => (
                    <td key={i} className="cell">
                      <div className="cellbox"
                        style={{ background: heatColor(v), color: heatText(v) }}>
                        {Math.round(v * 100)}
                      </div>
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="note">Darker = busier. Safari markets light up into the
          Jul–Aug Great Migration peak; coastal demand softens after the long rains.</p>
      </section>
    </>
  );
}

/* ===== TAB 2: listing planner =================================== */
function ListingPlanner({ data }) {
  const { listings, series, meta } = data;
  const marketsList = [...new Set(listings.map((l) => l.market))].sort();
  const [mkt, setMkt] = useState(marketsList[0]);
  const inMkt = listings.filter((l) => l.market === mkt)
    .sort((a, b) => b.avg_occ_90d - a.avg_occ_90d);
  const [lid, setLid] = useState(inMkt[0]?.listing_id);

  // keep listing valid when market changes
  useEffect(() => { setLid(inMkt[0]?.listing_id); }, [mkt]);
  const row = listings.find((l) => l.listing_id === lid) || inMkt[0];
  if (!row) return null;

  const fc = series.forecast[row.listing_id] || [];
  const fcSmooth = smooth(fc, 7);
  const hist = series.history[row.listing_id] || [];

  const chartData = [
    ...series.history_weeks.map((w, i) => ({ t: w, actual: hist[i], forecast: null })),
    ...series.forecast_dates.map((dt, i) => ({
      t: dt, actual: null, forecast: fcSmooth[i],
    })),
  ];

  const busiest = [...listings].sort((a, b) => b.peak_occ - a.peak_occ).slice(0, 7);
  const softest = [...listings].sort((a, b) => a.low_occ - b.low_occ).slice(0, 7);

  return (
    <>
      <div className="grid-planner rise">
        <section className="panel">
          <div className="kicker">Pick a listing</div>
          <h3 style={{ marginBottom: 14 }}>Plan ahead</h3>

          <div className="field" style={{ marginBottom: 12 }}>
            <label>Market</label>
            <select value={mkt} onChange={(e) => setMkt(e.target.value)}>
              {marketsList.map((m) => <option key={m}>{m}</option>)}
            </select>
          </div>
          <div className="field">
            <label>Listing</label>
            <select value={lid} onChange={(e) => setLid(Number(e.target.value))}>
              {inMkt.map((l) => (
                <option key={l.listing_id} value={l.listing_id}>
                  #{l.listing_id} · {l.archetype} · {fmtPct(l.avg_occ_90d)} occ
                </option>
              ))}
            </select>
          </div>

          <div className="statgrid">
            <div className="statbox">
              <div className="l">Avg occ · 90d</div>
              <div className="v">{fmtPct(row.avg_occ_90d)}</div>
            </div>
            <div className="statbox">
              <div className="l">Base rate</div>
              <div className="v">${row.base_price}<small>/night</small></div>
            </div>
            <div className="statbox">
              <div className="l">Peak week</div>
              <div className="v" style={{ color: "var(--sage)" }}>
                {fmtPct(row.peak_occ)}</div>
              <div className="meta-line" style={{ marginTop: 2 }}>
                {fmtDate(row.peak_week)}</div>
            </div>
            <div className="statbox">
              <div className="l">Soft week</div>
              <div className="v" style={{ color: "var(--terra)" }}>
                {fmtPct(row.low_occ)}</div>
              <div className="meta-line" style={{ marginTop: 2 }}>
                {fmtDate(row.low_week)}</div>
            </div>
          </div>

          <div className="reccard">💡 <b>Plan:</b> {row.recommendation}</div>
          <div className="meta-line">
            ⭐ <b>{row.review_score}</b> · {row.num_reviews} reviews ·{" "}
            {row.is_superhost ? "Superhost · " : ""}{row.archetype} · {row.market}
          </div>
        </section>

        <section className="panel">
          <div className="kicker">90-day forecast</div>
          <h3 style={{ marginBottom: 10 }}>Listing #{row.listing_id}</h3>
          <ResponsiveContainer width="100%" height={430}>
            <AreaChart data={chartData} margin={{ top: 6, right: 14, left: -8, bottom: 0 }}>
              <defs>
                <linearGradient id="fc" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#0e7c80" stopOpacity={0.35} />
                  <stop offset="100%" stopColor="#0e7c80" stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="#e8ddc9" vertical={false} />
              <XAxis dataKey="t" tickFormatter={fmtDate} minTickGap={46}
                tick={{ fill: "#8a7c66", fontSize: 12 }} stroke="#d8ccb6" />
              <YAxis domain={[0, 1]} tickFormatter={(v) => fmtPct(v)}
                tick={{ fill: "#8a7c66", fontSize: 12 }} stroke="#d8ccb6" width={46} />
              <Tooltip formatter={(v) => fmtPct(v, 1)} labelFormatter={fmtDateLong} />
              <ReferenceLine x={meta.origin} stroke="#c2603c" strokeDasharray="5 4"
                label={{ value: "today", fill: "#c2603c", fontSize: 11, position: "top" }} />
              <Area type="monotone" dataKey="actual" stroke="#a99878"
                strokeWidth={2} strokeDasharray="4 3" fill="none"
                name="Actual (trailing)" connectNulls dot={false}
                isAnimationActive={false} />
              <Area type="monotone" dataKey="forecast" stroke="#0e7c80"
                strokeWidth={3} fill="url(#fc)" name="Forecast"
                connectNulls dot={false} isAnimationActive={false} />
              <Legend wrapperStyle={{ fontSize: 13, paddingTop: 8 }} />
            </AreaChart>
          </ResponsiveContainer>
        </section>
      </div>

      <div className="grid2 rise" style={{ marginTop: 16 }}>
        <section className="panel">
          <div className="kicker" style={{ color: "var(--terra)" }}>🔥 Raise rates here</div>
          <ul className="actionlist" style={{ marginTop: 10 }}>
            {busiest.map((l) => (
              <li key={l.listing_id}>
                <span className="tag hot">{fmtPct(l.peak_occ)}</span>
                <span className="body">
                  <b>#{l.listing_id}</b> · {l.market} — peak week of {fmtDate(l.peak_week)}
                </span>
              </li>
            ))}
          </ul>
        </section>
        <section className="panel">
          <div className="kicker" style={{ color: "var(--teal)" }}>🧊 Fill these gaps</div>
          <ul className="actionlist" style={{ marginTop: 10 }}>
            {softest.map((l) => (
              <li key={l.listing_id}>
                <span className="tag cold">{fmtPct(l.low_occ)}</span>
                <span className="body">
                  <b>#{l.listing_id}</b> · {l.market} — soft week of {fmtDate(l.low_week)}
                </span>
              </li>
            ))}
          </ul>
        </section>
      </div>
    </>
  );
}

/* ===== TAB 3: demand drivers ==================================== */
function DemandDrivers({ importance, season }) {
  const max = Math.max(...importance.map((x) => x.pct));
  const SEASON_COLORS = { safari: "#c2603c", beach: "#0e7c80", city: "#d39a3c" };
  const months = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"];
  const seasonData = months.map((m, i) => {
    const row = { month: m };
    Object.keys(season).forEach((a) => { row[a] = season[a][i]; });
    return row;
  });

  return (
    <>
      <section className="panel rise">
        <div className="kicker">Model gain</div>
        <h3 style={{ marginBottom: 16 }}>What the forecaster watches</h3>
        <div className="bars">
          {importance.map((x) => (
            <div className="bar-row" key={x.name}>
              <div className="name">{x.name}</div>
              <div className="bar-track">
                <div className="bar-fill" style={{ width: `${(x.pct / max) * 100}%` }} />
              </div>
              <div className="pct">{x.pct}%</div>
            </div>
          ))}
        </div>
        <p className="note">It leans on a listing's <b>own track record</b>,
          <b> market seasonality</b>, the <b>annual season cycle</b> and
          <b> review quality</b> — the same signals an experienced host uses,
          quantified and projected forward. Price is deliberately left out, so the
          output is a clean <i>demand</i> forecast that hands off to the
          Dynamic-Pricing track.</p>
      </section>

      <section className="panel rise" style={{ marginTop: 16 }}>
        <div className="kicker">Learned from history</div>
        <h3 style={{ marginBottom: 16 }}>Seasonality by property type</h3>
        <ResponsiveContainer width="100%" height={340}>
          <LineChart data={seasonData} margin={{ top: 6, right: 14, left: -8, bottom: 0 }}>
            <CartesianGrid stroke="#e8ddc9" vertical={false} />
            <XAxis dataKey="month" tick={{ fill: "#8a7c66", fontSize: 12 }} stroke="#d8ccb6" />
            <YAxis domain={[0.3, 0.75]} tickFormatter={(v) => fmtPct(v)}
              tick={{ fill: "#8a7c66", fontSize: 12 }} stroke="#d8ccb6" width={46} />
            <Tooltip formatter={(v) => fmtPct(v, 1)} />
            <Legend wrapperStyle={{ fontSize: 13, paddingTop: 8 }} />
            {Object.keys(season).map((a) => (
              <Line key={a} type="monotone" dataKey={a}
                name={a[0].toUpperCase() + a.slice(1)}
                stroke={SEASON_COLORS[a] || "#6f8a64"} strokeWidth={3}
                dot={{ r: 3 }} activeDot={{ r: 5 }} isAnimationActive={false} />
            ))}
          </LineChart>
        </ResponsiveContainer>
        <p className="note">Safari peaks Jul–Oct (Great Migration) plus a Dec–Feb
          bump; coastal peaks Dec–Mar and dips in the Apr–May long rains; city
          demand stays flat — driven by business travel.</p>
      </section>
    </>
  );
}

/* ===== TAB 4: model & trust ===================================== */
function ModelTrust({ meta }) {
  const bestMw = Math.min(...meta.baselines.map((b) => b.mae_mw));
  const bestLm = Math.min(...meta.baselines.map((b) => b.mae_lm));
  const bestAuc = Math.max(...meta.baselines.map((b) => b.auc));
  return (
    <section className="panel rise">
      <div className="kicker">Honest, held-out</div>
      <h3 style={{ marginBottom: 8 }}>Rolling-origin back-testing</h3>
      <p className="note" style={{ marginBottom: 18 }}>
        Validated with <b>{meta.n_folds} rolling-origin folds</b>: at each origin
        the model sees only the past and forecasts the next {meta.horizon_days} days
        — a true plan-ahead test, no leakage. Lower MAE is better.
      </p>
      <table className="data">
        <thead>
          <tr>
            <th>Model</th>
            <th className="num">AUC ↑</th>
            <th className="num">MAE · market-week ↓</th>
            <th className="num">MAE · listing-month ↓</th>
          </tr>
        </thead>
        <tbody>
          {meta.baselines.map((b) => (
            <tr key={b.model} className={b.is_model ? "model" : ""}>
              <td>{b.model}{b.is_model ? "  ★" : ""}</td>
              <td className={`num ${b.auc === bestAuc ? "best" : ""}`}>{b.auc.toFixed(3)}</td>
              <td className={`num ${b.mae_mw === bestMw ? "best" : ""}`}>{b.mae_mw.toFixed(4)}</td>
              <td className={`num ${b.mae_lm === bestLm ? "best" : ""}`}>{b.mae_lm.toFixed(4)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="note">The model wins every column — it fuses each listing's
        track record <i>and</i> seasonality <i>and</i> quality, which no single
        baseline does. <i>Why isn't AUC near 1? Whether a specific night books is
        genuinely noisy; the business question is the occupancy</i> rate<i>, where
        the forecast is tight and well-calibrated.</i></p>

      <div className="grid2" style={{ marginTop: 18 }}>
        <img src="./figures/forecast_vs_actual.png" alt="Forecast vs actual"
          style={{ width: "100%", borderRadius: 12, border: "1px solid var(--line)" }} />
        <img src="./figures/calibration.png" alt="Calibration"
          style={{ width: "100%", borderRadius: 12, border: "1px solid var(--line)",
            background: "#fff" }} />
      </div>
    </section>
  );
}

function Footer({ meta }) {
  return (
    <footer className="foot">
      <p className="datanote">
        <b>Data note.</b> Pumzika's live booking history is private, so this entry
        sources its own data: a simulator grounded in documented East-African
        tourism seasonality (Great Migration, coastal long-rains, Eid/Christmas
        demand). The pipeline is dataset-agnostic — point it at real Pumzika
        exports with the same schema and it runs unchanged.
      </p>
      Pumzika Hackathon 2026 · Track 02 · Occupancy &amp; Demand Forecasting ·
      forecast origin {fmtDateLong(meta.origin)}
    </footer>
  );
}
