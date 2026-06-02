"use client";

import { useEffect, useMemo, useState, useCallback } from "react";
import {
  ResponsiveContainer, LineChart, Line, AreaChart, Area, XAxis, YAxis,
  CartesianGrid, Tooltip, ReferenceLine, Legend,
} from "recharts";
import {
  fmtPct, fmtDate, fmtDateLong, smooth, heatColor, heatText, weekly,
} from "./lib";

/* ---------- theme ------------------------------------------------------ */
const CHART = {
  light: {
    grid: "#e7e4dd", axis: "#6b6b6b",
    cats: ["#0e7c80", "#c2603c", "#b9842a", "#3f8f6b", "#7a5ea3", "#3d6b8e",
           "#a23b3b", "#557a55", "#9a6a2f", "#2f6f7a"],
    forecast: "#0e7c80", actual: "#9a948a", seasonal: "#c2603c",
    refline: "#c2603c", positive: "#3f8f6b",
    heat: ["#eef1ee", "#9fd2d1", "#2f9b9d", "#0a585b"],
  },
  dark: {
    grid: "#34343a", axis: "#a6a6ad",
    cats: ["#3fb8ba", "#e8835f", "#e0b14a", "#5fc08a", "#ad93d6", "#74a6d4",
           "#e0746f", "#7fae74", "#d6a05a", "#56b6c2"],
    forecast: "#3fb8ba", actual: "#8a8a92", seasonal: "#e8835f",
    refline: "#e8835f", positive: "#5fc08a",
    heat: ["#2b2f34", "#2f6f72", "#34a7a9", "#62d0d2"],
  },
};

function useTheme() {
  const [mode, setMode] = useState("system");
  const [resolved, setResolved] = useState("light");

  useEffect(() => {
    const p = new URLSearchParams(window.location.search).get("theme");
    const init = ["dark", "light", "system"].includes(p)
      ? p : (localStorage.getItem("pumzika-theme") || "system");
    setMode(init);
  }, []);

  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const apply = () => {
      const r = mode === "system" ? (mq.matches ? "dark" : "light") : mode;
      setResolved(r);
      document.documentElement.setAttribute("data-theme", r);
      document.documentElement.style.colorScheme = r;
    };
    apply();
    mq.addEventListener("change", apply);
    return () => mq.removeEventListener("change", apply);
  }, [mode]);

  const choose = useCallback((m) => {
    setMode(m);
    localStorage.setItem("pumzika-theme", m);
  }, []);

  return { mode, resolved, choose };
}

/* ---------- icons ------------------------------------------------------ */
const Icon = ({ d, ...p }) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
    strokeLinecap="round" strokeLinejoin="round" {...p}>{d}</svg>
);
const SunIcon = () => <Icon d={<><circle cx="12" cy="12" r="4" /><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" /></>} />;
const MoonIcon = () => <Icon d={<path d="M21 12.8A9 9 0 1 1 11.2 3 7 7 0 0 0 21 12.8z" />} />;
const SystemIcon = () => <Icon d={<><rect x="3" y="4" width="18" height="12" rx="2" /><path d="M8 20h8M12 16v4" /></>} />;

async function getJSON(name) {
  const res = await fetch(`./data/${name}`);
  return res.json();
}

/* ======================================================================= */
const TAB_SLUGS = ["outlook", "planner", "drivers", "trust"];
const TABS = ["Market outlook", "Listing planner", "Demand drivers", "Model & trust"];

export default function Page() {
  const theme = useTheme();
  const T = CHART[theme.resolved];
  const [d, setD] = useState(null);
  const [tab, setTab] = useState(0);

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
      getJSON("season.json"), getJSON("backtest.json"),
    ]).then(([meta, markets, listings, series, importance, season, backtest]) =>
      setD({ meta, markets, listings, series, importance, season, backtest })
    );
  }, []);

  if (!d) return (
    <div className="wrap">
      <div className="loading"><div className="spinner" />
        <span style={{ color: "var(--text-muted)" }}>Loading forecast…</span></div>
    </div>
  );

  return (
    <div className="wrap">
      <TopBar theme={theme} />
      <Hero meta={d.meta} />
      <StatBar meta={d.meta} />

      <nav className="tabs">
        {TABS.map((t, i) => (
          <button key={t} className={`tab ${tab === i ? "active" : ""}`}
            onClick={() => goTab(i)}>{t}</button>
        ))}
      </nav>

      <div key={tab} className="rise">
        {tab === 0 && <MarketOutlook markets={d.markets} T={T} />}
        {tab === 1 && <ListingPlanner data={d} T={T} />}
        {tab === 2 && <DemandDrivers importance={d.importance} season={d.season} T={T} />}
        {tab === 3 && <ModelTrust meta={d.meta} backtest={d.backtest} T={T} />}
      </div>

      <Footer meta={d.meta} />
    </div>
  );
}

/* ---------- chrome ----------------------------------------------------- */
function TopBar({ theme }) {
  const opts = [["light", SunIcon], ["dark", MoonIcon], ["system", SystemIcon]];
  return (
    <header className="topbar">
      <div className="brand">
        <div className="mark">P</div>
        <div>
          <b>Pumzika Demand Radar</b>
          <small>Occupancy &amp; demand forecasting</small>
        </div>
      </div>
      <div className="right">
        <div className="chips">
          <span className="chip">Tanzania</span>
          <span className="chip">Kenya</span>
          <span className="chip">Uganda</span>
        </div>
        <div className="theme-toggle" role="group" aria-label="Theme">
          {opts.map(([m, Ico]) => (
            <button key={m} aria-label={m} aria-pressed={theme.mode === m}
              onClick={() => theme.choose(m)}><Ico /></button>
          ))}
        </div>
      </div>
    </header>
  );
}

function Hero({ meta }) {
  return (
    <section className="hero">
      <div className="eyebrow"><span className="dot" />
        Pumzika Hackathon 2026 · Track 02</div>
      <h1>Know your peaks before they arrive.</h1>
      <p>
        A {meta.horizon_days}-day occupancy forecast for every host across East
        Africa — so owners can price, staff and promote ahead of demand instead
        of reacting after the fact. In held-out rolling back-testing it forecasts
        market occupancy <strong>{meta.headline.improvement_pct}% more accurately
        </strong> than the seasonal-average baseline.
      </p>
    </section>
  );
}

function StatBar({ meta }) {
  return (
    <div className="statbar">
      <div className="stat lead">
        <div className="label">Lift vs seasonal-naive</div>
        <div className="value">{meta.headline.improvement_pct}%</div>
        <div className="sub">lower forecast error, held-out</div>
      </div>
      <div className="stat">
        <div className="label">Portfolio occupancy · 90d</div>
        <div className="value">{fmtPct(meta.portfolio_avg_occ_90d)}</div>
        <div className="sub">forecast mean</div>
      </div>
      <div className="stat">
        <div className="label">Forecast error</div>
        <div className="value">{(meta.headline.model_mae * 100).toFixed(1)}<small style={{ fontSize: "1rem", color: "var(--text-muted)" }}> pts</small></div>
        <div className="sub">market-week MAE</div>
      </div>
      <div className="stat">
        <div className="label">Coverage</div>
        <div className="value">{meta.n_listings.toLocaleString()}</div>
        <div className="sub">listings · {meta.n_markets} markets</div>
      </div>
    </div>
  );
}

const axisProps = (T) => ({
  tick: { fill: T.axis, fontSize: 12 }, stroke: T.grid,
  tickLine: { stroke: T.grid },
});

/* ===== TAB 1: market outlook =========================================== */
function MarketOutlook({ markets, T }) {
  const all = markets.markets;
  const def = ["Serengeti", "Zanzibar", "Nairobi", "Maasai Mara"]
    .filter((m) => all.some((x) => x.market === m));
  const [sel, setSel] = useState(def.length ? def : all.slice(0, 4).map((m) => m.market));
  const [smoothOn, setSmoothOn] = useState(true);
  const toggle = (m) =>
    setSel((s) => (s.includes(m) ? s.filter((x) => x !== m) : [...s, m]));
  const chosen = all.filter((m) => sel.includes(m.market));

  const chartData = useMemo(() => markets.dates.map((dt, i) => {
    const row = { date: dt };
    chosen.forEach((m) => {
      const arr = smoothOn ? smooth(m.occ, 7) : m.occ;
      row[m.market] = arr[i];
    });
    return row;
  }), [sel, smoothOn, markets]);

  const weeks = useMemo(() => {
    const w = [];
    for (let i = 0; i < markets.dates.length; i += 7) w.push(markets.dates[i]);
    return w;
  }, [markets]);
  const heatRows = all.map((m) => ({
    market: m.market, cells: weekly(markets.dates, m.occ).map((x) => x.value),
  }));

  return (
    <div className="stack">
      <section className="panel">
        <div className="panel-head row-between">
          <div>
            <h2>Where demand is heading</h2>
            <p>Forecast occupancy for the next {markets.dates.length} days. Toggle
              markets to compare safari, coastal and city demand curves.</p>
          </div>
          <button className="toggle-btn" onClick={() => setSmoothOn((v) => !v)}>
            {smoothOn ? "Smoothed · 7-day" : "Daily"}
          </button>
        </div>

        <div className="pills" style={{ marginBottom: 18 }}>
          {all.map((m) => (
            <button key={m.market}
              className={`pill ${sel.includes(m.market) ? "on" : ""}`}
              onClick={() => toggle(m.market)}>{m.market}</button>
          ))}
        </div>

        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={chartData} margin={{ top: 6, right: 14, left: -6, bottom: 0 }}>
            <CartesianGrid stroke={T.grid} vertical={false} />
            <XAxis dataKey="date" tickFormatter={fmtDate} minTickGap={42} {...axisProps(T)} />
            <YAxis domain={[0, 1]} tickFormatter={(v) => fmtPct(v)} width={44} {...axisProps(T)} />
            <Tooltip formatter={(v) => fmtPct(v, 1)} labelFormatter={fmtDateLong}
              cursor={{ stroke: T.grid }} />
            <Legend wrapperStyle={{ fontSize: 13, paddingTop: 10 }} />
            {chosen.map((m) => (
              <Line key={m.market} type="monotone" dataKey={m.market}
                stroke={T.cats[all.indexOf(m) % T.cats.length]} strokeWidth={2.4}
                dot={false} activeDot={{ r: 4 }} isAnimationActive={false} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </section>

      <section className="panel">
        <div className="panel-head">
          <h2>Occupancy heatmap</h2>
          <p>Every market by week. Safari markets light up into the Jul–Aug Great
            Migration peak; coastal demand softens after the long rains.</p>
        </div>
        <div className="heat">
          <table>
            <thead>
              <tr><th></th>{weeks.map((w) => <th key={w}>{fmtDate(w)}</th>)}</tr>
            </thead>
            <tbody>
              {heatRows.map((r) => (
                <tr key={r.market}>
                  <td className="lbl">{r.market}</td>
                  {r.cells.map((v, i) => {
                    const c = heatColor(v, T.heat);
                    return (
                      <td key={i}>
                        <div className="cellbox"
                          style={{ background: c.rgb, color: heatText(c.lum) }}>
                          {Math.round(v * 100)}
                        </div>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

/* ===== TAB 2: listing planner ========================================== */
function ListingPlanner({ data, T }) {
  const { listings, series, meta } = data;
  const marketsList = [...new Set(listings.map((l) => l.market))].sort();
  const [mkt, setMkt] = useState(marketsList[0]);
  const inMkt = listings.filter((l) => l.market === mkt)
    .sort((a, b) => b.avg_occ_90d - a.avg_occ_90d);
  const [lid, setLid] = useState(inMkt[0]?.listing_id);
  useEffect(() => { setLid(inMkt[0]?.listing_id); }, [mkt]);
  const row = listings.find((l) => l.listing_id === lid) || inMkt[0];
  if (!row) return null;

  const fc = smooth(series.forecast[row.listing_id] || [], 7);
  const hist = series.history[row.listing_id] || [];
  const chartData = [
    ...series.history_weeks.map((w, i) => ({ t: w, actual: hist[i], forecast: null })),
    ...series.forecast_dates.map((dt, i) => ({ t: dt, actual: null, forecast: fc[i] })),
  ];
  const busiest = [...listings].sort((a, b) => b.peak_occ - a.peak_occ).slice(0, 7);
  const softest = [...listings].sort((a, b) => a.low_occ - b.low_occ).slice(0, 7);

  return (
    <div className="stack">
      <div className="cols planner">
        <section className="panel">
          <div className="panel-head"><h2>Plan ahead</h2></div>
          <div className="field">
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

          <div className="minigrid">
            <div className="box">
              <div className="l">Avg occ · 90d</div>
              <div className="v">{fmtPct(row.avg_occ_90d)}</div>
            </div>
            <div className="box">
              <div className="l">Base rate</div>
              <div className="v">${row.base_price}<small>/night</small></div>
            </div>
            <div className="box">
              <div className="l">Peak week</div>
              <div className="v" style={{ color: "var(--positive)" }}>{fmtPct(row.peak_occ)}</div>
              <div className="x">{fmtDate(row.peak_week)}</div>
            </div>
            <div className="box">
              <div className="l">Soft week</div>
              <div className="v" style={{ color: "var(--warm)" }}>{fmtPct(row.low_occ)}</div>
              <div className="x">{fmtDate(row.low_week)}</div>
            </div>
          </div>

          <div className="callout"><strong>Plan:</strong> {row.recommendation}</div>
          <div className="meta">
            ★ <strong>{row.review_score}</strong> · {row.num_reviews} reviews ·{" "}
            {row.is_superhost ? "Superhost · " : ""}{row.archetype}
          </div>
        </section>

        <section className="panel">
          <div className="panel-head"><h2>90-day forecast · #{row.listing_id}</h2></div>
          <ResponsiveContainer width="100%" height={420}>
            <AreaChart data={chartData} margin={{ top: 6, right: 16, left: -6, bottom: 0 }}>
              <CartesianGrid stroke={T.grid} vertical={false} />
              <XAxis dataKey="t" tickFormatter={fmtDate} minTickGap={46} {...axisProps(T)} />
              <YAxis domain={[0, 1]} tickFormatter={(v) => fmtPct(v)} width={44} {...axisProps(T)} />
              <Tooltip formatter={(v) => fmtPct(v, 1)} labelFormatter={fmtDateLong}
                cursor={{ stroke: T.grid }} />
              <ReferenceLine x={meta.origin} stroke={T.refline} strokeDasharray="5 4"
                label={{ value: "today", fill: T.refline, fontSize: 11, position: "top" }} />
              <Area type="monotone" dataKey="actual" stroke={T.actual} strokeWidth={2}
                strokeDasharray="4 3" fill="none" name="Actual (trailing)"
                connectNulls dot={false} isAnimationActive={false} />
              <Area type="monotone" dataKey="forecast" stroke={T.forecast} strokeWidth={2.6}
                fill={T.forecast} fillOpacity={0.10} name="Forecast"
                connectNulls dot={false} isAnimationActive={false} />
              <Legend wrapperStyle={{ fontSize: 13, paddingTop: 10 }} />
            </AreaChart>
          </ResponsiveContainer>
        </section>
      </div>

      <div className="cols two">
        <section className="panel">
          <span className="head-pill hot">Raise rates here</span>
          <ul className="actions">
            {busiest.map((l) => (
              <li key={l.listing_id}>
                <span className="tag hot">{fmtPct(l.peak_occ)}</span>
                <span className="body"><strong>#{l.listing_id}</strong> · {l.market}
                  {" "}— peak week of {fmtDate(l.peak_week)}</span>
              </li>
            ))}
          </ul>
        </section>
        <section className="panel">
          <span className="head-pill cold">Fill these gaps</span>
          <ul className="actions">
            {softest.map((l) => (
              <li key={l.listing_id}>
                <span className="tag cold">{fmtPct(l.low_occ)}</span>
                <span className="body"><strong>#{l.listing_id}</strong> · {l.market}
                  {" "}— soft week of {fmtDate(l.low_week)}</span>
              </li>
            ))}
          </ul>
        </section>
      </div>
    </div>
  );
}

/* ===== TAB 3: demand drivers =========================================== */
function DemandDrivers({ importance, season, T }) {
  const max = Math.max(...importance.map((x) => x.pct));
  const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const seasonData = months.map((m, i) => {
    const row = { month: m };
    Object.keys(season).forEach((a) => { row[a] = season[a][i]; });
    return row;
  });
  const sColor = { safari: T.cats[1], beach: T.cats[0], city: T.cats[2] };

  return (
    <div className="stack">
      <section className="panel">
        <div className="panel-head">
          <h2>What the forecaster watches</h2>
          <p>Relative model gain per signal. It leans on a listing's own track
            record, market seasonality, the annual season cycle and review quality
            — the same signals an experienced host uses, quantified and projected
            forward. Price is deliberately excluded, so the output is a clean
            demand forecast that hands off to the Dynamic-Pricing track.</p>
        </div>
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
      </section>

      <section className="panel">
        <div className="panel-head">
          <h2>Seasonality, learned from history</h2>
          <p>Safari peaks Jul–Oct (Great Migration) plus a Dec–Feb bump; coastal
            peaks Dec–Mar and dips in the Apr–May long rains; city demand stays
            flat — driven by business travel.</p>
        </div>
        <ResponsiveContainer width="100%" height={340}>
          <LineChart data={seasonData} margin={{ top: 6, right: 16, left: -6, bottom: 0 }}>
            <CartesianGrid stroke={T.grid} vertical={false} />
            <XAxis dataKey="month" {...axisProps(T)} />
            <YAxis domain={[0.3, 0.75]} tickFormatter={(v) => fmtPct(v)} width={44} {...axisProps(T)} />
            <Tooltip formatter={(v) => fmtPct(v, 1)} cursor={{ stroke: T.grid }} />
            <Legend wrapperStyle={{ fontSize: 13, paddingTop: 10 }} />
            {Object.keys(season).map((a) => (
              <Line key={a} type="monotone" dataKey={a}
                name={a[0].toUpperCase() + a.slice(1)} stroke={sColor[a] || T.cats[3]}
                strokeWidth={2.6} dot={{ r: 3 }} activeDot={{ r: 5 }} isAnimationActive={false} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </section>
    </div>
  );
}

/* ===== TAB 4: model & trust ============================================ */
function ModelTrust({ meta, backtest, T }) {
  const bestMw = Math.min(...meta.baselines.map((b) => b.mae_mw));
  const bestLm = Math.min(...meta.baselines.map((b) => b.mae_lm));
  const bestAuc = Math.max(...meta.baselines.map((b) => b.auc));

  const fvaMarket = "Maasai Mara";
  const fva = (backtest.fva[fvaMarket] || []).map((r) => ({
    week: r.week, Actual: r.actual, Forecast: r.pred, Seasonal: r.seasonal,
  }));
  const calib = backtest.calibration.map((r) => ({ p: r.p, o: r.o }));

  return (
    <div className="stack">
      <section className="panel">
        <div className="panel-head">
          <h2>Honest, held-out back-testing</h2>
          <p>Validated with {meta.n_folds} rolling-origin folds: at each origin the
            model sees only the past and forecasts the next {meta.horizon_days} days
            — a true plan-ahead test, no leakage. Lower MAE is better.</p>
        </div>
        <table className="data">
          <thead>
            <tr><th>Model</th><th className="num">AUC</th>
              <th className="num">MAE · market-week</th>
              <th className="num">MAE · listing-month</th></tr>
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
          track record <em>and</em> seasonality <em>and</em> quality, which no
          single baseline does. Whether a specific night books is genuinely noisy,
          so the business question is the occupancy <em>rate</em>, where the
          forecast is tight and well-calibrated.</p>
      </section>

      <div className="cols two">
        <section className="panel">
          <div className="panel-head"><h2>Forecast vs actual</h2>
            <p>{fvaMarket}, held-out — forecast tracks reality, beating seasonal-naive.</p></div>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={fva} margin={{ top: 6, right: 14, left: -6, bottom: 0 }}>
              <CartesianGrid stroke={T.grid} vertical={false} />
              <XAxis dataKey="week" tickFormatter={fmtDate} minTickGap={30} {...axisProps(T)} />
              <YAxis domain={[0, 1]} tickFormatter={(v) => fmtPct(v)} width={44} {...axisProps(T)} />
              <Tooltip formatter={(v) => fmtPct(v, 1)} labelFormatter={fmtDateLong}
                cursor={{ stroke: T.grid }} />
              <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
              <Line type="monotone" dataKey="Actual" stroke={T.actual} strokeWidth={2.4}
                dot={false} isAnimationActive={false} />
              <Line type="monotone" dataKey="Forecast" stroke={T.forecast} strokeWidth={2.6}
                dot={false} isAnimationActive={false} />
              <Line type="monotone" dataKey="Seasonal" stroke={T.seasonal} strokeWidth={1.6}
                strokeDasharray="5 4" dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </section>

        <section className="panel">
          <div className="panel-head"><h2>Calibration</h2>
            <p>Predicted probability vs observed frequency — close to the diagonal
              means the forecast can be trusted as a rate.</p></div>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={calib} margin={{ top: 6, right: 14, left: -6, bottom: 0 }}>
              <CartesianGrid stroke={T.grid} />
              <XAxis dataKey="p" type="number" domain={[0, 1]} tickFormatter={(v) => fmtPct(v)} {...axisProps(T)} />
              <YAxis domain={[0, 1]} tickFormatter={(v) => fmtPct(v)} width={44} {...axisProps(T)} />
              <Tooltip formatter={(v) => fmtPct(v, 1)} cursor={{ stroke: T.grid }} />
              <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]}
                stroke={T.axis} strokeDasharray="4 4" />
              <Line type="monotone" dataKey="o" stroke={T.forecast} strokeWidth={2.6}
                dot={{ r: 3 }} isAnimationActive={false} name="LightGBM" />
            </LineChart>
          </ResponsiveContainer>
        </section>
      </div>
    </div>
  );
}

function Footer({ meta }) {
  return (
    <footer className="foot">
      <p className="datanote"><strong>Data note.</strong> Pumzika's live booking
        history is private, so this entry sources its own data: a simulator
        grounded in documented East-African tourism seasonality (Great Migration,
        coastal long-rains, Eid/Christmas demand). The pipeline is dataset-agnostic
        — point it at real Pumzika exports with the same schema and it runs
        unchanged.</p>
      <p className="fine">Pumzika Hackathon 2026 · Track 02 · Occupancy &amp; Demand
        Forecasting · forecast origin {fmtDateLong(meta.origin)}</p>
    </footer>
  );
}
