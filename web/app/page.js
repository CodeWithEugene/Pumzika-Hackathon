"use client";

import React, { useEffect, useMemo, useState, useCallback } from "react";
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

function downloadCSV(markets) {
  if (!markets) return;
  const rows = [["date", ...markets.markets.map((m) => m.market)]];
  markets.dates.forEach((d, i) => {
    rows.push([d, ...markets.markets.map((m) => String(m.occ?.[i] ?? ""))]);
  });
  const csv = rows.map((r) => r.join(",")).join("\n");
  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = "pumzika-forecast.csv";
  a.click();
  URL.revokeObjectURL(url);
}

async function getJSON(name) {
  const res = await fetch(`./data/${name}`);
  return res.json();
}

/* ======================================================================= */
const TAB_SLUGS = ["outlook", "planner", "drivers", "trust", "calendar"];
const TABS = ["Market Outlook", "Listing Planner", "Demand Drivers", "Model & Trust", "Calendar"];

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
      getJSON("season.json"), getJSON("backtest.json"), getJSON("real.json"),
      getJSON("kaggle.json"), getJSON("east_africa.json"),
    ]).then(([meta, markets, listings, series, importance, season, backtest, real, kaggle, eastAfrica]) =>
      setD({ meta, markets, listings, series, importance, season, backtest, real, kaggle, eastAfrica })
    ).catch(() => setD({ error: true }));
  }, []);

  if (!d) return (
    <div className="wrap">
      <div className="loading"><div className="spinner" />
        <span style={{ color: "var(--text-muted)" }}>Loading forecast…</span></div>
    </div>
  );

  return (
    <div className="wrap">
      <TopBar theme={theme} meta={d.meta} onExport={() => downloadCSV(d.markets)} />
      <Hero meta={d.meta} kaggle={d.kaggle} />

      <StatBar meta={d.meta} kaggle={d.kaggle} />
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
        {tab === 3 && <ModelTrust meta={d.meta} backtest={d.backtest} real={d.real} kaggle={d.kaggle} eastAfrica={d.eastAfrica} T={T} />}
        {tab === 4 && <CalendarView markets={d.markets} T={T} />}
      </div>

      <Footer />
    </div>
  );
}

/* ---------- chrome ----------------------------------------------------- */
const ExportIcon = () => <Icon d={<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M7 10l5 5 5-5M12 15V3" />} width="20" height="20" />;

function TopBar({ theme, meta, onExport }) {
  const opts = [["light", SunIcon], ["dark", MoonIcon], ["system", SystemIcon]];
  const isKaggle = meta?.source === "kaggle";
  return (
    <header className="topbar">
      <div className="brand">
        <img className="mark" src="/pumzika-mark.png" alt="Pumzika" width="38" height="38" />
        <div>
          <b>Pumzika Demand Radar</b>
          <small>Occupancy &amp; Demand Forecasting</small>
        </div>
      </div>
      <div className="right">
        <div className="chips">
          {isKaggle ? (
            <span className="chip">Portugal</span>
          ) : (
            <><span className="chip">Tanzania</span><span className="chip">Kenya</span><span className="chip">Uganda</span></>
          )}
        </div>
        {onExport && <button className="export-btn" onClick={onExport}
          aria-label="Download CSV"><ExportIcon /></button>}
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

function Hero({ meta, kaggle }) {
  return (
    <section className="hero">
      <div className="eyebrow"><span className="dot" />
        Pumzika Hackathon 2026 · Track 02</div>
      <h1>Know Your Peaks Before They Arrive.</h1>
      <p>
      A {meta.horizon_days}-day hotel occupancy forecast for every property
      — so managers can price, staff and promote ahead of demand instead of
      reacting after the fact. Validated on the official Hotel Booking Demand
      dataset (Antonio, Almeida, Nunes 2019), the model beats the seasonal
      baseline by <strong>{kaggle ? kaggle.headline.improvement_pct : meta.headline.improvement_pct}%</strong>
      {" "}      in held-out rolling back-testing.{kaggle
        ? " The same pipeline runs on synthetic East African STR data and real Inside Airbnb Cape Town data — because the model is dataset-agnostic."
        : " Demonstrated on full East African STR data and real Airbnb Cape Town data."}
      </p>
    </section>
  );
}

function StatBar({ meta, kaggle }) {
  const k = kaggle;
  return (
    <div className="statbar">
      <div className="stat lead">
        <div className="label">Lift Vs Seasonal-Naive</div>
        <div className="value">{k ? k.headline.improvement_pct : meta.headline.improvement_pct}%</div>
        <div className="sub">Lower Forecast Error, Held-Out</div>
      </div>
      <div className="stat">
        <div className="label">Occupancy Rate</div>
        <div className="value">{k ? fmtPct(k.mean_occupancy) : fmtPct(meta.portfolio_avg_occ_90d)}</div>
        <div className="sub">Forecast Mean{k ? " · 2 Hotels" : ""}</div>
      </div>
      <div className="stat">
        <div className="label">Forecast Error</div>
        <div className="value">{(k ? (k.headline.model_hotel_week_MAE * 100).toFixed(1) : (meta.headline.model_mae * 100).toFixed(1))}<small style={{ fontSize: "1rem", color: "var(--text-muted)" }}> pts</small></div>
        <div className="sub">Hotel-Week MAE</div>
      </div>
      <div className="stat">
        <div className="label">Data</div>
        <div className="value">{k ? k.n_bookings.toLocaleString() : meta.n_listings.toLocaleString()}</div>
        <div className="sub">{k ? "Bookings" : "Listings"} · {k ? "2 Hotels (Kaggle)" : meta.n_markets + " Markets"}</div>
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
      if (m.lower && m.upper) {
        const lo = smoothOn ? smooth(m.lower, 7) : m.lower;
        const hi = smoothOn ? smooth(m.upper, 7) : m.upper;
        row[m.market + "_lower"] = lo[i];
        row[m.market + "_upper"] = hi[i];
      }
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
            <h2>Where Demand Is Heading</h2>
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
            {chosen.map((m) => {
              const col = T.cats[all.indexOf(m) % T.cats.length];
              return (
              <React.Fragment key={m.market}>
                {m.lower && m.upper && <>
                  <Area type="monotone" dataKey={m.market + "_upper"} fill={col}
                    fillOpacity={0.12} stroke="none" dot={false} isAnimationActive={false} />
                  <Area type="monotone" dataKey={m.market + "_lower"} fill={col}
                    fillOpacity={0.12} stroke="none" dot={false} isAnimationActive={false} />
                </>}
                <Line type="monotone" dataKey={m.market} stroke={col} strokeWidth={2.4}
                  dot={false} activeDot={{ r: 4 }} isAnimationActive={false} />
              </React.Fragment>
            );})}
          </LineChart>
        </ResponsiveContainer>
      </section>

      <section className="panel">
        <div className="panel-head">
          <h2>Occupancy Heatmap</h2>
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

      <PricingCrossover markets={markets} T={T} />
      <WhatIfScenario markets={markets} T={T} />
    </div>
  );
}

/* ---------- pricing crossover ----------------------------------------- */
function PricingCrossover({ markets, T }) {
  const all = markets.markets.filter((m) => m.avg_adr);
  const ELASTICITY = 0.3;
  const SCENARIOS = [0, 5, 10, 15, 20];
  if (all.length === 0) return null;
  return (
    <section className="panel">
      <div className="panel-head">
        <h2>From Forecast → Revenue</h2>
        <p>Expected revenue per room at different rate uplifts,
          assuming {ELASTICITY * 100}% demand elasticity.
          <span className="data-note"> Starting point for Track 01: Dynamic Pricing.</span>
        </p>
      </div>
      <div className="rev-grid">
        {all.map((m) => {
          const adr = m.avg_adr;
          const occ = m.avg;
          const curRev = adr * occ;
          return (
            <div key={m.market} className="rev-card">
              <h4>{m.market}</h4>
              <div className="sub">${adr.toFixed(0)} avg rate · {fmtPct(occ)} forecast occ</div>
              <table>
                <thead>
                  <tr>
                    <th>Scenario</th><th>Rate</th><th>Occupancy</th>
                    <th>Rev/night</th><th>Change</th>
                  </tr>
                </thead>
                <tbody>
                  {SCENARIOS.map((u) => {
                    const mult = 1 + u / 100;
                    const rate = adr * mult;
                    const adjOcc = Math.min(1, Math.max(0, occ * (1 - ELASTICITY * u / 100)));
                    const rev = rate * adjOcc;
                    const chg = (rev - curRev) / curRev;
                    return (
                      <tr key={u}>
                        <td>{u === 0 ? "Current" : `+${u}% rate`}</td>
                        <td>${rate.toFixed(0)}</td>
                        <td>{fmtPct(adjOcc)}</td>
                        <td>${rev.toFixed(1)}</td>
                        <td className={chg >= 0 ? "pos" : "neg"}>
                          {chg >= 0 ? "+" : ""}{fmtPct(chg)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          );
        })}
      </div>
    </section>
  );
}

/* ---------- what-if scenario ------------------------------------------- */
function WhatIfScenario({ markets, T }) {
  const all = markets.markets.filter((m) => m.avg_adr);
  const [demandOff, setDemandOff] = useState(0);
  const [priceOff, setPriceOff] = useState(0);
  if (all.length === 0) return null;
  return (
    <section className="panel">
      <div className="panel-head">
        <h2>What-If Scenario</h2>
        <p>Shift demand or adjust pricing to see the revenue impact.
          <span className="data-note"> All values per room·night.</span>
        </p>
      </div>
      <div className="scenario-sliders">
        <div className="slider-grp">
          <label>Demand <b>{demandOff >= 0 ? "+" : ""}{demandOff}%</b></label>
          <input type="range" min="-20" max="20" value={demandOff}
            onChange={(e) => setDemandOff(+e.target.value)} />
          <div className="slider-labels"><span>-20%</span><span>+20%</span></div>
        </div>
        <div className="slider-grp">
          <label>Price <b>{priceOff >= 0 ? "+" : ""}{priceOff}%</b></label>
          <input type="range" min="-20" max="20" value={priceOff}
            onChange={(e) => setPriceOff(+e.target.value)} />
          <div className="slider-labels"><span>-20%</span><span>+20%</span></div>
        </div>
      </div>
      <div className="scenario-table-wrap">
        <table className="data">
          <thead>
            <tr>
              <th>Market</th>
              <th className="num">Occ</th>
              <th className="num">Adj. Occ</th>
              <th className="num">Rev</th>
              <th className="num">Adj. Rev</th>
              <th className="num">Change</th>
            </tr>
          </thead>
          <tbody>
            {all.map((m) => {
              const adjOcc = Math.min(1, Math.max(0, m.avg * (1 + demandOff / 100)));
              const baseRev = m.avg * m.avg_adr;
              const adjRev = adjOcc * m.avg_adr * (1 + priceOff / 100);
              const chg = (adjRev - baseRev) / baseRev;
              return (
                <tr key={m.market}>
                  <td>{m.market}</td>
                  <td className="num">{fmtPct(m.avg)}</td>
                  <td className="num">{fmtPct(adjOcc)}</td>
                  <td className="num">${baseRev.toFixed(1)}</td>
                  <td className="num">${adjRev.toFixed(1)}</td>
                  <td className={`num ${chg >= 0 ? "pos" : "neg"}`}>
                    {chg >= 0 ? "+" : ""}{fmtPct(chg)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

/* ===== TAB 5: calendar view =========================================== */
function CalendarView({ markets, T }) {
  const all = markets.markets;
  const [mkt, setMkt] = useState(all[0].market);
  const market = all.find((m) => m.market === mkt);
  const DOWS = ["S", "M", "T", "W", "T", "F", "S"];

  const months = useMemo(() => {
    if (!market) return [];
    const out = [];
    let cur = null;
    markets.dates.forEach((ds, i) => {
      const d = new Date(ds + "T00:00:00");
      const key = d.getFullYear() + "-" + String(d.getMonth() + 1).padStart(2, "0");
      if (key !== cur) {
        cur = key;
        const first = new Date(d.getFullYear(), d.getMonth(), 1);
        out.push({
          label: d.toLocaleString("en-US", { month: "long", year: "numeric" }),
          key, pad: first.getDay(), days: [],
        });
      }
      out[out.length - 1].days.push({
        date: ds, day: d.getDate(), dow: d.getDay(), occ: market.occ?.[i],
      });
    });
    return out;
  }, [market, markets]);

  if (!market) return null;

  return (
    <div className="stack">
      <section className="panel">
        <div className="panel-head">
          <h2>Forecast Calendar</h2>
          <p>Daily occupancy by month — colour intensity shows demand level.
            <span className="data-note"> Use for housekeeping &amp; ops planning.</span>
          </p>
        </div>
        <div className="pills" style={{ marginBottom: 18 }}>
          {all.map((m) => (
            <button key={m.market} className={`pill ${mkt === m.market ? "on" : ""}`}
              onClick={() => setMkt(m.market)}>{m.market}</button>
          ))}
        </div>
        <div className="cal-months">
          {months.map((mo) => (
            <div key={mo.key} className="cal-month">
              <div className="cal-title">{mo.label}</div>
              <div className="cal-dows">
                {DOWS.map((d) => <div key={d} className="cal-dow">{d}</div>)}
              </div>
              <div className="cal-days">
                {Array.from({ length: mo.pad }).map((_, i) => (
                  <div key={`p-${i}`} className="cal-cell cal-empty" />
                ))}
                {mo.days.map((day) => {
                  const c = heatColor(day.occ, T.heat);
                  return (
                    <div key={day.date} className="cal-cell"
                      style={{ background: c.rgb, color: heatText(c.lum) }}
                      title={`${day.date}: ${Math.round(day.occ * 100)}%`}>
                      <span className="cal-num">{day.day}</span>
                      <span className="cal-val">{Math.round(day.occ * 100)}%</span>
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

/* ===== TAB 2: listing planner ========================================== */
function ListingPlanner({ data, T }) {
  const { listings, series, meta } = data;
  const isHotel = listings.length > 0 && !("archetype" in listings[0]);
  const marketsList = [...new Set(listings.map((l) => l.market))].sort();
  const [mkt, setMkt] = useState(marketsList[0]);
  const inMkt = listings.filter((l) => l.market === mkt)
    .sort((a, b) => b.avg_occ_90d - a.avg_occ_90d);
  const [lid, setLid] = useState(inMkt[0]?.listing_id);
  useEffect(() => { setLid(inMkt[0]?.listing_id); }, [mkt]);
  const row = listings.find((l) => l.listing_id === lid) || inMkt[0];
  if (!row) return null;

  const fc = smooth(series.forecast[row.listing_id] || [], 7);
  const fcRawLower = series.forecast_lower?.[row.listing_id] || [];
  const fcRawUpper = series.forecast_upper?.[row.listing_id] || [];
  const fcLower = fcRawLower.length ? smooth(fcRawLower, 7) : [];
  const fcUpper = fcRawUpper.length ? smooth(fcRawUpper, 7) : [];
  const hasInterval = fcLower.length > 0 && fcUpper.length > 0;
  const hist = series.history[row.listing_id] || [];
  const chartData = [
    ...series.history_weeks.map((w, i) => ({ t: w, actual: hist[i], forecast: null, lower: null, upper: null })),
    ...series.forecast_dates.map((dt, i) => ({
      t: dt, actual: null, forecast: fc[i],
      lower: hasInterval ? fcLower[i] : null,
      upper: hasInterval ? fcUpper[i] : null,
    })),
  ];
  const busiest = [...listings].sort((a, b) => b.peak_occ - a.peak_occ).slice(0, 7);
  const softest = [...listings].sort((a, b) => a.low_occ - b.low_occ).slice(0, 7);

  return (
    <div className="stack">
      <div className="cols planner">
        <section className="panel">
          <div className="panel-head"><h2>Plan Ahead</h2></div>
          <div className="field">
            <label>Market</label>
            <select value={mkt} onChange={(e) => setMkt(e.target.value)}>
              {marketsList.map((m) => <option key={m}>{m}</option>)}
            </select>
          </div>
          <div className="field">
            <label>Listing</label>
            <select value={lid} onChange={(e) => setLid(typeof inMkt[0]?.listing_id === "number" ? Number(e.target.value) : e.target.value)}>
              {inMkt.map((l) => (
                <option key={l.listing_id} value={l.listing_id}>
                  {isHotel ? l.listing_id + " · " + fmtPct(l.avg_occ_90d) + " occ"
                    : "#" + l.listing_id + " · " + l.archetype + " · " + fmtPct(l.avg_occ_90d) + " occ"}
                </option>
              ))}
            </select>
          </div>

          <div className="minigrid">
            <div className="box">
              <div className="l">Avg occ · 90d</div>
              <div className="v">{fmtPct(row.avg_occ_90d)}</div>
            </div>
            {!isHotel && <div className="box">
              <div className="l">Base rate</div>
              <div className="v">${row.base_price}<small>/night</small></div>
            </div>}
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
          {!isHotel && <div className="meta">
            ★ <strong>{row.review_score}</strong> · {row.num_reviews} reviews ·{" "}
            {row.is_superhost ? "Superhost · " : ""}{row.archetype}
          </div>}
        </section>

        <section className="panel">
          <div className="panel-head"><h2>90-Day Forecast · {row.listing_id}</h2></div>
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
              {hasInterval && <>
                <Area type="monotone" dataKey="upper" fill={T.forecast} fillOpacity={0.12}
                  stroke="none" dot={false} isAnimationActive={false} />
                <Area type="monotone" dataKey="lower" fill={T.forecast} fillOpacity={0.12}
                  stroke="none" dot={false} isAnimationActive={false} />
              </>}
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
  const seasonKeys = Object.keys(season);
  const isHotelSeason = seasonKeys.length > 0 && !["beach", "safari", "city"].includes(seasonKeys[0]);
  const sColor = isHotelSeason
    ? { "City Hotel": T.cats[0], "Resort Hotel": T.cats[1] }
    : { safari: T.cats[1], beach: T.cats[0], city: T.cats[2] };

  return (
    <div className="stack">
      <section className="panel">
        <div className="panel-head">
          <h2>What The Forecaster Watches</h2>
          <p>{isHotelSeason
            ? "Relative model gain per signal. The model leans on average daily rate, lead time, market segment mix and calendar seasonality — the same signals revenue managers use to set rates and allocate inventory."
            : "Relative model gain per signal. It leans on a listing's own track record, market seasonality, the annual season cycle and review quality — the same signals an experienced host uses, quantified and projected forward. Price is deliberately excluded, so the output is a clean demand forecast that hands off to the Dynamic-Pricing track."}</p>
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
          <h2>Seasonality, Learned From History</h2>
          <p>{isHotelSeason
            ? "Both hotels show clear seasonal patterns. Resort Hotel peaks in summer holiday months and troughs in winter; City Hotel maintains steadier occupancy year-round driven by business and conference travel."
            : "Safari peaks Jul–Oct (Great Migration) plus a Dec–Feb bump; coastal peaks Dec–Mar and dips in the Apr–May long rains; city demand stays flat — driven by business travel."}</p>
        </div>
        <ResponsiveContainer width="100%" height={340}>
          <LineChart data={seasonData} margin={{ top: 6, right: 16, left: -6, bottom: 0 }}>
            <CartesianGrid stroke={T.grid} vertical={false} />
            <XAxis dataKey="month" {...axisProps(T)} />
            <YAxis domain={[0.3, 0.75]} tickFormatter={(v) => fmtPct(v)} width={44} {...axisProps(T)} />
            <Tooltip formatter={(v) => fmtPct(v, 1)} cursor={{ stroke: T.grid }} />
            <Legend wrapperStyle={{ fontSize: 13, paddingTop: 10 }} />
            {seasonKeys.map((a) => (
              <Line key={a} type="monotone" dataKey={a}
                name={a} stroke={sColor[a] || T.cats[3]}
                strokeWidth={2.6} dot={{ r: 3 }} activeDot={{ r: 5 }} isAnimationActive={false} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </section>
    </div>
  );
}

/* ===== TAB 4: model & trust ============================================ */
function KaggleValidation({ kaggle, T }) {
  if (!kaggle || kaggle.error) return null;
  const bestMae = Math.min(...kaggle.baselines.map((b) => b.mae_wk));
  const calib = (kaggle.detail?.calibration || []).map((r) => ({ p: r.p, o: r.o }));
  const hotelFva = kaggle.detail?.fva || {};
  const imps = Object.entries(kaggle.importance || {}).slice(0, 8);
  const facts = [
    [kaggle.n_bookings.toLocaleString(), "bookings"],
    [kaggle.n_hotels, "hotels (City · Resort)"],
    [fmtPct(kaggle.mean_occupancy), "mean occupancy"],
    ["2,009", "daily records"],
  ];
  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Validated On Official Kaggle Dataset</h2>
        <p>The model was re-run on the official challenge dataset —
          <strong> {kaggle.source}</strong> — beating the seasonal-naive baseline
          by <strong>{kaggle.headline.improvement_pct}%</strong> in held-out
          rolling back-testing (hotel-week MAE). This is a real hotel booking
          dataset from 2 Portuguese hotels (2015–2017), the exact data provided
          for Track 02.</p>
      </div>
      <div className="facts">
        {facts.map(([v, l]) => (
          <div className="fact" key={l}><b>{v}</b><span>{l}</span></div>
        ))}
      </div>

      <div className="cols two" style={{ marginTop: 16 }}>
        <table className="data" style={{ alignSelf: "start" }}>
          <thead>
            <tr><th>Model</th>
              <th className="num">MAE · hotel-week</th>
              <th className="num">MAE · daily</th></tr>
          </thead>
          <tbody>
            {kaggle.baselines.map((b) => (
              <tr key={b.model} className={b.is_model ? "model" : ""}>
                <td>{b.model}{b.is_model ? "  ★" : ""}</td>
                <td className={`num ${b.mae_wk === bestMae ? "best" : ""}`}>{b.mae_wk.toFixed(4)}</td>
                <td className="num">{b.mae_daily.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>

        {imps.length > 0 && (
          <div>
            <p className="note" style={{ marginTop: 0, marginBottom: 10 }}>
              Top demand drivers — ADR, lead time and segment mix join
              seasonality as key signals in the hotel context.</p>
            <div className="bars" style={{ gap: 0 }}>
              {imps.map(([name, pct]) => (
                <div className="bar-row" key={name} style={{ fontSize: "0.82rem" }}>
                  <div className="name">{name}</div>
                  <div className="bar-track">
                    <div className="bar-fill" style={{ width: `${Math.min(pct, 100)}%` }} />
                  </div>
                  <div className="pct" style={{ fontSize: "0.78rem" }}>{pct.toFixed(0)}%</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="cols two" style={{ marginTop: 20 }}>
        {Object.entries(hotelFva).map(([hotel, data]) => (
          <section className="panel" key={hotel}>
            <div className="panel-head"><h2>{hotel}</h2>
              <p>Held-out forecast vs actual (last fold).</p></div>
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={data} margin={{ top: 6, right: 14, left: -6, bottom: 0 }}>
                <CartesianGrid stroke={T.grid} vertical={false} />
                <XAxis dataKey="week" tickFormatter={fmtDate} minTickGap={20} {...axisProps(T)} />
                <YAxis domain={[0, 1]} tickFormatter={(v) => fmtPct(v)} width={44} {...axisProps(T)} />
                <Tooltip formatter={(v) => fmtPct(v, 1)} labelFormatter={fmtDateLong}
                  cursor={{ stroke: T.grid }} />
                <Legend wrapperStyle={{ fontSize: 11, paddingTop: 6 }} />
                <Line type="monotone" dataKey="actual" stroke={T.actual} strokeWidth={2.4}
                  dot={false} isAnimationActive={false} name="Actual" />
                <Line type="monotone" dataKey="pred" stroke={T.forecast} strokeWidth={2.6}
                  dot={false} isAnimationActive={false} name="Forecast" />
                <Line type="monotone" dataKey="seasonal" stroke={T.seasonal} strokeWidth={1.6}
                  strokeDasharray="5 4" dot={false} isAnimationActive={false} name="Seasonal" />
              </LineChart>
            </ResponsiveContainer>
          </section>
        ))}
        {calib.length > 0 && (
          <section className="panel">
            <div className="panel-head"><h2>Calibration</h2>
              <p>Predicted vs observed occupancy rate — close to diagonal means
                trustworthy probability estimates.</p></div>
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={calib} margin={{ top: 6, right: 14, left: -6, bottom: 0 }}>
                <CartesianGrid stroke={T.grid} />
                <XAxis dataKey="p" type="number" domain={[0, 1]} tickFormatter={(v) => fmtPct(v)} {...axisProps(T)} />
                <YAxis domain={[0, 1]} tickFormatter={(v) => fmtPct(v)} width={44} {...axisProps(T)} />
                <Tooltip formatter={(v) => fmtPct(v, 1)} cursor={{ stroke: T.grid }} />
                <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke={T.axis} strokeDasharray="4 4" />
                <Line type="monotone" dataKey="o" stroke={T.forecast} strokeWidth={2.6}
                  dot={{ r: 3 }} isAnimationActive={false} name="Model" />
              </LineChart>
            </ResponsiveContainer>
          </section>
        )}
      </div>

      <p className="note">Data: <a href={kaggle.source_url} target="_blank"
        rel="noopener noreferrer" style={{ color: "var(--accent)", fontWeight: 600 }}>
        Hotel Booking Demand Dataset</a> (Antonio, Almeida, Nunes 2019),
        CC0 license. The pipeline is dataset-agnostic — the same code runs the
        East African demo, this hotel data, and the Inside Airbnb real-estate
        data without changes.</p>
    </section>
  );
}

function RealValidation({ real, T }) {
  if (!real || real.error) return null;
  const bestAuc = Math.max(...real.baselines.map((b) => b.auc));
  const bestMw = Math.min(...real.baselines.map((b) => b.mae_mw));
  const calib = (real.detail?.calibration || []).map((r) => ({ p: r.p, o: r.o }));
  const facts = [
    [real.n_rows.toLocaleString(), "real listing-nights"],
    [real.n_listings.toLocaleString(), "listings"],
    [real.n_markets, "neighbourhoods"],
    [fmtPct(real.occupancy_rate), "occupancy"],
    [real.headline.model_AUC.toFixed(3), "model AUC"],
  ];
  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Validated On Real STR Data (Cape Town)</h2>
        <p>The exact same model and leakage-safe pipeline, re-run on real
          short-term-rental data from <strong>{real.source}</strong>. It reaches
          <strong> AUC {real.headline.model_AUC.toFixed(2)}</strong> on genuine
          market data (strong night-level discrimination) and edges the seasonal
          baseline on occupancy rate.</p>
      </div>
      <div className="facts">
        {facts.map(([v, l]) => (
          <div className="fact" key={l}><b>{v}</b><span>{l}</span></div>
        ))}
      </div>
      <div className="cols two" style={{ marginTop: 16 }}>
        <table className="data" style={{ alignSelf: "start" }}>
          <thead>
            <tr><th>Model</th><th className="num">AUC</th>
              <th className="num">MAE · market-week</th></tr>
          </thead>
          <tbody>
            {real.baselines.map((b) => (
              <tr key={b.model} className={b.is_model ? "model" : ""}>
                <td>{b.model}{b.is_model ? "  ★" : ""}</td>
                <td className={`num ${b.auc === bestAuc ? "best" : ""}`}>{b.auc.toFixed(3)}</td>
                <td className={`num ${b.mae_mw === bestMw ? "best" : ""}`}>{b.mae_mw.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div>
          <p className="note" style={{ marginTop: 0, marginBottom: 10 }}>
            Calibration on real data — predicted vs observed frequency.</p>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={calib} margin={{ top: 6, right: 14, left: -6, bottom: 0 }}>
              <CartesianGrid stroke={T.grid} />
              <XAxis dataKey="p" type="number" domain={[0, 1]} tickFormatter={(v) => fmtPct(v)} {...axisProps(T)} />
              <YAxis domain={[0, 1]} tickFormatter={(v) => fmtPct(v)} width={44} {...axisProps(T)} />
              <Tooltip formatter={(v) => fmtPct(v, 1)} cursor={{ stroke: T.grid }} />
              <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke={T.axis} strokeDasharray="4 4" />
              <Line type="monotone" dataKey="o" stroke={T.forecast} strokeWidth={2.6} dot={{ r: 3 }} isAnimationActive={false} name="Model" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
      <p className="note">Data: <a href="https://insideairbnb.com" target="_blank"
        rel="noopener noreferrer" style={{ color: "var(--accent)", fontWeight: 600 }}>Inside
        Airbnb</a>, Cape Town, CC BY 4.0.</p>
    </section>
  );
}

function EastAfricaValidation({ eastAfrica, T }) {
  if (!eastAfrica || eastAfrica.error) return null;
  const bestMw = Math.min(...eastAfrica.baselines.map((b) => b.mae_mw));
  const bestLm = Math.min(...eastAfrica.baselines.map((b) => b.mae_lm));
  const bestAuc = Math.max(...eastAfrica.baselines.map((b) => b.auc));
  const calib = (eastAfrica.detail?.calibration || []).map((r) => ({ p: r.p, o: r.o }));
  const facts = [
    [eastAfrica.n_listings.toLocaleString(), "listings"],
    [eastAfrica.n_markets, "markets (TZ · KE · UG)"],
    [eastAfrica.n_rows.toLocaleString(), "listing-nights generated"],
    [eastAfrica.n_folds + " folds", "rolling-origin"],
  ];
  return (
    <section className="panel">
      <div className="panel-head">
        <h2>Same Pipeline · East African STR Demo</h2>
        <p>The identical model architecture, re-run on synthetic East African
          short-term-rental data — <strong>{eastAfrica.n_listings} listings</strong>
          across <strong>{eastAfrica.n_markets} markets</strong> in Tanzania, Kenya
          and Uganda. The model beats the seasonal baseline by
          <strong> {eastAfrica.headline.improvement_pct}%</strong> (market-week MAE).</p>
      </div>
      <div className="facts">
        {facts.map(([v, l]) => (
          <div className="fact" key={l}><b>{v}</b><span>{l}</span></div>
        ))}
      </div>

      <div className="cols two" style={{ marginTop: 16 }}>
        <table className="data" style={{ alignSelf: "start" }}>
          <thead>
            <tr><th>Model</th><th className="num">AUC</th>
              <th className="num">MAE · market-week</th>
              <th className="num">MAE · listing-month</th></tr>
          </thead>
          <tbody>
            {eastAfrica.baselines.map((b) => (
              <tr key={b.model} className={b.is_model ? "model" : ""}>
                <td>{b.model}{b.is_model ? "  ★" : ""}</td>
                <td className={`num ${b.auc === bestAuc ? "best" : ""}`}>{b.auc.toFixed(3)}</td>
                <td className={`num ${b.mae_mw === bestMw ? "best" : ""}`}>{b.mae_mw.toFixed(4)}</td>
                <td className={`num ${b.mae_lm === bestLm ? "best" : ""}`}>{b.mae_lm.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>

        {eastAfrica.importance && eastAfrica.importance.length > 0 && (
          <div>
            <p className="note" style={{ marginTop: 0, marginBottom: 10 }}>
              Top demand drivers — track record, seasonality and review quality
              lead, as any experienced host would expect.</p>
            <div className="bars" style={{ gap: 0 }}>
              {eastAfrica.importance.map((x) => (
                <div className="bar-row" key={x.name} style={{ fontSize: "0.82rem" }}>
                  <div className="name">{x.name}</div>
                  <div className="bar-track">
                    <div className="bar-fill" style={{ width: `${x.pct}%` }} />
                  </div>
                  <div className="pct" style={{ fontSize: "0.78rem" }}>{x.pct.toFixed(0)}%</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {calib.length > 0 && (
        <div className="cols two" style={{ marginTop: 20 }}>
          <section className="panel">
            <div className="panel-head"><h2>Calibration</h2>
              <p>Predicted vs observed occupancy rate on held-out East African data.</p></div>
            <ResponsiveContainer width="100%" height={250}>
              <LineChart data={calib} margin={{ top: 6, right: 14, left: -6, bottom: 0 }}>
                <CartesianGrid stroke={T.grid} />
                <XAxis dataKey="p" type="number" domain={[0, 1]} tickFormatter={(v) => fmtPct(v)} {...axisProps(T)} />
                <YAxis domain={[0, 1]} tickFormatter={(v) => fmtPct(v)} width={44} {...axisProps(T)} />
                <Tooltip formatter={(v) => fmtPct(v, 1)} cursor={{ stroke: T.grid }} />
                <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke={T.axis} strokeDasharray="4 4" />
                <Line type="monotone" dataKey="o" stroke={T.forecast} strokeWidth={2.6} dot={{ r: 3 }} isAnimationActive={false} name="Model" />
              </LineChart>
            </ResponsiveContainer>
          </section>
        </div>
      )}

      <p className="note">Data: Synthetic STR marketplace grounded in real East
        African tourism dynamics (safari, coastal, city archetypes). The pipeline
        is dataset-agnostic — the same code runs the Kaggle hotel data, this East
        African demo, and the Inside Airbnb Cape Town data without changes.</p>
    </section>
  );
}

function ModelTrust({ meta, backtest, real, T, kaggle, eastAfrica }) {
  const isKaggle = meta?.source === "kaggle";

  return (
    <div className="stack">
      <KaggleValidation kaggle={kaggle} T={T} />

      {isKaggle && <EastAfricaValidation eastAfrica={eastAfrica} T={T} />}

      {!isKaggle && <section className="panel">
        <div className="panel-head">
          <h2>Honest, Held-Out Back-Testing</h2>
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
            {meta.baselines.map((b) => {
              const bestMw = Math.min(...meta.baselines.map((x) => x.mae_mw));
              const bestLm = Math.min(...meta.baselines.map((x) => x.mae_lm));
              const bestAuc = Math.max(...meta.baselines.map((x) => x.auc));
              return (
              <tr key={b.model} className={b.is_model ? "model" : ""}>
                <td>{b.model}{b.is_model ? "  ★" : ""}</td>
                <td className={`num ${b.auc === bestAuc ? "best" : ""}`}>{b.auc.toFixed(3)}</td>
                <td className={`num ${b.mae_mw === bestMw ? "best" : ""}`}>{b.mae_mw.toFixed(4)}</td>
                <td className={`num ${b.mae_lm === bestLm ? "best" : ""}`}>{b.mae_lm.toFixed(4)}</td>
              </tr>
            );})}
          </tbody>
        </table>
        <p className="note" style={{ textAlign: "center", maxWidth: "76ch", margin: "16px auto 0" }}>The model wins every column — it fuses each
          listing's track record <em>and</em> seasonality <em>and</em> quality,
          which no single baseline does. Whether a specific night books is
          genuinely noisy, so the business question is the occupancy
          <em>rate</em>, where the forecast is tight and well-calibrated.</p>
      </section>}

      {!isKaggle && <div className="cols two">
        <section className="panel">
          <div className="panel-head"><h2>Forecast Vs Actual</h2>
            <p>Maasai Mara, held-out — forecast tracks reality, beating seasonal-naive.</p></div>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={(backtest.fva["Maasai Mara"] || []).map((r) => ({
              week: r.week, Actual: r.actual, Forecast: r.pred, Seasonal: r.seasonal,
            }))} margin={{ top: 6, right: 14, left: -6, bottom: 0 }}>
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
            <LineChart data={backtest.calibration.map((r) => ({ p: r.p, o: r.o }))} margin={{ top: 6, right: 14, left: -6, bottom: 0 }}>
              <CartesianGrid stroke={T.grid} />
              <XAxis dataKey="p" type="number" domain={[0, 1]} tickFormatter={(v) => fmtPct(v)} {...axisProps(T)} />
              <YAxis domain={[0, 1]} tickFormatter={(v) => fmtPct(v)} width={44} {...axisProps(T)} />
              <Tooltip formatter={(v) => fmtPct(v, 1)} cursor={{ stroke: T.grid }} />
              <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]}
                stroke={T.axis} strokeDasharray="4 4" />
              <Line type="monotone" dataKey="o" stroke={T.forecast} strokeWidth={2.6}
                dot={{ r: 3 }} isAnimationActive={false} name="Model" />
            </LineChart>
          </ResponsiveContainer>
        </section>
      </div>}

      <RealValidation real={real} T={T} />
    </div>
  );
}

function Footer() {
  return (
    <footer className="foot">
      <p className="fine">A <a href="https://codewitheugene.top/" target="_blank"
        rel="noopener noreferrer">CodeWithEugene</a> Creation.</p>
    </footer>
  );
}
