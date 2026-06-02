// Small shared helpers for the dashboard.

export const fmtPct = (x, d = 0) =>
  x == null ? "–" : `${(x * 100).toFixed(d)}%`;

export const fmtDate = (s) => {
  const [y, m, d] = s.split("-").map(Number);
  const months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  return `${months[m - 1]} ${d}`;
};

export const fmtDateLong = (s) => {
  const [y, m, d] = s.split("-").map(Number);
  const months = ["January", "February", "March", "April", "May", "June",
                  "July", "August", "September", "October", "November", "December"];
  return `${months[m - 1]} ${d}, ${y}`;
};

// centred rolling mean
export function smooth(arr, w = 7) {
  const half = Math.floor(w / 2);
  return arr.map((_, i) => {
    let s = 0, n = 0;
    for (let j = Math.max(0, i - half); j <= Math.min(arr.length - 1, i + half); j++) {
      if (arr[j] != null) { s += arr[j]; n++; }
    }
    return n ? s / n : null;
  });
}

// interpolate occupancy -> warm teal heat colour
export function heatColor(v, lo = 0.45, hi = 0.85) {
  const t = Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
  // sand (#f0e7d4) -> teal (#0e7c80) -> deep (#0a4244)
  const stops = [
    [240, 231, 212],
    [124, 192, 191],
    [14, 124, 128],
    [10, 66, 68],
  ];
  const seg = t * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(seg));
  const f = seg - i;
  const c = stops[i].map((a, k) => Math.round(a + (stops[i + 1][k] - a) * f));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

export const heatText = (v) => (v > 0.66 ? "#fff" : "#3a2f1c");

// chunk daily arrays into weeks of 7 and average
export function weekly(dates, arr) {
  const out = [];
  for (let i = 0; i < dates.length; i += 7) {
    const slice = arr.slice(i, i + 7).filter((x) => x != null);
    out.push({
      label: fmtDate(dates[i]),
      value: slice.length ? slice.reduce((a, b) => a + b, 0) / slice.length : null,
    });
  }
  return out;
}
