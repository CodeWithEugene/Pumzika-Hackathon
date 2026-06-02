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

const hexToRgb = (h) => {
  const n = parseInt(h.replace("#", ""), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
};

// interpolate occupancy -> heat colour across theme-provided hex stops
export function heatColor(v, stops, lo = 0.45, hi = 0.85) {
  const rgbs = stops.map(hexToRgb);
  const t = Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
  const seg = t * (rgbs.length - 1);
  const i = Math.min(rgbs.length - 2, Math.floor(seg));
  const f = seg - i;
  const c = rgbs[i].map((a, k) => Math.round(a + (rgbs[i + 1][k] - a) * f));
  return { rgb: `rgb(${c[0]},${c[1]},${c[2]})`, lum: 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2] };
}

export const heatText = (lum) => (lum > 150 ? "#1a1a1a" : "#ffffff");

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
