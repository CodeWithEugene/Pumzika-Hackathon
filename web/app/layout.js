import { GeistSans } from "geist/font/sans";
import { GeistMono } from "geist/font/mono";
import "./globals.css";

export const metadata = {
  title: "Pumzika Demand Radar · Occupancy & Demand Forecasting",
  description:
    "A 90-day occupancy forecast for every Pumzika host across Tanzania, Kenya and Uganda. Pumzika Hackathon 2026 — Track 02.",
};

// Set the theme before first paint to avoid a flash of the wrong colours.
const themeScript = `
(function(){try{
  var p = new URLSearchParams(location.search).get('theme');
  var m = (p === 'dark' || p === 'light' || p === 'system') ? p
        : (localStorage.getItem('pumzika-theme') || 'system');
  var sys = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  var t = m === 'system' ? sys : m;
  var r = document.documentElement;
  r.setAttribute('data-theme', t);
  r.style.colorScheme = t;
}catch(e){}})();
`;

export default function RootLayout({ children }) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${GeistSans.variable} ${GeistMono.variable}`}
    >
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body>{children}</body>
    </html>
  );
}
