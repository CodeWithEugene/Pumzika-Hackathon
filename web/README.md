# Pumzika Demand Radar — Web (Next.js, Vercel-ready)

A fully static Next.js dashboard that visualises the Track-02 occupancy forecast.
It reads precomputed JSON exported from the Python model (`web/public/data/`), so
there's **no backend** — it deploys anywhere static, and is first-class on Vercel.

## Local development

```bash
cd web
npm install
npm run dev        # http://localhost:3000
```

## Production build (static export)

```bash
npm run build      # outputs a static site to web/out/
npx serve out      # preview the production build locally
```

## Refreshing the data

The committed JSON in `public/data/` is generated from the model. To regenerate
after re-running the Python pipeline (`src/train.py` + `src/forecast.py`):

```bash
./prepare-data.sh        # runs src/export_web.py + copies figures
```

## Deploy to Vercel

This app lives in the **`web/` subdirectory**, so point Vercel at it.

### Option A — GitHub (recommended)
1. Push the repo to GitHub.
2. On [vercel.com](https://vercel.com) → **Add New… → Project** → import the repo.
3. Set **Root Directory = `web`** (Vercel auto-detects Next.js and `output: export`).
4. **Deploy.** Done — you get a `*.vercel.app` URL.

### Option B — Vercel CLI
```bash
cd web
npx vercel            # first run: log in + link the project
npx vercel --prod     # deploy to production
```

No environment variables, no config needed — `next.config.js` already sets
`output: "export"`.

## Stack
- Next.js 14 (App Router, static export) · React 18 · Recharts
- Fonts: Fraunces (display) + Hanken Grotesk (body)
- Design: "savanna-luxe" — warm sand canvas, teal / gold / terracotta accents
