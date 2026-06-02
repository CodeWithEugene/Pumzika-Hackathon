/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "export",          // fully static -> deploys anywhere, incl. Vercel
  images: { unoptimized: true },
  trailingSlash: true,
};
module.exports = nextConfig;
