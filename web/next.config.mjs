/** Static export (D9): Caddy serves the files; no Node server in production. */
const nextConfig = {
  output: "export",
  reactStrictMode: true,
  images: { unoptimized: true },
};

export default nextConfig;
