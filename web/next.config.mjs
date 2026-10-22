/** Static export (D9): Caddy serves the files; no Node server in production. */
const nextConfig = {
  output: "export",
  reactStrictMode: true,
  images: { unoptimized: true },
  // Always defined, so `NEXT_PUBLIC_MOCK === "1"` is a build-time constant and production
  // bundles drop the mock backend entirely.
  env: { NEXT_PUBLIC_MOCK: process.env.NEXT_PUBLIC_MOCK ?? "0" },
};

export default nextConfig;
