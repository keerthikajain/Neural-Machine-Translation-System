import { copyFile, mkdir } from "node:fs/promises";
import { build } from "esbuild";

const configuredOrigin = process.env.SAARTHI_API_ORIGIN?.trim();
if (!configuredOrigin) {
  throw new Error("Set SAARTHI_API_ORIGIN to the Render API origin, such as https://your-service.onrender.com.");
}
const api = new URL(configuredOrigin);
const isLocal = api.protocol === "http:" && ["localhost", "127.0.0.1"].includes(api.hostname);
if ((api.protocol !== "https:" && !isLocal) ||
    ![api.origin, api.origin + "/"].includes(configuredOrigin)) {
  throw new Error("SAARTHI_API_ORIGIN must be an HTTPS origin with no path or credentials. Loopback HTTP is allowed for local testing.");
}

const assets = {
  "index.html": "saarthi_app/index.html",
  "styles.css": "saarthi_app/styles.css",
  "logo.svg": "public/illustrations/saarthi-logo.svg",
  "skyline-desktop.png": "public/illustrations/desktop-landscape-fixed(1).png",
  "skyline-mobile.png": "public/illustrations/splash-landscape.png",
};
await mkdir("dist", { recursive: true });
await build({
  entryPoints: ["saarthi_app/app.jsx"],
  outfile: "dist/app.js",
  bundle: true,
  minify: true,
  jsx: "automatic",
  target: "es2020",
  define: { SAARTHI_API_ORIGIN: JSON.stringify(api.origin) },
});
await Promise.all(Object.entries(assets).map(([name, source]) => copyFile(source, "dist/" + name)));
