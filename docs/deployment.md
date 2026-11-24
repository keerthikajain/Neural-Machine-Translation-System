# Saarthi V1 split deployment

Vercel serves the static React build. The browser sends JSON to `POST /api/translate` on Render. Render runs the existing Python server and pinned IndicTrans2 INT8 model on CPU. The local build continues to use the same-origin API.

## Render backend

Create a Web Service from the GitHub repository and select `feature/saarthi-v1-split-deploy`, **Docker**, and **Free**. Use the repository root and its `Dockerfile`. Leave the Docker command at its default and set the HTTP health check path to `/healthz`. Render supplies `PORT`. The Docker entry point binds it on `0.0.0.0` and defaults to 10000 for local container runs.

The build installs the pinned Python dependencies and downloads and checks the public model into the image. The server verifies its weights before the first inference. Instance restarts load the model from the image, so they require no runtime download or persistent disk. The Node build tools stay in a separate stage. A frontend copy in the final image also supports the original same-origin flow. The build context excludes local model weights, private artifacts, and Git metadata.

After Render assigns the HTTPS URL, open `/healthz` and check for `{"status":"ok"}`. This endpoint checks HTTP process liveness. Verify inference separately with the frontend or the API.

## Vercel frontend

Create a Hobby project for the same repository, with the production branch set to `feature/saarthi-v1-split-deploy`. Use the repository root. `vercel.json` selects the static build with `npm ci`, `npm run build:vercel`, and output in `dist`. Node.js 22 matches the Docker build.

| Host | Variable | Value |
|---|---|---|
| Vercel | `SAARTHI_API_ORIGIN` | Actual HTTPS Render origin, for example `https://your-service.onrender.com`. The URL is compiled into public browser code. |
| Render | `SAARTHI_ALLOWED_ORIGINS` | Exact public Vercel HTTPS origin. Use a comma-separated list only if you need more than one known origin. Omit trailing slashes and paths. |

After Vercel assigns the URL, set `SAARTHI_ALLOWED_ORIGINS` on Render and redeploy Render. Use the public production URL in the submission. Each additional preview URL requires explicit allowance. CORS governs browser access and does not authenticate API clients.

## Smoke test and free-tier limits

1. Open the public frontend in a signed-out browser. Translate “Anita took the train from Pune to Mumbai. The journey lasted 3 hours.” Verify that the four facts are retained.
2. Translate a different sentence and verify that the result changes. Check browser console and Render logs if the request fails.
3. Restart the Render service, wait for `/healthz` to respond, and repeat the translation. Open the app on a phone before recording.

Render Free has 512 MB RAM, spins down after 15 idle minutes, and can take about a minute to wake. Model loading adds to the first request. A local CPU check measured about 305 MiB peak process memory while serving two short translations sequentially. Host memory and response times remain to be verified. The model occupies about 214 MiB locally. The complete image size includes Python and the native dependencies and requires a Docker build to measure. Use the free plan for rehearsal and test it under the actual hosting limits before relying on it. Open the backend health URL shortly before recording if it has gone idle.

[Render free tier](https://render.com/docs/free) · [Render pricing](https://render.com/pricing) · [Render health checks](https://render.com/docs/health-checks) · [Vercel project configuration](https://vercel.com/docs/project-configuration/vercel-json)
