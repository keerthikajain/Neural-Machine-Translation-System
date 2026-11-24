FROM node:22-bookworm-slim AS frontend
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY saarthi_app/app.jsx saarthi_app/app.jsx
RUN npm run build

FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements-app.txt ./
RUN python -m pip install --no-cache-dir -r requirements-app.txt
COPY saarthi_app ./saarthi_app
COPY public/illustrations ./public/illustrations
COPY --from=frontend /app/saarthi_app/static/app.js ./saarthi_app/static/app.js
RUN python -m saarthi_app.setup_model

USER 10001:10001
EXPOSE 10000
CMD ["sh", "-c", "exec python -m saarthi_app --host 0.0.0.0 --port \"${PORT:-10000}\""]
