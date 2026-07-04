FROM node:24-slim AS frontend

WORKDIR /src
COPY package.json package-lock.json ./
RUN npm ci
COPY index.html vite.config.ts wrangler.jsonc tsconfig*.json ./
COPY src ./src
RUN npm run build

FROM python:3.12-slim AS runtime

ENV DICT_PATH=/data/words.sqlite \
    FALLBACK=vdu \
    MIGRATIONS_DIR=/migrations \
    PYTHONUNBUFFERED=1 \
    STATIC_DIR=/app/static \
    TAGGER_URL=http://tagger:8001

WORKDIR /app
RUN pip install --no-cache-dir uv
COPY local/app/pyproject.toml ./
COPY local/app/kirciuokle ./kirciuokle
COPY migrations /migrations
RUN uv pip install --system .
COPY --from=frontend /src/dist/client ./static

EXPOSE 8000
CMD ["uvicorn", "kirciuokle.server:app", "--host", "0.0.0.0", "--port", "8000"]
