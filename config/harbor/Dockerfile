FROM node:22-bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash ca-certificates curl git python3 unzip procps \
    && rm -rf /var/lib/apt/lists/*

# The lockfile pins the same Playwright and axe-core used by the benchmark.
# Agents can use these public tools for their own accessibility checks.
COPY package.json package-lock.json /opt/a11y-tools/
RUN cd /opt/a11y-tools && npm ci --no-audit --no-fund \
    && npx playwright install --with-deps chromium
ENV NODE_PATH=/opt/a11y-tools/node_modules
ENV PATH=/opt/a11y-tools/node_modules/.bin:$PATH

RUN mkdir -p /app/output && git init /app
WORKDIR /app
