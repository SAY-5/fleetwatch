# Build the C++ aggregator, then assemble the Python service that invokes it.
# Build context is the repository root.
FROM gcc:13 AS agg
WORKDIR /src
RUN apt-get update && apt-get install -y --no-install-recommends cmake && \
    rm -rf /var/lib/apt/lists/*
COPY apps/aggregator /src/apps/aggregator
RUN cmake -S /src/apps/aggregator -B /build -DCMAKE_BUILD_TYPE=Release -DFWAGG_BUILD_TESTS=OFF && \
    cmake --build /build -j

FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    FLEETWATCH_AGGREGATOR=/usr/local/bin/fleetwatch-aggregator
WORKDIR /repo
COPY --from=agg /build/fleetwatch-aggregator /usr/local/bin/fleetwatch-aggregator
COPY README.md /repo/README.md
COPY apps/metrics /repo/apps/metrics
WORKDIR /repo/apps/metrics
RUN pip install --no-cache-dir poetry==2.1.3 && \
    poetry config virtualenvs.create false && \
    poetry install --only main --no-interaction --no-ansi
EXPOSE 8000
CMD ["uvicorn", "fleetwatch.api:app", "--host", "0.0.0.0", "--port", "8000"]
