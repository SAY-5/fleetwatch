# Build the C++ aggregator and ship a small runtime image.
FROM gcc:13 AS build
WORKDIR /src
RUN apt-get update && apt-get install -y --no-install-recommends cmake && \
    rm -rf /var/lib/apt/lists/*
COPY . /src/apps/aggregator
RUN cmake -S /src/apps/aggregator -B /build -DCMAKE_BUILD_TYPE=Release -DFWAGG_BUILD_TESTS=OFF && \
    cmake --build /build -j

FROM debian:bookworm-slim AS runtime
COPY --from=build /build/fleetwatch-aggregator /usr/local/bin/fleetwatch-aggregator
ENTRYPOINT ["/usr/local/bin/fleetwatch-aggregator"]
