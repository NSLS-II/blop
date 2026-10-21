FROM ghcr.io/prefix-dev/pixi:0.82.1

# Refresh the system trust store even when the base image predates a CA update.
RUN apt-get update \
    && apt-get install -y --only-upgrade ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy the pixi project definition and install
COPY pixi.toml /app/pixi.toml
RUN pixi install

# Copy application scripts
COPY startup.py /app/startup.py
COPY zmq_tiled_bridge.py /app/zmq_tiled_bridge.py

ENTRYPOINT ["pixi", "run"]
