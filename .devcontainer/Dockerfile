FROM almalinux:9

# Install basic dependencies
RUN dnf upgrade -y ca-certificates \
    && dnf install -y --allowerasing \
    git \
    curl \
    wget \
    which \
    tar \
    gzip \
    bzip2 \
    ca-certificates \
    && dnf clean all

# Install OpenCode
RUN curl -fsSL https://opencode.ai/install | bash

# Install Pixi with the emSign CA security fix
RUN curl -fsSL https://pixi.sh/install.sh | PIXI_VERSION=0.82.1 sh

# Add Pixi to PATH
ENV PATH="/root/.pixi/bin:${PATH}"

# Set working directory
WORKDIR /workspace

# Verify installations
RUN pixi --version
