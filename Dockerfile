FROM ubuntu:24.04 AS builder
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        clang \
        curl \
        libclang-dev \
    && rm -rf /var/lib/apt/lists/*

ENV RUSTUP_HOME=/usr/local/rustup
ENV CARGO_HOME=/usr/local/cargo
ENV PATH=/usr/local/cargo/bin:${PATH}
ENV LIBCLANG_PATH=/usr/lib/llvm-16/lib

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain 1.89.0

COPY .cargo ./.cargo
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY configs ./configs
COPY artifacts ./artifacts
COPY start.sh ./start.sh
COPY README.md ./README.md

RUN cargo build --release

FROM ubuntu:24.04 AS runtime
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgcc-s1 \
        libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/Prismguand-Rust /app/Prismguand-Rust
COPY --from=builder /app/configs /app/configs
COPY --from=builder /app/artifacts /app/artifacts
COPY --from=builder /app/start.sh /app/start.sh
COPY --from=builder /app/README.md /app/README.md

EXPOSE 8080

ENTRYPOINT ["/app/Prismguand-Rust"]
