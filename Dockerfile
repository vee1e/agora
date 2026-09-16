FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc libc6-dev binutils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

USER 65534:65534

CMD ["sleep", "infinity"]
