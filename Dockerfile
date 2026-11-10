# syntax=docker/dockerfile:1
# Minimal image for optimized_dp, with the exact package versions the toolbox is tested with
# (HeteroCL 0.3, LLVM 6.0, Python 3.8).
#
#   Build:          docker build -t optimized_dp .
#   Run the tests:  docker run --rm optimized_dp pytest test
#   Work in a shell with your local checkout mounted:
#                   docker run --rm -it -v "$PWD:/opt/optimized_dp" optimized_dp bash
#
#   Build from the archived conda packages instead of the conda channels (no conda network access
#   needed; get conda-pkgs/ from the archive or with scripts/download_conda_packages.sh):
#                   docker build --build-context conda-pkgs=./conda-pkgs -t optimized_dp .
#
#   Archive / restore the built image:
#                   docker save optimized_dp | gzip > optimized_dp.tar.gz
#                   docker load < optimized_dp.tar.gz

# Empty by default; replaced by the archived packages when built with --build-context conda-pkgs=...
FROM scratch AS conda-pkgs

# Ubuntu 22.04 (jammy), pinned by digest
FROM mambaorg/micromamba:1.5.10-jammy@sha256:210b5453f73baf5dce1f72ec269564c11fa6bcc60fa7f4cc8b883e7e3272c6e4

# HeteroCL's runtime links against the system libtinfo.so.5, which Ubuntu 22.04 still ships
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends libtinfo5 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /opt/optimized_dp \
    && chown $MAMBA_USER:$MAMBA_USER /opt/optimized_dp
USER $MAMBA_USER

# Exact conda packages, exported with `conda list --explicit --md5` from the tested environment.
# Packages found in the conda-pkgs build context are installed from there, the rest are downloaded.
COPY --chown=$MAMBA_USER:$MAMBA_USER environment-linux-64.lock /tmp/environment-linux-64.lock
RUN --mount=type=bind,from=conda-pkgs,target=/tmp/conda-pkgs \
    while read -r line; do \
        case "$line" in \
            https://*) url="${line%%#*}"; \
                       f="/tmp/conda-pkgs/$(basename "$(dirname "$url")")/$(basename "$url")"; \
                       if [ -f "$f" ]; then line="file://$f#${line##*#}"; fi ;; \
        esac; \
        echo "$line"; \
    done < /tmp/environment-linux-64.lock > /tmp/environment-resolved.lock \
    && echo "$(grep -c '^file://' /tmp/environment-resolved.lock) of" \
            "$(grep -c '://' /tmp/environment-resolved.lock) conda packages taken from the local archive" \
    && micromamba install --name base --yes --file /tmp/environment-resolved.lock \
    && micromamba clean --all --yes

# Run the following commands inside the conda environment
ARG MAMBA_DOCKERFILE_ACTIVATE=1

WORKDIR /opt/optimized_dp
COPY --chown=$MAMBA_USER:$MAMBA_USER . .
RUN pip install --no-cache-dir pytest==8.3.5 \
    && pip install --no-cache-dir --no-deps -e .
