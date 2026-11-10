#!/usr/bin/env bash
# Download every conda package listed in environment-linux-64.lock into conda-pkgs/
# and check each file against the md5 recorded in the lock file.
#
# The resulting directory is what we archive (e.g. on Zenodo) so that the Docker image can be
# rebuilt even if the cornell-zhang (HeteroCL) or Anaconda channels drop these old packages.
# When conda-pkgs/ is populated, `docker build` installs from it instead of the network.
#
#   Usage:  bash scripts/download_conda_packages.sh
#   Then:   tar -czf odp-conda-pkgs-linux-64.tar.gz conda-pkgs
set -euo pipefail

cd "$(dirname "$0")/.."
LOCK=environment-linux-64.lock
DEST=conda-pkgs
mkdir -p "$DEST"

grep '^https://' "$LOCK" | while read -r line; do
    url="${line%%#*}"
    md5="${line##*#}"
    subdir="$(basename "$(dirname "$url")")"   # linux-64 or noarch, the layout micromamba expects
    mkdir -p "$DEST/$subdir"
    file="$DEST/$subdir/$(basename "$url")"
    if [ ! -f "$file" ] || [ "$(md5sum "$file" | cut -d' ' -f1)" != "$md5" ]; then
        echo "Downloading $(basename "$url")"
        curl -sSfL --retry 3 -o "$file" "$url"
    fi
    if [ "$(md5sum "$file" | cut -d' ' -f1)" != "$md5" ]; then
        echo "md5 mismatch for $file" >&2
        exit 1
    fi
done

echo "All $(grep -c '^https://' "$LOCK") packages present and verified in $DEST/"
