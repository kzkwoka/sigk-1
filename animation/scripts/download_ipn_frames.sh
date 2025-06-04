#!/bin/bash

set -e

# Step 1: Install gdown if needed
if ! command -v gdown &> /dev/null; then
    echo "Installing gdown..."
    pip install gdown
fi

mkdir -p frames_tgz
cd frames_tgz

# Step 2: Define updated file IDs
IDS=(
    "17yn_1n3LrMHLVSCT4zAbN6sbGXKy1GlA"
    "1OBlvjl-Z0Wr6xXnLaCGGwFUEXJQLCkDw"
    "1YCDW763mlXQlfuydFHX_EHG_hyBWC48F"
    "1I90cK_4gyyQQRkLjFn2p90CCIPV4q41Y"
    "1SY_TEQX80MtjygS8RqASqRMDDPyiENlZ"
)

# Step 3: Download each .tgz file
for id in "${IDS[@]}"; do
    echo "Downloading file ID: $id"
    gdown --id "$id"
done

# Step 4: Extract and remove each .tgz file
mkdir -p ../frames_merged
for tgz in *.tgz; do
    echo "Extracting $tgz"
    tar -xzf "$tgz" -C ../frames_merged
    rm "$tgz"
done

echo "✅ Done. All frames extracted to ./frames_merged and .tgz files removed."
