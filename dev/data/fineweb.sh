#!/bin/bash

# Usage:
# ./fineweb.sh -100 [MAX_SHARDS]
# ./fineweb.sh -10  [MAX_SHARDS]

# -------------------------------
# Parse dataset size flag
# -------------------------------
DATASET_SIZE="100B"  # default

for arg in "$@"; do
    case $arg in
        -10)
            DATASET_SIZE="10B"
            shift
            ;;
        -100)
            DATASET_SIZE="100B"
            shift
            ;;
    esac
done

# -------------------------------
# Set shard limits based on dataset
# -------------------------------
if [ "$DATASET_SIZE" == "100B" ]; then
    MAX_AVAILABLE_SHARDS=1028
    TRAIN_BASE_URL="https://huggingface.co/datasets/chrisdryden/FineWebTokenizedGPT2/resolve/main/fineweb_train_"
    VAL_URL="https://huggingface.co/datasets/chrisdryden/FineWebTokenizedGPT2/resolve/main/fineweb_val_000000.bin?download=true"
    SAVE_DIR="fineweb100B"
    FILE_PREFIX="fineweb"
elif [ "$DATASET_SIZE" == "10B" ]; then
    MAX_AVAILABLE_SHARDS=102
    TRAIN_BASE_URL="https://huggingface.co/datasets/chrisdryden/FineWebTokenizedGPT2/resolve/main/fineweb_train_"
    VAL_URL="https://huggingface.co/datasets/chrisdryden/FineWebTokenizedGPT2/resolve/main/fineweb_val_000000.bin?download=true"
    SAVE_DIR="fineweb10B"
    FILE_PREFIX="fineweb"
else
    echo "Invalid dataset size. Use -10 or -100"
    exit 1
fi

# -------------------------------
# Optional positional MAX_SHARDS
# -------------------------------
if [[ "$1" =~ ^[0-9]+$ ]]; then
    MAX_SHARDS=$1
else
    MAX_SHARDS=$MAX_AVAILABLE_SHARDS
fi

# Clamp to max allowed
if [ "$MAX_SHARDS" -gt "$MAX_AVAILABLE_SHARDS" ]; then
    MAX_SHARDS=$MAX_AVAILABLE_SHARDS
fi

# -------------------------------
# Setup
# -------------------------------
mkdir -p "$SAVE_DIR"

download() {
    local FILE_URL=$1
    local SAVE_DIR=$2
    local FILE_NAME=$(basename "$FILE_URL" | cut -d'?' -f1)
    local FILE_PATH="${SAVE_DIR}/${FILE_NAME}"

    # Skip if already downloaded and >1MB (not a failed partial download)
    if [ -f "$FILE_PATH" ] && [ $(stat -c%s "$FILE_PATH") -gt 1048576 ]; then
        echo "Skipping $FILE_NAME (already exists)"
        return
    fi

    curl -s -L -o "$FILE_PATH" "$FILE_URL"
    local SIZE=$(stat -c%s "$FILE_PATH" 2>/dev/null || echo 0)
    if [ "$SIZE" -lt 1048576 ]; then
        echo "WARNING: $FILE_NAME looks too small ($SIZE bytes) - download may have failed"
    else
        echo "Downloaded $FILE_NAME ($((SIZE / 1048576)) MB)"
    fi
}

run_in_parallel() {
    local max_jobs=$1
    shift
    local commands=("$@")
    local job_count=0

    for cmd in "${commands[@]}"; do
        eval "$cmd" &
        ((job_count++))
        if (( job_count >= max_jobs )); then
            wait -n
            ((job_count--))
        fi
    done
    wait
}

export -f download

# -------------------------------
# Download validation shard
# -------------------------------
download "$VAL_URL" "$SAVE_DIR" &

# -------------------------------
# Train shards
# -------------------------------
train_commands=()
for i in $(seq -f "%06g" 1 $MAX_SHARDS); do
    FILE_URL="${TRAIN_BASE_URL}${i}.bin?download=true"
    train_commands+=("download \"$FILE_URL\" \"$SAVE_DIR\"")
done

run_in_parallel 40 "${train_commands[@]}"

echo "Downloaded $DATASET_SIZE ($MAX_SHARDS shards) into $SAVE_DIR"