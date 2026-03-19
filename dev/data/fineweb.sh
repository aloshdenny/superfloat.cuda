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
elif [ "$DATASET_SIZE" == "10B" ]; then
    MAX_AVAILABLE_SHARDS=102
    TRAIN_BASE_URL="https://huggingface.co/datasets/chrisdryden/FineWebTokenizedGPT2/resolve/main/fineweb10B_train_"
    VAL_URL="https://huggingface.co/datasets/chrisdryden/FineWebTokenizedGPT2/resolve/main/fineweb10B_val_000000.bin?download=true"
    SAVE_DIR="fineweb10B"
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
    local FILE_NAME=$(basename "$FILE_URL" | cut -d'?' -f1)
    local FILE_PATH="${SAVE_DIR}/${FILE_NAME}"

    curl -s -L -o "$FILE_PATH" "$FILE_URL"
    echo "Downloaded $FILE_NAME"
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
download "$VAL_URL" &

# -------------------------------
# Train shards
# -------------------------------
train_commands=()
for i in $(seq -f "%06g" 1 $MAX_SHARDS); do
    FILE_URL="${TRAIN_BASE_URL}${i}.bin?download=true"
    train_commands+=("download \"$FILE_URL\"")
done

run_in_parallel 40 "${train_commands[@]}"

echo "Downloaded $DATASET_SIZE ($MAX_SHARDS shards) into $SAVE_DIR"