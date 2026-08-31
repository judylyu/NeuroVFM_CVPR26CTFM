#!/bin/bash
set -e

# Default paths for Docker environment
INPUT_DIR="${INPUT_DIR:-/workspace/inputs}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/outputs}"
MASKS_DIR="${MASKS_DIR:-}"  # Required masks directory
CHECKPOINT="${CHECKPOINT:-./checkpoints/neurovfm-encoder}"

if [ -z "$MASKS_DIR" ]; then
    echo "MASKS_DIR is required for ROI extraction" >&2
    exit 1
fi

CMD="python3 extract_feat_LP_ROI.py -i \"$INPUT_DIR\" -o \"$OUTPUT_DIR\" --checkpoint \"$CHECKPOINT\" --masks_path \"$MASKS_DIR\""

# Run feature extraction
eval $CMD
