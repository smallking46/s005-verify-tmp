#!/bin/sh
# Competition contract (streaming compression): /bin/sh /app/run.sh INPUT_DIR OUTPUT_DIR
# Every regular, non-hidden file in INPUT_DIR is one item (FFV1/Matroska, sha256 name, no extension);
# one H.264/HEVC/VP9/AV1 file per item is written to OUTPUT_DIR under the same name.
# The streaming rules of sn85-compression-005 (manifest "streaming_rules"):
BITS_PER_PIXEL=0.05
MAX_KEYFRAME_INTERVAL=2.0
PEAK_WINDOW=2.0
PEAK_RATIO=1.5
# No `set -e`: a failing item must never abort the batch; the script always exits 0.
set -u
umask 022
PATH=/usr/local/bin:/usr/bin:/bin
PYTHONDONTWRITEBYTECODE=1
export PATH PYTHONDONTWRITEBYTECODE
[ "$#" -eq 2 ] || { echo "usage: /app/run.sh INPUT_DIR OUTPUT_DIR" >&2; exit 64; }
app_dir=${CONTENDER_APP_DIR:-/app}
python3 "$app_dir/encode.py" --bits-per-pixel "$BITS_PER_PIXEL" --max-keyframe-interval "$MAX_KEYFRAME_INTERVAL" \
    --peak-window "$PEAK_WINDOW" --peak-ratio "$PEAK_RATIO" "$1" "$2"
rm -f -- "$2"/.partial-* 2>/dev/null
exit 0
