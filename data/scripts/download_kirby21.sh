#!/usr/bin/env bash
set -euo pipefail

RAW_DIR="/home/ubuntu/yolo_sxc/data/raw"
TMP_DIR="/tmp/kirby21_download"
BASE_URL="https://multimodal.projects.nitrc.org/downloads/kki2009"
MAX_PARALLEL=4

mkdir -p "$RAW_DIR" "$TMP_DIR"

download_and_extract() {
    local session_id=$(printf "%02d" "$1")
    local archive="KKI2009-${session_id}.tar.bz2"
    local target_nii="KKI2009-${session_id}-MPRAGE.nii"
    local output_gz="${RAW_DIR}/${target_nii}.gz"

    if [[ -f "$output_gz" ]]; then
        echo "[SKIP] ${output_gz} already exists"
        return 0
    fi

    echo "[DOWN] Downloading session ${session_id}..."
    curl -sL -o "${TMP_DIR}/${archive}" "${BASE_URL}/${archive}"

    echo "[EXTR] Extracting MPRAGE from session ${session_id}..."
    tar xjf "${TMP_DIR}/${archive}" -C "$TMP_DIR" "$target_nii"

    echo "[GZIP] Compressing ${target_nii}..."
    gzip -c "${TMP_DIR}/${target_nii}" > "$output_gz"

    rm -f "${TMP_DIR}/${archive}" "${TMP_DIR}/${target_nii}"
    echo "[DONE] Session ${session_id} -> $(du -h "$output_gz" | cut -f1)"
}

echo "=== Kirby21 MPRAGE Download ==="
echo "Target: ${RAW_DIR}"
echo "Sessions: 01-42 (21 subjects x 2 scans)"
echo ""

running=0
for i in $(seq 1 42); do
    download_and_extract "$i" &
    running=$((running + 1))
    if [[ $running -ge $MAX_PARALLEL ]]; then
        wait -n
        running=$((running - 1))
    fi
done
wait

echo ""
echo "=== Download Complete ==="
ls -lh "$RAW_DIR"/*.nii.gz 2>/dev/null | wc -l
echo "files downloaded to ${RAW_DIR}"
rm -rf "$TMP_DIR"
