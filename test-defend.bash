#!/bin/bash
set -euo pipefail

# Image to run (replace with your image/tag or SHA as needed)
IMAGE="nudt_yolo:latest"

# GPU device to use (format matches --gpus "device=X")
GPU_DEVICE="0"

# Host paths
HOST_INPUT="/data6/user23215430/nudt/vehicle_yolo/input"
HOST_OUTPUT="/data6/user23215430/nudt/vehicle_yolo/output"

mkdir -p "${HOST_OUTPUT}"

methods=("neural_cleanse" "pgd" "fgsm")

for m in "${methods[@]}"; do
  echo "Running defend method: ${m}"
  docker run \
    --rm \
    --gpus "device=${GPU_DEVICE}" \
    -v "${HOST_OUTPUT}":/tmp/output \
    -v "${HOST_INPUT}":/tmp/input \
    -e OUTPUT_PATH=/tmp/output \
    -e INPUT_PATH=/tmp/input \
    -e WORKERS=0 \
    -e PROCESS=defend \
    -e BATCH=1 \
    -e TASK=detect \
    -e DATA=coco8 \
    -e MODEL=yolov8 \
    -e CLASS_NUMBER=80 \
    -e DEFEND_METHOD=${m} \
    "${IMAGE}" \
    python main.py

  # Check output
  out_dir="${HOST_OUTPUT}/clean_images"
  if [ -d "${out_dir}" ] && [ "$(ls -1 "${out_dir}" 2>/dev/null | wc -l)" -gt 0 ]; then
    echo "[OK] ${m}: clean images generated in ${out_dir}"
  else
    echo "[WARN] ${m}: no outputs found in ${out_dir}"
  fi
done

echo "All defend tests completed."


