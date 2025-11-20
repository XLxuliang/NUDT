#!/bin/bash

docker run \
    --gpus '"device=4"' \
    -v /tmp/wind_service_backend/tmp5:/tmp/wind_service_backend/tmp/ \
    -v /home/chenxiangyu/Downloads/github/中山/NUDT_attack_vehicle/vehicle/input/input:/tmp/input \
    -e OUTPUT_PATH=/tmp/wind_service_backend/tmp\
    -e INPUT_PATH=/tmp/input\
    -e WORKERS=0\
    -e PROCESS=attack\
    -e BATCH=10\
    -e TASK=detect\
    -e DATA=coco8\
    -e MODEL=yolov8\
    -e CLASS_NUMBER=80\
    -e ATTACK_METHOD=deepfool\
    zhongshan:latest \
    python main.py 
