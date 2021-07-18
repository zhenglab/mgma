#!/bin/bash

DATA_PATH=path_to_csv_files
MGMA_TYPE=TSA
MGMA_NUM_GROUPS=8

python tools/run_net.py \
--cfg configs/Ucf101/UCF101_MGMA.yaml \
DATA.PATH_TO_DATA_DIR ${DATA_PATH} \
MGMA.TYPE ${MGMA_TYPE} \
MGMA.NUM_GROUPS ${MGMA_NUM_GROUPS}
