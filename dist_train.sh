#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
arg3=$3

CHECKPOINT=""
if [[ $arg3 == "--checkpoint" ]]; then
    CHECKPOINT=$4
    shift 4
else
    shift 2
fi

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29503}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    ${CHECKPOINT:+--checkpoint $CHECKPOINT} \
    --launcher pytorch "$@"
