#!/bin/bash
# 

TOTAL_NUM_UPDATES=250000
WARMUP_UPDATES=50000
PEAK_LR=0.00005
TOKENS_PER_SAMPLE=8192
MAX_POSITIONS=8192
BATCH_SIZE=64
MAX_SENTENCES=4
if [ -z ${1+x} ]; then echo "task not set" && exit 1; else echo "task = ${1}"; fi
if [ -z ${2+x} ]; then echo "model not set" && exit 1; else echo "model = ${2}"; fi
CHECKPOINT_SUFFIX=nsp_${1}_$(basename ${2%.pt})
MUSICBERT_PATH=${2}

CUDA_VISIBLE_DEVICES=0 fairseq-train ${1}_data_bin --user-dir musicbert \
    --restore-file $MUSICBERT_PATH \
    --max-update $TOTAL_NUM_UPDATES \
    --batch-size $MAX_SENTENCES \
    --max-positions $MAX_POSITIONS \
    --max-tokens $((${TOKENS_PER_SAMPLE} * ${MAX_SENTENCES})) \
    --reset-optimizer --reset-dataloader --reset-meters \
    --arch musicbert_${CHECKPOINT_SUFFIX##*_} \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.1 \
    --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-6 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $PEAK_LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --log-format simple --log-interval 100 \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --shorten-method "truncate" \
    --checkpoint-suffix _${CHECKPOINT_SUFFIX} \
    --no-epoch-checkpoints \
    --disable-validation \
    --find-unused-parameters \
