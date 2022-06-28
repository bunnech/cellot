#! /bin/bash

for mode in ood iid; do
for sample in $(cat datasets/scrna-lupuspatients/sample_ids.txt); do
for model in identity random cae scgen; do
    echo \
        python ./scripts/train.py \
        --outdir ./results/oos-lupuspatients/holdout-${sample}/mode-${mode}/model-${model} \
        --config ./configs/tasks/oos-lupuspatients.yaml \
        --config ./configs/models/${model}.yaml \
        --config.datasplit.holdout $sample \
        --config.datasplit.mode $mode;
done;
done;
done;


# This is to use the same encoder as scgen
for mode in ood iid; do
for sample in $(cat datasets/scrna-lupuspatients/sample_ids.txt); do
    echo \
        python ./scripts/train.py \
        --outdir ./results/oos-lupuspatients/holdout-${sample}/mode-${mode}/model-cellot \
        --config ./configs/tasks/oos-lupuspatients.yaml \
        --config ./configs/models/cellot.yaml \
        --config.datasplit.holdout $sample \
        --config.datasplit.mode $mode \
        --config.data.ae_emb.path ./results/oos-lupuspatients/holdout-${sample}/mode-${mode}/model-scgen;
done;
done
