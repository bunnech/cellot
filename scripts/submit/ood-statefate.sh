#! /bin/bash

for mode in ood iid; do
for population in $(cat datasets/scrna-statefate/populations.txt); do
    for model in identity random scgen cae; do
        echo \
            python ./scripts/train.py \
            --outdir ./results/ood-statefate/holdout-${population}/mode-${mode}/model-${model} \
            --config ./configs/tasks/ood-statefate.yaml \
            --config ./configs/models/${model}.yaml \
            --config.datasplit.holdout $population \
            --config.datasplit.mode $mode;
    done;
done;
done

# This is to use the same encoder as scgen
for mode in ood iid; do
for population in $(cat datasets/scrna-statefate/populations.txt); do
    echo \
        python ./scripts/train.py \
        --outdir ./results/ood-statefate/holdout-${population}/mode-${mode}/model-cellot \
        --config ./configs/tasks/ood-statefate.yaml \
        --config ./configs/models/cellot.yaml \
        --config.datasplit.holdout $population \
        --config.datasplit.mode $mode \
        --config.data.ae_emb.path ./results/ood-statefate/holdout-${population}/mode-${mode}/model-scgen;
done;
done
