#! /bin/bash

for drug in $(cat datasets/4i/drugs.txt); do
for model in cellot cae scgen identity random; do
    if [ $model == cae ]; then
        model_config=./configs/models/cae-4i.yaml
    elif [ $model == scgen ]; then
        model_config=./configs/models/scgen-4i.yaml
    else
        model_config=./configs/models/${model}.yaml
    fi 

    echo \
        python ./scripts/train.py \
        --outdir ./results/4i/drug-${drug}/model-${model} \
        --config ./configs/tasks/4i.yaml \
        --config $model_config \
        --config.data.target $drug 
done;
done


for drug in $(cat datasets/scrna-sciplex3/drugs.txt); do
for model in cae scgen identity random; do
    echo \
        python ./scripts/train.py \
        --outdir ./results/scrna-sciplex3/drug-${drug}/model-${model} \
        --config ./configs/tasks/sciplex3.yaml \
        --config ./configs/models/${model}.yaml \
        --config.data.target $drug 
done;
done

# This is to use the same encoder as scgen
for drug in $(cat datasets/scrna-sciplex3/drugs.txt); do
    echo \
        python ./scripts/train.py \
        --outdir ./results/scrna-sciplex3/drug-${drug}/model-${model} \
        --config ./configs/tasks/sciplex3.yaml \
        --config ./configs/models/${model}.yaml \
        --config.data.target $drug \
        --config.data.ae_emb.path ./results/scrna-sciplex3/drug-${drug}/model-scgen
done
