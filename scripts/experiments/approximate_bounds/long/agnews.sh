#!/bin/bash
models=('textattack/distilbert-base-uncased-ag-news'
    'textattack/bert-base-uncased-ag-news'
    'textattack/roberta-base-ag-news'
)



# Loop through each model
for model in "${models[@]}"; do
    poetry run python src/evaluation/naopc_beam/compute_approximate_bounds.py --model $model --dataset_name 'ag_news' --dataset_length 'long' --use_exact_limits 'False' --beam_size 1000
done