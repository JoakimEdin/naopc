#!/bin/bash
models=('textattack/bert-base-uncased-SST-2'
    'textattack/roberta-base-SST-2'
    'textattack/bert-base-uncased-yelp-polarity'
    'VictorSanh/roberta-base-finetuned-yelp-polarity'
    'textattack/bert-base-uncased-imdb'
    'textattack/roberta-base-imdb'
)



# Loop through each model
for model in "${models[@]}"; do
    poetry run python src/evaluation/naopc_beam/compute_approximate_bounds.py --model $model --dataset_name 'sst2' --dataset_length 'short' --use_exact_limits 'True' --beam_size 5
done