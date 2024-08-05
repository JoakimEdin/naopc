#!/bin/bash
models=('textattack/bert-base-uncased-SST-2'
    'textattack/roberta-base-SST-2'
    'textattack/bert-base-uncased-yelp-polarity'
    'VictorSanh/roberta-base-finetuned-yelp-polarity'
    'textattack/bert-base-uncased-imdb'
    'textattack/roberta-base-imdb'
)

cuda_device_id='1'

# Loop through each model
for model in "${models[@]}"; do
    poetry run python compute_approximate_bounds.py --model $model --cuda_id $cuda_device_id --dataset_name 'sst2' --dataset_length 'long' --use_exact_limits 'False' --beam_size 5
done