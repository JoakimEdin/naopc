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
    CUDA_VISIBLE_DEVICES=5 poetry run python src/evaluation/naopc_beam/compute_approximate_bounds_subsets.py --model $model --dataset_name 'yelp' --dataset_length 'long' --use_exact_limits 'False' --beam_size 1
    CUDA_VISIBLE_DEVICES=5 poetry run python src/evaluation/naopc_beam/compute_approximate_bounds_subsets.py --model $model --dataset_name 'yelp' --dataset_length 'long' --use_exact_limits 'False' --beam_size 2
    CUDA_VISIBLE_DEVICES=5 poetry run python src/evaluation/naopc_beam/compute_approximate_bounds_subsets.py --model $model --dataset_name 'yelp' --dataset_length 'long' --use_exact_limits 'False' --beam_size 5
    CUDA_VISIBLE_DEVICES=5 poetry run python src/evaluation/naopc_beam/compute_approximate_bounds_subsets.py --model $model --dataset_name 'yelp' --dataset_length 'long' --use_exact_limits 'False' --beam_size 10
    CUDA_VISIBLE_DEVICES=5 poetry run python src/evaluation/naopc_beam/compute_approximate_bounds_subsets.py --model $model --dataset_name 'yelp' --dataset_length 'long' --use_exact_limits 'False' --beam_size 50
done