#!/bin/bash
models=('textattack/bert-base-uncased-snli'
    'textattack/distilbert-base-cased-snli'
    'varun-v-rao/gpt2-snli-model1'
)



# Loop through each model
for model in "${models[@]}"; do
    poetry run python src/evaluation/naopc_beam/compute_approximate_bounds_subsets.py --model $model --dataset_name 'snli' --dataset_length 'long' --use_exact_limits 'False' --beam_size 1
    poetry run python src/evaluation/naopc_beam/compute_approximate_bounds_subsets.py --model $model --dataset_name 'snli' --dataset_length 'long' --use_exact_limits 'False' --beam_size 2
    poetry run python src/evaluation/naopc_beam/compute_approximate_bounds_subsets.py --model $model --dataset_name 'snli' --dataset_length 'long' --use_exact_limits 'False' --beam_size 5
    poetry run python src/evaluation/naopc_beam/compute_approximate_bounds_subsets.py --model $model --dataset_name 'snli' --dataset_length 'long' --use_exact_limits 'False' --beam_size 10
    poetry run python src/evaluation/naopc_beam/compute_approximate_bounds_subsets.py --model $model --dataset_name 'snli' --dataset_length 'long' --use_exact_limits 'False' --beam_size 50
done