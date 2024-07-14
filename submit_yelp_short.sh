#!/bin/bash
models=('textattack/bert-base-uncased-SST-2'
    'textattack/roberta-base-SST-2'
    'textattack/bert-base-uncased-yelp-polarity'
    'VictorSanh/roberta-base-finetuned-yelp-polarity'
    'textattack/bert-base-uncased-imdb'
    'textattack/roberta-base-imdb'
)

cuda_device_id='4'

# Loop through each model
for model in "${models[@]}"; do
    poetry run python compute_approximate_bounds.py --model $model --cuda_id $cuda_device_id --dataset_name 'yelp' --explanation_attributions 'gradient_x_input' --beam_size 1
    poetry run python compute_approximate_bounds.py --model $model --cuda_id $cuda_device_id --dataset_name 'yelp' --explanation_attributions 'gradient_x_input' --beam_size 5
    poetry run python compute_approximate_bounds.py --model $model --cuda_id $cuda_device_id --dataset_name 'yelp' --explanation_attributions 'gradient_x_input' --beam_size 10
    poetry run python compute_approximate_bounds.py --model $model --cuda_id $cuda_device_id --dataset_name 'yelp' --explanation_attributions 'gradient_x_input' --beam_size 20
    poetry run python compute_approximate_bounds.py --model $model --cuda_id $cuda_device_id --dataset_name 'yelp' --explanation_attributions 'gradient_x_input' --beam_size 50
    poetry run python compute_approximate_bounds.py --model $model --cuda_id $cuda_device_id --dataset_name 'yelp' --explanation_attributions 'deeplift' --beam_size 1
    poetry run python compute_approximate_bounds.py --model $model --cuda_id $cuda_device_id --dataset_name 'yelp' --explanation_attributions 'deeplift' --beam_size 5
    poetry run python compute_approximate_bounds.py --model $model --cuda_id $cuda_device_id --dataset_name 'yelp' --explanation_attributions 'deeplift' --beam_size 10
    poetry run python compute_approximate_bounds.py --model $model --cuda_id $cuda_device_id --dataset_name 'yelp' --explanation_attributions 'deeplift' --beam_size 20
    poetry run python compute_approximate_bounds.py --model $model --cuda_id $cuda_device_id --dataset_name 'yelp' --explanation_attributions 'deeplift' --beam_size 50
    poetry run python compute_approximate_bounds.py --model $model --cuda_id $cuda_device_id --dataset_name 'yelp' --explanation_attributions 'decompx' --beam_size 1
    poetry run python compute_approximate_bounds.py --model $model --cuda_id $cuda_device_id --dataset_name 'yelp' --explanation_attributions 'decompx' --beam_size 5
    poetry run python compute_approximate_bounds.py --model $model --cuda_id $cuda_device_id --dataset_name 'yelp' --explanation_attributions 'decompx' --beam_size 10
    poetry run python compute_approximate_bounds.py --model $model --cuda_id $cuda_device_id --dataset_name 'yelp' --explanation_attributions 'decompx' --beam_size 20
    poetry run python compute_approximate_bounds.py --model $model --cuda_id $cuda_device_id --dataset_name 'yelp' --explanation_attributions 'decompx' --beam_size 50
    poetry run python compute_approximate_bounds.py --model $model --cuda_id $cuda_device_id --dataset_name 'yelp' --beam_size 1
    poetry run python compute_approximate_bounds.py --model $model --cuda_id $cuda_device_id --dataset_name 'yelp' --beam_size 5
    poetry run python compute_approximate_bounds.py --model $model --cuda_id $cuda_device_id --dataset_name 'yelp' --beam_size 10
    poetry run python compute_approximate_bounds.py --model $model --cuda_id $cuda_device_id --dataset_name 'yelp' --beam_size 20
    poetry run python compute_approximate_bounds.py --model $model --cuda_id $cuda_device_id --dataset_name 'yelp' --beam_size 50
done