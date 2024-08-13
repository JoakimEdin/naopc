# evaluate AOPC of all feature attribution method and model combinations.
poetry run python src/feature_attribution_methods/calculate_feature_attribution_scores.py
poetry run python src/evaluation/evaluate_feature_attributions.py

# calculate the lower and upper limits using the NAOPC_beam
bash scripts/experiments/approximate_bounds/long/sst2.sh
bash scripts/experiments/approximate_bounds/long/imdb.sh
bash scripts/experiments/approximate_bounds/long/yelp.sh
bash scripts/experiments/approximate_bounds/short/sst2.sh
bash scripts/experiments/approximate_bounds/short/yelp.sh

# plot the ranking plots
poetry run python reports/ranking_plot_long.py
poetry run python reports/ranking_plot_short.py

# create table with correlations between AOPC and NAOPC
poetry run python reports/correlation_table.py


