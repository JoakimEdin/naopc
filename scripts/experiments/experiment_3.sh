# calculate the lower and upper limits using different beam sizes
bash scripts/experiments/increasing_beam_sizes/sst2.sh
bash scripts/experiments/increasing_beam_sizes/imdb.sh
bash scripts/experiments/increasing_beam_sizes/yelp.sh

# plot the beam size box plots
poetry run python reports/compare_increasing_beam_sizes.py



