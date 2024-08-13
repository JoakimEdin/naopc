# calculate the AOPC limits using exhaustive search
poetry run python src/evaluation/naopc_exact/calculate_permutations.py
poetry run python src/evaluation/naopc_exact/calculate_aopc_limits.py

# plot the limits
poetry run python reports/plot_aopc_limits.py