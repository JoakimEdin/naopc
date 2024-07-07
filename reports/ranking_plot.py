from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import numpy as np


model_names = [
    # "JiaqiLee/robust-bert-yelp",
    "textattack/bert-base-uncased-yelp-polarity",
    "VictorSanh/roberta-base-finetuned-yelp-polarity",
    "textattack/bert-base-uncased-imdb",
    "textattack/roberta-base-imdb",
]

model_dict = {
    "JiaqiLee/robust-bert-yelp": "Robust BERT Yelp",
    "textattack/bert-base-uncased-yelp-polarity": "BERT Yelp",
    "VictorSanh/roberta-base-finetuned-yelp-polarity": "RoBERTa Yelp",
    "textattack/bert-base-uncased-imdb": "BERT IMDB",
    "textattack/roberta-base-imdb": "RoBERTa IMDB",
}

explainer_dict = {
    "occlusion_1": "Occlusion@1",
    "lime": "LIME",
    "kernelshap": "SHAP",
    "integrated_gradient": "IG",
    "random_baseline": "Random",
    "gradient_x_input": "GxI",
    "attention": "Attention",
    "attingrad": "AttInGrad",
    "decompx": "DecompX",
    "deeplift": "DeepLIFT",
}
explainers_to_include = ["DecompX", "LIME", "IG"]
datafram_list = []
for model_name in model_names:
    results_df = pd.read_csv(
        f"results/yelp_polarity_results_{model_name.split('/')[1]}.csv"
    )
    results_df["model"] = model_dict[model_name]
    datafram_list.append(results_df)
df = pd.concat(datafram_list)
df = df[df["prob"]>0.5]
df["model_norm_comprehensiveness"] = df["comprehensiveness"]/df["prob"]
df["explanation_method"] = df["explanation_method"].apply(lambda x: explainer_dict[x])
df = df[df["explanation_method"].isin(explainers_to_include)]
results = df.groupby([ "explanation_method", "model"]).agg(
    comprehensiveness_mean=('comprehensiveness', 'mean'),
    normalized_comprehensiveness_mean=('normalized_comprehensiveness', 'mean'),
    model_norm_comprehensiveness_mean=('model_norm_comprehensiveness', 'mean'),
    sufficiency_mean=('sufficiency', 'mean'),
    normalized_sufficiency_mean=('normalized_sufficiency', 'mean')
).reset_index()
results["Combination"] = results["model"]+ " + "+ results["explanation_method"] 
results = results.drop(columns=["model", "explanation_method"])
results["Original"] = results["comprehensiveness_mean"].rank(ascending=False)
results["Model norm"] = results["model_norm_comprehensiveness_mean"].rank(ascending=False)
results["Exact algorithm"] = results["normalized_comprehensiveness_mean"].rank(ascending=False)
results["Approximation algorithm"] = results["normalized_comprehensiveness_mean"].rank(ascending=False)
results[["Combination", "Original", "comprehensiveness_mean"]]
results[["Combination", "Exact algorithm", "normalized_comprehensiveness_mean"]]
results = results[["Combination", "Original", "Exact algorithm", "Approximation algorithm"]]

# Plot settings
plt.rcParams['ytick.major.pad']='-20'
fig, ax = plt.subplots(figsize=(14, 10))

# Define the positions and colors of the columns
positions = ['Original', 'Exact algorithm', 'Approximation algorithm']
colors = ['#E24A33', '#348ABD', '#988ED5']

# Plot each country's line
for i in range(results.shape[0]):
    ax.plot(positions, results.iloc[i, 1:], marker='o', linestyle='-', color='grey', alpha=0.5)

# Highlight specific countries (optional)
highlight_combinations = ['BERT IMDB + DecompX', 'RoBERTa IMDB + DecompX', 'RoBERTa Yelp + DecompX', 'RoBERTa IMDB + LIME', 'BERT Yelp + LIME']
highlight_colors = ['#FF0000', '#00FF00', '#0000FF', '#FFA500', '#800080']
for combination, color in zip(highlight_combinations, highlight_colors):
    combination_data = results[results['Combination'] == combination]
    ax.plot(positions, combination_data.iloc[0, 1:], marker='o', linestyle='-', color=color, linewidth=2)

# Customize the plot appearance
ax.set_xticks(positions)
ax.invert_yaxis()
# ax.grid(True, linestyle='--', alpha=0.6)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)



ax.set_yticks([1,5,10])
ax.set_yticklabels(["1st","5th","10th"], fontsize=18, fontweight='semibold')
ax.tick_params(left = False)
ax.tick_params(top=False, labeltop=True, bottom=False, labelbottom=False)
ax.set_xticklabels(['Original', 'Exact algorithm', 'Approximation algorithm'], fontsize=22, fontweight='bold')

# # Custom legend
# legend_elements = [Patch(facecolor='grey', edgecolor='grey', label='Other countries')]
# for combination, color in zip(highlight_combinations, highlight_colors):
#     legend_elements.append(Patch(facecolor=color, edgecolor=color, label=combination))

# ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

# Annotate specific points (optional)
for i in range(results.shape[0]):
    if results.iloc[i, 0] in highlight_combinations:
        continue
    rank = results.iloc[i, 3]+0.1
    pos = 2.02
    ax.text(pos, rank, f'{results.iloc[i, 0]}', horizontalalignment='left', size=12, color='grey', weight='semibold', )


# first column
for combination in highlight_combinations:
    combination_data = results[results['Combination'] == combination]
    pos = 0.01
    rank = combination_data.iloc[0, 1]
    # ax.text(pos+0.01, rank+0.05, f'{combination}', horizontalalignment='left', size='small', color="black", weight='bold')

# second column
for combination in highlight_combinations:
    combination_data = results[results['Combination'] == combination]
    pos = 1.01
    rank = combination_data.iloc[0, 2]
    # ax.text(pos+0.01, rank+0.05, f'{combination}', horizontalalignment='left', size='small', color="black", weight='bold')

# third column
for combination in highlight_combinations:
    combination_data = results[results['Combination'] == combination]
    pos = 2.02
    rank = combination_data.iloc[0, 3]
    ax.text(pos, rank+0.1, f'{combination}', horizontalalignment='left', size=12, color="black", weight='bold')


# Display the plot
plt.tight_layout()
plt.show()

plt.savefig('results/ranking_plot.png')