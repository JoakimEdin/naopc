import matplotlib.pyplot as plt
import pandas as pd

dataset_name = "yelp"
length = "short"

model_names = [
    "textattack/bert-base-uncased-SST-2",
    "textattack/roberta-base-SST-2",
    "textattack/bert-base-uncased-yelp-polarity",
    "VictorSanh/roberta-base-finetuned-yelp-polarity",
    "textattack/bert-base-uncased-imdb",
    "textattack/roberta-base-imdb",
]

model_dict = {
    "textattack/roberta-base-SST-2": "RoBERTa$_{\\text{SST2}}$",
    "textattack/bert-base-uncased-SST-2": "BERT$_{\\text{SST2}}$",
    "textattack/bert-base-uncased-yelp-polarity": "BERT$_{\\text{Yelp}}$",
    "VictorSanh/roberta-base-finetuned-yelp-polarity": "RoBERTa$_{\\text{Yelp}}$",
    "textattack/bert-base-uncased-imdb": "BERT$_{\\text{IMDB}}$",
    "textattack/roberta-base-imdb": "RoBERTa$_{\\text{IMDB}}$",
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
explainers_to_include = ["DecompX", "LIME", "IG", "Attention"]
dataframe_list = []
for model_name in model_names:
    results_df = pd.read_csv(
        f"results/aopc_scores_{length}/{dataset_name}_{model_name.split('/')[1]}.csv"
    )
    aopc_limits = pd.read_parquet(
        f"results/aopc_limits_exact/{dataset_name}_{model_name.split('/')[1]}.parquet"
    )
    results_df = results_df.merge(
        aopc_limits[["id", "upper_limit", "lower_limit"]], on="id"
    )
    results_df["model"] = model_dict[model_name]
    dataframe_list.append(results_df)
df = pd.concat(dataframe_list)

df["normalized_comprehensiveness"] = (df["comprehensiveness"] - df["lower_limit"]) / (
    df["upper_limit"] - df["lower_limit"]
)
df["normalized_sufficiency"] = (df["sufficiency"] - df["lower_limit"]) / (
    df["upper_limit"] - df["lower_limit"]
)

df.loc[(df["upper_limit"] - df["lower_limit"]) == 0, "normalized_comprehensiveness"] = 0
df.loc[(df["upper_limit"] - df["lower_limit"]) == 0, "normalized_sufficiency"] = 0

df = df[df["prob"] > 0.5]
df["model_norm_comprehensiveness"] = df["comprehensiveness"] / df["prob"]
df["explanation_method"] = df["explanation_method"].apply(lambda x: explainer_dict[x])
df = df[df["explanation_method"].isin(explainers_to_include)]
for metric in ["comprehensiveness", "sufficiency"]:
    if metric == "comprehensiveness":
        ascending = False
    else:
        ascending = True

    results = (
        df.groupby(["explanation_method", "model"])
        .agg(
            comprehensiveness_mean=("comprehensiveness", "mean"),
            normalized_comprehensiveness_mean=("normalized_comprehensiveness", "mean"),
            model_norm_comprehensiveness_mean=("model_norm_comprehensiveness", "mean"),
            sufficiency_mean=("sufficiency", "mean"),
            normalized_sufficiency_mean=("normalized_sufficiency", "mean"),
        )
        .reset_index()
    )
    # delete when we have approximation results
    results[f"approximation_{metric}_mean"] = results[f"normalized_{metric}_mean"]

    results["Combination"] = results["model"] + " + " + results["explanation_method"]
    results["Original"] = results[f"{metric}_mean"].rank(ascending=ascending)
    results["Exact algorithm"] = results[f"normalized_{metric}_mean"].rank(
        ascending=ascending
    )
    results["Approximation algorithm"] = results[f"approximation_{metric}_mean"].rank(
        ascending=ascending
    )
    results = results[
        [
            "explanation_method",
            "Original",
            "Exact algorithm",
            "Approximation algorithm",
            f"{metric}_mean",
            f"normalized_{metric}_mean",
            f"approximation_{metric}_mean",
            "model",
        ]
    ]

    # Plot settings
    plt.rcParams["ytick.major.pad"] = "0"
    fig, ax = plt.subplots(figsize=(10, 10))

    # Define the positions and colors of the columns
    positions = ["Original", "Exact algorithm", "Approximation algorithm"]

    # Highlight specific countries (optional)
    models = [
        "BERT$_{\\text{IMDB}}$",
        "RoBERTa$_{\\text{IMDB}}$",
        "BERT$_{\\text{Yelp}}$",
        "RoBERTa$_{\\text{Yelp}}$",
        "BERT$_{\\text{SST2}}$",
        "RoBERTa$_{\\text{SST2}}$",
    ]
    model_colors = [
        "#377eb8",
        "#ff7f00",
        "#4daf4a",
        "#f781bf",
        "#a65628",
        # "#984ea3",
        # "#999999",
        "#e41a1c",
        # "#dede00",
    ]

    for model, color in zip(models, model_colors):
        model_data = results[results["model"] == model]
        for i in range(model_data.shape[0]):
            if i == 0:
                ax.plot(
                    positions,
                    model_data.iloc[i, 1:4],
                    marker="o",
                    linestyle="-",
                    color=color,
                    linewidth=3,
                    label=model,
                )
            else:
                ax.plot(
                    positions,
                    model_data.iloc[i, 1:4],
                    marker="o",
                    linestyle="-",
                    color=color,
                    linewidth=3,
                )

    # Customize the plot appearance
    ax.set_xticks(positions)
    ax.invert_yaxis()
    # ax.grid(True, linestyle='--', alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.set_yticks([1, 5, 10, 15, 20])
    ax.set_yticklabels(
        ["1st", "5th", "10th", "15th", "20th"], fontsize=16, fontweight="semibold"
    )
    ax.tick_params(left=False)
    ax.tick_params(top=False, labeltop=True, bottom=False, labelbottom=False)
    ax.set_xticklabels(
        ["No normalization", "Exaustive search", "Beam search"],
        fontsize=18,
        fontweight="bold",
    )

    # # Custom legend
    # legend_elements = [Patch(facecolor='grey', edgecolor='grey', label='Other countries')]
    # for combination, color in zip(highlight_combinations, highlight_colors):
    #     legend_elements.append(Patch(facecolor=color, edgecolor=color, label=combination))

    # ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

    # Annotate with name specific points (optional)
    for i in range(results.shape[0]):
        rank = results.iloc[i, 3] + 0.1
        pos = 2.1
        ax.text(
            pos,
            rank,
            f"{results.iloc[i, 0]}",
            horizontalalignment="left",
            size=16,
            color="black",
        )

    for i in range(results.shape[0]):
        for column in [0, 1, 2]:
            rank = results.iloc[i, column + 1] - 0.2
            pos = column
            score = results.iloc[i, column + 4]
            ax.text(
                pos,
                rank,
                f"{score:.2f}",
                horizontalalignment="center",
                size=16,
                color="black",
            )

    # legends
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        ncol=3,
        fontsize=20,
        frameon=False,
    )

    # Display the plot
    plt.tight_layout()
    plt.show()

    plt.savefig(f"figures/ranking_plot_{metric}.pdf", bbox_inches="tight", format="pdf")
    plt.clf()
