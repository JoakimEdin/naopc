import matplotlib.pyplot as plt
import pandas as pd

datasets = ["yelp", "sst2", "imdb"]
length = "long"

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
# Plot settings
plt.rcParams["ytick.major.pad"] = "2"
fig, ax = plt.subplots(2, 3, figsize=(20, 18), sharex="all", sharey="all")

for col_idx, dataset_name in enumerate(datasets):
    for model_name in model_names:
        results_df = pd.read_csv(
            f"results/aopc_scores_{length}/{dataset_name}_{model_name.split('/')[1]}.csv"
        )
        aopc_approx_limits = pd.read_parquet(
            f"results/aopc_limits_approx/{dataset_name}_{length}_no_preprocessing_beam_size_5_{model_name.split('/')[1]}.parquet"
        )
        results_df = results_df.merge(
            aopc_approx_limits,
            on="id",
            suffixes=("", "_approx"),
        )

        results_df["model"] = model_dict[model_name]
        dataframe_list.append(results_df)
    df = pd.concat(dataframe_list)

    df["approx_normalized_comprehensiveness"] = (
        df["comprehensiveness"] - df["sufficiency_approx"]
    ) / (df["comprehensiveness_approx"] - df["sufficiency_approx"])
    df["approx_normalized_sufficiency"] = (
        df["sufficiency"] - df["sufficiency_approx"]
    ) / (df["comprehensiveness_approx"] - df["sufficiency_approx"])

    df.loc[
        (df["comprehensiveness_approx"] - df["sufficiency_approx"]) == 0,
        "approx_normalized_comprehensiveness",
    ] = 0
    df.loc[
        (df["comprehensiveness_approx"] - df["sufficiency_approx"]) == 0,
        "approx_normalized_sufficiency",
    ] = 0

    df = df[df["prob"] > 0.5]
    df["model_norm_comprehensiveness"] = df["comprehensiveness"] / df["prob"]
    df["explanation_method"] = df["explanation_method"].apply(
        lambda x: explainer_dict[x]
    )
    df = df[df["explanation_method"].isin(explainers_to_include)]
    for row_idx, metric in enumerate(["comprehensiveness", "sufficiency"]):
        if metric == "comprehensiveness":
            ascending = False
        else:
            ascending = True

        results = (
            df.groupby(["explanation_method", "model"])
            .agg(
                comprehensiveness_mean=("comprehensiveness", "mean"),
                approximation_comprehensiveness_mean=(
                    "approx_normalized_comprehensiveness",
                    "mean",
                ),
                model_norm_comprehensiveness_mean=(
                    "model_norm_comprehensiveness",
                    "mean",
                ),
                sufficiency_mean=("sufficiency", "mean"),
                approximation_sufficiency_mean=(
                    "approx_normalized_sufficiency",
                    "mean",
                ),
            )
            .reset_index()
        )

        results["Combination"] = (
            results["model"] + " + " + results["explanation_method"]
        )
        results["Original"] = results[f"{metric}_mean"].rank(ascending=ascending)
        results["Approximation algorithm"] = results[
            f"approximation_{metric}_mean"
        ].rank(ascending=ascending)
        results = results[
            [
                "explanation_method",
                "Original",
                "Approximation algorithm",
                f"{metric}_mean",
                f"approximation_{metric}_mean",
                "model",
            ]
        ]

        # Define the positions and colors of the columns
        positions = ["Original", "Approximation algorithm"]

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
                    ax[row_idx, col_idx].plot(
                        positions,
                        model_data.iloc[i, 1:3],
                        marker="o",
                        linestyle="-",
                        color=color,
                        linewidth=4,
                        label=model,
                    )
                else:
                    ax[row_idx, col_idx].plot(
                        positions,
                        model_data.iloc[i, 1:3],
                        marker="o",
                        linestyle="-",
                        color=color,
                        linewidth=3,
                    )

        # Customize the plot appearance
        ax[row_idx, col_idx].set_xticks(positions)

        # ax[row_idx, col_idx].grid(True, linestyle='--', alpha=0.6)
        ax[row_idx, col_idx].spines["top"].set_visible(False)
        ax[row_idx, col_idx].spines["right"].set_visible(False)
        ax[row_idx, col_idx].spines["bottom"].set_visible(False)
        ax[row_idx, col_idx].spines["left"].set_visible(False)

        ax[row_idx, col_idx].set_yticks([1, 5, 10, 15, 20])
        ax[row_idx, col_idx].set_yticklabels(
            ["1st", "5th", "10th", "15th", "20th"], fontsize=16, fontweight="semibold"
        )
        ax[row_idx, col_idx].set_xticklabels(
            ["AOPC", "NAOPC$_{\\text{beam}}$"],
            fontsize=18,
            fontweight="bold",
        )

        # ax[row_idx, col_idx].tick_params(
        #     top=False, labeltop=True, bottom=False, labelbottom=False, left=False
        # )

        # do not show xtix for bottom row

        # Set the title
        if row_idx == 0:
            dataset_name = (
                dataset_name.replace("imdb", "IMDB$_{\\text{long}}$")
                .replace("sst2", "SST-2$_{\\text{long}}$")
                .replace("yelp", "Yelp$_{\\text{long}}$")
            )
            ax[row_idx, col_idx].set_title(
                f"{dataset_name}",
                fontsize=25,
                fontweight="bold",
            )
            # ax[row_idx, col_idx].tick_params(labelbottom=True)

        # # Custom legend
        # legend_elements = [Patch(facecolor='grey', edgecolor='grey', label='Other countries')]
        # for combination, color in zip(highlight_combinations, highlight_colors):
        #     legend_elements.append(Patch(facecolor=color, edgecolor=color, label=combination))

        # ax[row_idx, col_idx].legend(handles=legend_elements, loc='upper right', fontsize=12)

        # Annotate with name specific points (optional)
        for i in range(results.shape[0]):
            rank = results.iloc[i, 2] + 0.1
            pos = 1.08
            ax[row_idx, col_idx].text(
                pos,
                rank,
                f"{results.iloc[i, 0]}",
                horizontalalignment="left",
                size=16,
                color="black",
            )

        for i in range(results.shape[0]):
            for column in [0, 1]:
                rank = results.iloc[i, column + 1] - 0.2
                pos = column
                score = results.iloc[i, column + 3]
                ax[row_idx, col_idx].text(
                    pos,
                    rank,
                    f"{score:.2f}",
                    horizontalalignment="center",
                    size=16,
                    color="black",
                )

        # legends

        # Display the plot

ax[0, 0].set_ylabel("Comprehensiveness", fontsize=25, fontweight="bold", labelpad=10)
ax[1, 0].set_ylabel("Sufficiency", fontsize=25, fontweight="bold", labelpad=10)
plt.gca().invert_yaxis()

# Custom legend
handles, labels = ax[0, 0].get_legend_handles_labels()
leg = fig.legend(
    handles,
    labels,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.04),
    ncol=6,
    fontsize=22,
    frameon=False,
)
for line in leg.get_lines():
    line.set_linewidth(20.0)

plt.tight_layout()
plt.show()
plt.savefig("figures/ranking_plot_long.pdf", bbox_inches="tight", format="pdf")
