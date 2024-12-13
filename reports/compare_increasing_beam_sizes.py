import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2, palette="colorblind")

model_names = [
    "textattack/bert-base-uncased-SST-2",
    "textattack/roberta-base-SST-2",
    "textattack/bert-base-uncased-yelp-polarity",
    "VictorSanh/roberta-base-finetuned-yelp-polarity",
    "textattack/bert-base-uncased-imdb",
    "textattack/roberta-base-imdb",
]
datasets = ["yelp", "sst2", "imdb"]
model_map = {
    "bert-base-uncased-SST-2": "BERT$_{\\text{IMDB}}$",
    "roberta-base-SST-2": "RoBERTa$_{\\text{IMDB}}$",
    "bert-base-uncased-yelp-polarity": "BERT$_{\\text{Yelp}}$",
    "roberta-base-finetuned-yelp-polarity": "RoBERTa$_{\\text{Yelp}}$",
    "bert-base-uncased-imdb": "BERT$_{\\text{SST2}}$",
    "roberta-base-imdb": "RoBERTa$_{\\text{SST2}}$",
}
dataset_map = {"yelp": "Yelp", "sst2": "SST-2", "imdb": "IMDB"}

beam_sizes = [1, 2, 5, 10, 15, 50]

file_template = "results/aopc_limits_approx_increasing_beams/{}_long_no_preprocessing_beam_size_{}_{}.parquet"

fig_big, ax_big = plt.subplots(
    6, 3, figsize=(12, 20), sharex="all", sharey="all", layout="constrained"
)

for column_idx, dataset in enumerate(datasets):
    for row_idx, model in enumerate(model_names):
        frames = []
        for beam_size in beam_sizes:
            file = file_template.format(dataset, beam_size, model.split("/")[1])
            df = pd.read_parquet(file)
            df["beam_size"] = beam_size
            frames.append(df)
        
        df = pd.concat(frames)
        # make metric a column for seaborn
        df = pd.melt(df, id_vars=["beam_size"], value_vars=["comprehensiveness", "sufficiency"], var_name="metric", value_name="aopc")

        # big plot
        sns.boxplot(ax=ax_big[row_idx, column_idx], data=df, y="aopc", x="beam_size", hue="metric", palette="colorblind", showfliers=False)
        # no legend nor labels
        ax_big[row_idx, column_idx].get_legend().remove()
        ax_big[row_idx, column_idx].set_xlabel("")
        ax_big[row_idx, column_idx].set_ylabel("")

        ax_big[row_idx, column_idx].set_ylim(-0.1, 1.0)
        ax_big[row_idx, column_idx].tick_params(axis="both", which="both", labelsize=12)

        if row_idx == 0:
            dataset_name = dataset_map[dataset]
            ax_big[row_idx, column_idx].set_title(
                dataset_name, fontsize=18, fontweight="bold"
            )

        if row_idx == 5:
            ax_big[row_idx, column_idx].set_xlabel("Beam Size", fontsize=14)

        if column_idx == 0:
            model_name = model_map[model.split("/")[1]]
            if "roberta" in model_name.lower():
                y_pos = 0.2
            else:
                y_pos = 0.3

            ax_big[row_idx, column_idx].text(
                -1.7,
                y_pos,
                model_name,
                fontsize=18,
                rotation=90,
                rotation_mode="anchor",
                fontweight="bold",
            )
            ax_big[row_idx, column_idx].set_ylabel("AOPC", fontsize=14)
            # move the y label to the right
            ax_big[row_idx, column_idx].yaxis.set_label_coords(-0.1, 0.5)
            # .annotate(model_name, (-0.65, 0.5), xycoords = 'axes fraction', rotation = 90, va = 'center', fontweight = 'bold', fontsize = 18)

        # individual boxplot
        fig, ax = plt.subplots(figsize=(4, 4))

        sns.boxplot(ax=ax, data=df, y="aopc", x="beam_size", hue="metric", palette="colorblind", showfliers=False)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, ["Upper AOPC limit", "Lower AOPC limit"], bbox_to_anchor=(0.2, 0.25), fontsize=12, frameon=False,)

        plt.ylabel("AOPC")
        plt.xlabel("Beam Size")
        plt.ylim(-0.1, 1.0)
        fig.tight_layout()
        fig.savefig(
            f"figures/boxplots/{dataset}_{model.split('/')[-1]}_increasing_beam_sizes.pdf",
            format="pdf",
        )
# move the box sligly down
leg = fig_big.legend(
    handles=ax_big[0, 0].get_legend_handles_labels()[0],
    labels=["Upper AOPC limit", "Lower AOPC limit"],
    bbox_to_anchor=(0.8, 0.59),
    ncol=2,
    fontsize=18,
    frameon=False,
)

fig_big.savefig("figures/boxplots/all_increasing_beam_sizes.pdf", format="pdf")