from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

path_to_images = Path(__file__).parent.parent / "figures" / "violinplots"
path_to_images.mkdir(parents=True, exist_ok=True)

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

        comp_data = [frame["comprehensiveness"] for frame in frames]
        suff_data = [frame["sufficiency"] for frame in frames]

        # big plot
        parts1 = ax_big[row_idx, column_idx].violinplot(
            comp_data, positions=range(len(beam_sizes)), showmeans=True
        )
        for pc in parts1["bodies"]:
            pc.set_facecolor("#377eb8")
            pc.set_edgecolor("black")
            pc.set_alpha(0.7)

        parts2 = ax_big[row_idx, column_idx].violinplot(
            suff_data, positions=range(len(beam_sizes)), showmeans=True
        )
        for pc in parts2["bodies"]:
            pc.set_facecolor("#4daf4a")
            pc.set_edgecolor("black")
            pc.set_alpha(0.7)

        ax_big[row_idx, column_idx].set_xticks(range(len(beam_sizes)))
        ax_big[row_idx, column_idx].set_xticklabels(beam_sizes)
        ax_big[row_idx, column_idx].set_ylim(-0.1, 1.05)
        ax_big[row_idx, column_idx].tick_params(axis="both", which="both", labelsize=12)
        ax_big[row_idx, column_idx].grid(axis="y", which="both")

        if row_idx == 0:
            dataset_name = dataset_map[dataset]
            ax_big[row_idx, column_idx].set_title(
                dataset_name, fontsize=18, fontweight="bold"
            )

        if row_idx == 5:
            ax_big[row_idx, column_idx].set_xlabel("Beam Size", fontsize=14)

        if column_idx == 0:
            model_name = model_map[model.split("/")[1]]
            y_pos = 0.3 if "bert" in model_name.lower() else 0.15

            ax_big[row_idx, column_idx].text(
                -0.8,
                y_pos,
                model_name,
                fontsize=18,
                rotation=90,
                rotation_mode="anchor",
                fontweight="bold",
            )
            ax_big[row_idx, column_idx].set_ylabel("AOPC", fontsize=14)

        # individual violin plot
        fig, ax = plt.subplots(figsize=(3, 3))
        v1 = ax.violinplot(comp_data, positions=range(len(beam_sizes)), showmeans=True)
        v2 = ax.violinplot(suff_data, positions=range(len(beam_sizes)), showmeans=True)
        for v in v1["bodies"]:
            v.set_facecolor("#377eb8")
            v.set_edgecolor("black")
            v.set_alpha(0.7)
        for v in v2["bodies"]:
            v.set_facecolor("#ff7f00")
            v.set_edgecolor("black")
            v.set_alpha(0.7)

        ax.set_xticks(range(len(beam_sizes)))
        ax.set_xticklabels(beam_sizes)
        ax.legend(
            ["Upper limit", "Lower limit"],
            loc="center right",
            bbox_to_anchor=(0.8, 0.3),
        )
        plt.ylabel("AOPC")
        plt.xlabel("Beam Size")
        plt.ylim(-0.1, 1.05)
        plt.grid(axis="y", which="both")
        fig.tight_layout()
        fig.savefig(
            path_to_images
            / f"violin_{dataset}_{model.split('/')[-1]}_increasing_beam_sizes.pdf",
            format="pdf",
        )

leg = fig_big.legend(
    ["Upper AOPC limit", "Lower AOPC limit"],
    bbox_to_anchor=(0.8, 0.6),
    ncol=2,
    fontsize=18,
    frameon=False,
)

fig_big.savefig(path_to_images / "violin_all_increasing_beam_sizes.pdf", format="pdf")
