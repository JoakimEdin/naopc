import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2, palette="colorblind")
model_names = [
    "textattack/distilbert-base-uncased-ag-news",
    "textattack/bert-base-uncased-ag-news",
    "textattack/roberta-base-ag-news",
]
dataset = "ag_news"
model_map = {
    "distilbert-base-uncased-ag-news": "DistillBERT$_{\\text{AG News}}$",
    "bert-base-uncased-ag-news": "BERT$_{\\text{AG News}}$",
    "roberta-base-ag-news": "RoBERTa$_{\\text{AG News}}$",
}
dataset_map = {"ag_news": "AG News"}

beam_sizes = [1, 5, 10, 50, 100, 200, 500, 1000, 2500, 5000, 10000]

file_template = "results/aopc_limits_approx_increasing_beams/{}_long_no_preprocessing_beam_size_{}_{}.parquet"
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2, palette="colorblind")
for row_idx, model in enumerate(model_names):
    frames = []
    for beam_size in beam_sizes:
        file = file_template.format(dataset, beam_size, model.split("/")[1])
        df = pd.read_parquet(file)
        df["beam_size"] = beam_size
        frames.append(df)

    df = pd.concat(frames)
    # make metric a column for seaborn
    df = pd.melt(
        df,
        id_vars=["beam_size"],
        value_vars=["comprehensiveness", "sufficiency"],
        var_name="metric",
        value_name="aopc",
    )

    # individual boxplot
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(
        ax=ax,
        data=df,
        y="aopc",
        x="beam_size",
        hue="metric",
        palette="colorblind",
        showfliers=False,
    )

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        ["Upper AOPC limit", "Lower AOPC limit"],
        fontsize=12,
        frameon=False,
        bbox_to_anchor=(0.64, 0.2),
    )

    plt.ylabel("AOPC")
    plt.xlabel("Beam Size")
    plt.ylim(-0.1, 1.0)
    fig.tight_layout()
    fig.savefig(
        f"figures/boxplots/{dataset}_{model.split('/')[-1]}_increasing_beam_sizes.pdf",
        format="pdf",
    )
