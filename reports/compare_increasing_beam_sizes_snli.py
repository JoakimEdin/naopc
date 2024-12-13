import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2, palette="colorblind")
model_names = [
    'textattack/bert-base-uncased-snli',
    'textattack/distilbert-base-cased-snli',
    'varun-v-rao/gpt2-snli-model1',
]
dataset = "snli"
model_map = {
    'bert-base-uncased-snli': 'BERT$_{\\text{SNLI}}$',
    'distilbert-base-cased-snli': 'RoBERTa$_{\\text{SNLI}}$',
    'gpt2-snli-model1': 'GPT-2$_{\\text{SNLI}}$',
}
dataset_map = {"snli": "SNLI"}

beam_sizes = [1, 2, 5, 10, 50]

file_template = "results/aopc_limits_approx_increasing_beams/{}_long_no_preprocessing_beam_size_{}_{}.parquet"

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

    # individual boxplot
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.boxplot(ax=ax, data=df, y="aopc", x="beam_size", hue="metric", showfliers=False)

    handles, labels = ax.get_legend_handles_labels()
    if model == 'textattack/bert-base-uncased-snli':
        ax.legend(handles, ["Upper AOPC limit", "Lower AOPC limit"], loc="upper center", fontsize=12, frameon=False,)
    else:
        ax.legend(handles, ["Upper AOPC limit", "Lower AOPC limit"], loc="lower center", fontsize=12, frameon=False,)

    plt.ylabel("AOPC")
    plt.xlabel("Beam Size")
    plt.ylim(-0.8, 1.0)
    fig.tight_layout()
    fig.savefig(
        f"figures/boxplots/{dataset}_{model.split('/')[-1]}_increasing_beam_sizes.pdf",
        format="pdf",
    )


