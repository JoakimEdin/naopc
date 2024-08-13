from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dataset_names = ["yelp", "sst2"]
model_names = [
    # "textattack/bert-base-uncased-SST-2",
    # "textattack/roberta-base-SST-2",
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
for dataset_name in dataset_names:
    dataframes = []
    for model_name in model_names:
        file_path = Path(
            f"results/aopc_limits_exact/{dataset_name}_{model_name.split('/')[1]}.parquet"
        )
        if not file_path.exists():
            continue
        temp_df = pd.read_parquet(file_path)
        temp_df["model"] = model_dict[model_name]
        dataframes.append(temp_df)

    df = pd.concat(dataframes)
    # df = df[df["prob"] > 0.5]
    sns.set_theme(
        style="whitegrid", context="paper", palette="colorblind", font_scale=1.5
    )
    g = sns.histplot(data=df, x="upper_limit", hue="model", kde=True, bins=30)
    plt.xlabel("AOPC score")
    plt.ylabel("Frequency")
    # no legend title for the model
    plt.legend(title=None)
    # plot vertical line representing the average
    print(df.groupby("model")["upper_limit"].describe())
    print(df.groupby("model")["lower_limit"].describe())
    plt.tight_layout()
    plt.xlim(-0.1, 1.1)

    plt.savefig(f"figures/{dataset_name}_upper_limit.pdf", format="pdf")

    plt.clf()

    g = sns.histplot(data=df, x="lower_limit", hue="model", kde=True, bins=70)
    plt.xlabel("AOPC score")
    plt.ylabel("Frequency")
    plt.xlim(-0.2, 0.5)
    plt.tight_layout()
    g.legend_.set_title(None)

    plt.savefig(f"figures/{dataset_name}_lower_limit.pdf", format="pdf")
