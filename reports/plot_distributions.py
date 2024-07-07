from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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

dataframes = []
for model_name in model_names:
    file_path = Path(f"results/yelp_best_scores_{model_name.split('/')[1]}.parquet")
    if not file_path.exists():
        continue
    temp_df = pd.read_parquet(file_path)
    temp_df["model"] = model_dict[model_name]
    dataframes.append(temp_df)


df = pd.concat(dataframes)
sns.set_theme(style="whitegrid", context="paper", palette="colorblind", font_scale=1.5)
g = sns.histplot(data=df, x="comprehensiveness", hue="model", kde=True, bins=20)
plt.xlabel("AOPC score")
plt.ylabel("Frequency")
# no legend title for the model
plt.legend(title=None)
# plot vertical line representing the average
print(df.groupby("model")["comprehensiveness"].describe())
print(df.groupby("model")["sufficiency"].describe())
plt.tight_layout()
plt.xlim(-0.1, 1.1)

plt.savefig("results/comprehensiveness.pdf", format="pdf")

plt.clf()


g=sns.histplot(data=df, x="sufficiency", hue="model", kde=True, bins=50)
plt.xlabel("AOPC score")
plt.ylabel("Frequency")
plt.xlim(-0.2, 0.5)
plt.tight_layout()
g.legend_.set_title(None)

plt.savefig("results/sufficiency.pdf", format="pdf")
