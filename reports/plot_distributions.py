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
sns.set_theme(style="whitegrid", context="paper", palette="colorblind", font_scale=1)
sns.histplot(data=df, x="comprehensiveness", hue="model", kde=True, bins=30)
plt.xlabel("Comprehensiveness")
plt.ylabel("Frequency")
# plot vertical line representing the average
print(df.groupby("model")["comprehensiveness"].describe())
print(df.groupby("model")["sufficiency"].describe())
plt.xlim(-0.1, 1.1)

plt.savefig("results/comprehensiveness.png")

plt.clf()


sns.histplot(data=df, x="sufficiency", hue="model", kde=True, bins=50)
plt.xlabel("Sufficiency")
plt.ylabel("Frequency")
plt.title("Sufficiency")
plt.xlim(-0.2, 0.5)
plt.savefig("results/sufficiency.png")
