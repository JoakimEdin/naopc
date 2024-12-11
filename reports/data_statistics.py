import pandas as pd
from datasets import load_dataset
from matplotlib import pyplot as plt



imdb = load_dataset("csv", data_files="data/imdb_test_long.csv", split="train")
sst2 = load_dataset("csv", data_files="data/sst2_test_long.csv", split="train")
yelp = load_dataset("csv", data_files="data/yelp_test_long.csv", split="train")
snli = load_dataset("csv", data_files="data/snli_test_long.csv", split="train")
agnews = load_dataset("csv", data_files="data/ag_news_test_long.csv", split="train")

snli = snli.map(lambda x: {"text": x["premise"] + " " + x["hypothesis"]})

for i, (dataset, name) in enumerate(zip([sst2, yelp, imdb, snli, agnews], ["SST-2", "Yelp", "IMDB", "SNLI", "AG News"])):
    lengths = dataset.with_format("pandas")["text"].str.split().apply(len)
    print(name, lengths.describe())


# plot length histograms

fig, axs = plt.subplots(5, 1, figsize=(10, 25))
for i, (dataset, name) in enumerate(zip([sst2, yelp, imdb, snli, agnews], ["SST-2", "Yelp", "IMDB", "SNLI", "AG News"])):
    lengths = dataset.with_format("pandas")["text"].str.split().apply(len)
    axs[i].hist(lengths, bins=50)
    axs[i].set_title(f"{name} length distribution")
    axs[i].set_xlabel("Number of tokens")
    axs[i].set_ylabel("Frequency")
plt.tight_layout()
plt.savefig("figures/length_histograms.png", format="png")

sst2 = load_dataset("csv", data_files="data/sst2_test_short.csv", split="train")
yelp = load_dataset("csv", data_files="data/yelp_test_short.csv", split="train")
for i, (dataset, name) in enumerate(zip([sst2, yelp], ["SST-2", "Yelp"])):
    lengths = dataset.with_format("pandas")["text"].str.split().apply(len)
    print(name, lengths.describe())






