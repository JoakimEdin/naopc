import pandas as pd
from datasets import load_dataset
from matplotlib import pyplot as plt



imdb = load_dataset("csv", data_files="data/imdb_test.csv", split="train")
sst2 = load_dataset("csv", data_files="data/sst2_test.csv", split="train")
yelp = load_dataset("csv", data_files="data/yelp_test.csv", split="train")






# plot length histograms

fig, axs = plt.subplots(3, 1, figsize=(10, 15))
for i, (dataset, name) in enumerate(zip([sst2, imdb, yelp], ["SST-2", "IMDB", "Yelp"])):
    lengths = dataset.with_format("pandas")["text"].str.split().apply(len)
    axs[i].hist(lengths, bins=50)
    axs[i].set_title(f"{name} length distribution")
    axs[i].set_xlabel("Number of tokens")
    axs[i].set_ylabel("Frequency")
plt.tight_layout()
plt.savefig("results/length_histograms.png", format="png")

sst2 = load_dataset("csv", data_files="data/sst2_test_short.csv", split="train")
yelp = load_dataset("csv", data_files="data/yelp_test_short.csv", split="train")
for i, (dataset, name) in enumerate(zip([sst2, yelp], ["SST-2", "Yelp"])):
    lengths = dataset.with_format("pandas")["text"].str.split().apply(len)
    print(name, lengths.describe())






