import pandas as pd
import matplotlib.pyplot as plt


model_names = [
    "textattack/bert-base-uncased-SST-2",
    "textattack/roberta-base-SST-2",
    "textattack/bert-base-uncased-yelp-polarity",
    "VictorSanh/roberta-base-finetuned-yelp-polarity",
    "textattack/bert-base-uncased-imdb",
    "textattack/roberta-base-imdb",
]
datasets = ["yelp", "sst2", "imdb"]

beam_sizes = [1, 2, 5, 10, 15, 50]

file_template = "results/aopc_limits_approx_increasing_beams/{}_long_no_preprocessing_beam_size_{}_{}.parquet"

for dataset in datasets:

    for i, model in enumerate(model_names):
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        frames = []
        for beam_size in beam_sizes:
            file = file_template.format(dataset, beam_size, model.split("/")[1])
            df = pd.read_parquet(file)
            df["beam_size"] = beam_size
            frames.append(df)

        comp_data = [frame["comprehensiveness"] for frame in frames]
        suff_data = [frame["sufficiency"] for frame in frames]
        ax[0].boxplot(comp_data, showmeans=True, meanline=True, tick_labels=beam_sizes, showfliers=False)
        ax[0].set_title(f"{model.split("/")[-1]}")
        ax[1].boxplot(suff_data, showmeans=True, meanline=True, tick_labels=beam_sizes, showfliers=False)
        ax[1].set_title(f"{model.split("/")[-1]}")

        for j, d in enumerate(comp_data):
            ax[0].text(j + 1, d.mean(), f"{d.mean():.4f}", ha='center', va='bottom', color='black')
        for j, d in enumerate(suff_data):
            ax[1].text(j + 1, d.mean(), f"{d.mean():.4f}", ha='center', va='bottom', color='black')
    
        fig.suptitle(f"Comprehensiveness and Sufficiency by Beam Size for {dataset.title()}")
            
        plt.savefig(f"figures/{dataset}_{model.split("/")[-1]}_increasing_beam_sizes.png")
        plt.close()
        plt.clf()