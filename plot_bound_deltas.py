import pandas as pd
import matplotlib.pyplot as plt


model_names = [
    "JiaqiLee/robust-bert-yelp",
    "textattack/bert-base-uncased-yelp-polarity",
    "VictorSanh/roberta-base-finetuned-yelp-polarity",
    "textattack/bert-base-uncased-imdb",
    "textattack/roberta-base-imdb",
]

model_name_mappings = {
    "JiaqiLee/robust-bert-yelp": "BERT Yelp",
    "textattack/bert-base-uncased-yelp-polarity": "BERT Yelp",
    "VictorSanh/roberta-base-finetuned-yelp-polarity": "RoBERTa Yelp",
    "textattack/bert-base-uncased-imdb": "BERT IMDB",
    "textattack/roberta-base-imdb": "RoBERTa IMDB",

}

model_name_to_plots_dict = {
    "BERT Yelp": {
        "comprehensiveness": [],
        "sufficiency": [],
    },
    "RoBERTa Yelp": {
        "comprehensiveness": [],
        "sufficiency": [],
    },
    "BERT IMDB": {
        "comprehensiveness": [],
        "sufficiency": [],
    },
    "RoBERTa IMDB": {
        "comprehensiveness": [],
        "sufficiency": [],
    },
}

for model_name in model_names:
    results = pd.read_csv(f"results/yelp_polarity_results_{model_name.split("/")[-1]}.csv")
    results_comprehensiveness = results[results["explanation_method"] == "comprehensiveness_solver"]
    results_sufficiency = results[results["explanation_method"] == "suffiency_solver"]
    results_comprehensiveness = results_comprehensiveness[["id", "comprehensiveness"]]
    results_sufficiency = results_sufficiency[["id", "sufficiency"]]

    lower_upper_bound_comprehensiveness_and_sufficiency = pd.read_parquet(
        f"results/yelp_best_scores_{model_name.split('/')[1]}.parquet"
    )

    comprehensiveness_deltas = []
    sufficiency_deltas = []

    # print(results)
    # print(results_comprehensiveness.shape)
    # print(results_sufficiency.shape)
    for _, row in results_comprehensiveness.iterrows():
        id = row["id"]
        lower_upper_bound_comprehensiveness_and_sufficiency_id = (
            lower_upper_bound_comprehensiveness_and_sufficiency[
                lower_upper_bound_comprehensiveness_and_sufficiency["id"] == id
            ]
        )
        max_comprehensiveness = lower_upper_bound_comprehensiveness_and_sufficiency_id[
            "comprehensiveness"
        ].values[0]

        delta = abs(row["comprehensiveness"] - max_comprehensiveness)
        comprehensiveness_deltas.append(delta)

    for _, row in results_sufficiency.iterrows():
        id = row["id"]
        lower_upper_bound_comprehensiveness_and_sufficiency_id = (
            lower_upper_bound_comprehensiveness_and_sufficiency[
                lower_upper_bound_comprehensiveness_and_sufficiency["id"] == id
            ]
        )
        min_sufficiency = lower_upper_bound_comprehensiveness_and_sufficiency_id[
            "sufficiency"
        ].values[0]

        delta = abs(row["sufficiency"] - min_sufficiency)
        sufficiency_deltas.append(delta)
    
    model_name_to_plots_dict[model_name_mappings.get(model_name)]["comprehensiveness"].extend(comprehensiveness_deltas)
    model_name_to_plots_dict[model_name_mappings.get(model_name)]["sufficiency"].extend(sufficiency_deltas)


avg_comprehension_diff = {}
avg_sufficiency_diff = {}

for model_name, plots in model_name_to_plots_dict.items():
    avg_comprehension_diff[model_name] = sum(plots["comprehensiveness"]) / len(plots["comprehensiveness"])
    avg_sufficiency_diff[model_name] = sum(plots["sufficiency"]) / len(plots["sufficiency"])


fig_comp, ax_comp = plt.subplots()
fig_suff, ax_suff = plt.subplots()

for model_name, plots in model_name_to_plots_dict.items():
    ax_comp.hist(plots["comprehensiveness"], bins=20, alpha=0.5, label=model_name)
    ax_suff.hist(plots["sufficiency"], bins=20, alpha=0.5, label=model_name)


# Save fig_comp
ax_comp.set_title("Comprehensiveness")
ax_comp.set_xlabel("Distance to upper bound")
ax_comp.set_ylabel("Frequency")
ax_comp.legend()
fig_comp.savefig("results/comprehensiveness_delta_histogram.png")

# Save fig_suff
ax_suff.set_title("Sufficiency")
ax_suff.set_xlabel("Distance to lower bound")
ax_suff.set_ylabel("Frequency")
ax_suff.legend()
fig_suff.savefig("results/sufficiency_delta_histogram.png")

print(avg_comprehension_diff)
print(avg_sufficiency_diff)