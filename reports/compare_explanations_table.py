import pandas as pd

model_names = [
    "JiaqiLee/robust-bert-yelp",
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

explainer_dict = {
    "occlusion_1": "Occlusion@1",
    "lime": "LIME",
    "kernelshap": "SHAP",
    "integrated_gradient": "IG",
    "random_baseline": "Random",
    "gradient_x_input": "G x I",
    "attention": "Attention",
    "attingrad": "AttInGrad",
    "decompx": "DecompX",
    "deeplift": "DeepLIFT",
}

datafram_list = []
for model_name in model_names:
    results_df = pd.read_csv(
        f"results/yelp_polarity_results_{model_name.split('/')[1]}.csv"
    )
    results_df["model"] = model_dict[model_name]
    datafram_list.append(results_df)
df = pd.concat(datafram_list)
df = df[df["prob"]>0.5]
df["explanation_method"] = df["explanation_method"].apply(lambda x: explainer_dict[x])
results = df.groupby([ "explanation_method", "model"]).agg(
    comprehensiveness_mean=('comprehensiveness', 'mean'),
    normalized_comprehensiveness_mean=('normalized_comprehensiveness', 'mean'),
    sufficiency_mean=('sufficiency', 'mean'),
    normalized_sufficiency_mean=('normalized_sufficiency', 'mean')
).reset_index()
results.set_index(['explanation_method', 'model'], inplace=True)
results = results.style.format(
    decimal=".",
    thousands=" ",
    precision=2,
)
print(results.to_latex(hrules=True, multicol_align=True))

