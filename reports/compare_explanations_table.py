import pandas as pd

dataset_names = ["yelp", "sst2"]
model_names = [
    "textattack/bert-base-uncased-SST-2",
    "textattack/roberta-base-SST-2",
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
    "sufficiency_order": "Best sufficiency order",
    "comprehensiveness_order": "Best comprehensiveness order",
}
for dataset_name in dataset_names:
    dataframe_list = []
    for model_name in model_names:
        aopc_scores_df = pd.read_csv(
            f"results/aopc_scores_short/{dataset_name}_{model_name.split('/')[1]}.csv"
        )
        aopc_scores_df["model"] = model_dict[model_name]
        

        aopc_limits_df = pd.read_csv(
            f"results/aopc_limits_exact/{dataset_name}_{model_name.split('/')[1]}.csv"
        )
        aopc_scores_df = aopc_scores_df.merge(aopc_limits_df, on="id")
        
        dataframe_list.append(aopc_scores_df)

    df = pd.concat(dataframe_list)
    df = df[df["prob"]>0.5]
    df["explanation_method"] = df["explanation_method"].apply(lambda x: explainer_dict[x])
    df = df[df["explanation_method"]!="AttInGrad"]
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
    print(results.to_latex(hrules=True, multicol_align=True).replace("\multirow", "\midrule\n\multirow"))

