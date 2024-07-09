import datasets
from transformers import AutoTokenizer
from utils.tokenizer import get_word_map_callable

yelp = datasets.load_dataset("yelp_polarity", split="test")

model_names = [
    "JiaqiLee/robust-bert-yelp",
    "textattack/bert-base-uncased-yelp-polarity",
    "VictorSanh/roberta-base-finetuned-yelp-polarity",
    "textattack/bert-base-uncased-imdb",
    "textattack/roberta-base-imdb",
]

tokenizers = {}

for model_name in model_names:
    tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)


word_map_callables = {}

for model_name in model_names:
    is_roberta = "roberta" in model_name
    word_map_callables[model_name] = get_word_map_callable(is_roberta=is_roberta, text_tokenizer=tokenizers[model_name])

yelp = yelp.map(
    lambda x: {
        f"input_ids_{model_name}": tokenizer(x["text"])["input_ids"]
        for model_name, tokenizer in tokenizers.items()
    },
    batched=True,
)
yelp = yelp.map(
    lambda x: {
        f"word_map_{model_name}": word_map_callables[model_name](x[f"input_ids_{model_name}"])
        for model_name in model_names
    },
)

yelp = yelp.map(
    lambda x: {
        f"word_length_{model_name}": len(set(x[f"word_map_{model_name}"])) for model_name in model_names
    }
)

yelp = yelp.map(
    lambda x: {
        "word_length": max([x[f"word_length_{model_name}"] for model_name in model_names])
    }
)

yelp = yelp.map(
    lambda x: {
        "token_length": max([len(x[f"input_ids_{model_name}"]) for model_name in model_names])
    }
)

yelp = yelp.filter(lambda x: all(x[f"word_length_{model_name}"] == x[f"word_length_{model_names[0]}"] for model_name in model_names))
yelp = yelp.filter(lambda x: (x["word_length"] <= 12) and (x["label"] == 1))
yelp = yelp.sort("word_length", reverse=True)
yelp_df = yelp.to_pandas().reset_index()
yelp_df= yelp_df.rename(columns={"index": "id"})
yelp_df[["word_length", "token_length", "label", "id", "text"]].to_csv("yelp_polarity_test_small.csv", index=False)

