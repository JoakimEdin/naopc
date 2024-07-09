import torch
from transformers import RobertaTokenizer, BertTokenizer, PreTrainedTokenizerBase
import re




def get_tokens(
    input_ids: torch.Tensor | list, text_tokenizer: PreTrainedTokenizerBase
) -> list[str]:
    return text_tokenizer.convert_ids_to_tokens(
        torch.tensor(input_ids).squeeze().tolist()
    )

def convert_word_map_to_dict(word_map: torch.Tensor) -> dict[int, list[int]]:
    """Convert a word map to a dictionary where the key is the word index and the value is a list of token indices that belong to that word.
    
    Args:
        word_map (torch.Tensor): Word map to convert
    
    Returns:
        dict[int, list[int]]: Word map as a dictionary
    """
    word_map = word_map.tolist()
    return {k: [i for i, x in enumerate(word_map) if x == k] for k in set(word_map)}

def get_word_map_callable(
    text_tokenizer: RobertaTokenizer | BertTokenizer,
    include_space: bool = True,
    is_roberta: bool = False,
):

    def get_word_map_bert(
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Get a list mapping each token to a word. This is necessary because the BERT tokenizer splits words into subwords.
        The spaces are joined with the preceding word.
        
        Args:
            input_ids (torch.Tensor): Input ids to get word map for
            text_tokenizer (BertTokenizer): Text tokenizer to use
            include_space (bool): Include spaces in the word map
        
        Returns:
            torch.Tensor: Word map of same length as input_ids
        """
        tokens = get_tokens(input_ids, text_tokenizer)
        word_map = []
        idx = 0
        space_between_words = False
        special_token_seen = False

        for i, token in enumerate(tokens):
            if re.match(r"^\[.*\]$", token):
                special_token_seen = False
                if idx in word_map:
                    idx += 1
                word_map.append(idx)
                idx += 1
                space_between_words = False
                continue

            if token.startswith("##"):
                special_token_seen = False
                word_map.append(idx)
                continue

            if token == "\\" and i + 1 < len(tokens) - 1 and tokens[i+1].startswith("n"):
                idx += 1
                word_map.append(idx)
                continue

            # If token is all numbers and the previous token is a currency symbol, we treat it as a separate word
            if re.match(r"^[0-9]*$", token) and i > 0 and re.match(r"^[\$\#\¢\£\€]*$", tokens[i-1]):
                word_map.append(idx)
                continue

            # This is slightly wonky, but the Roberta BPE encoding handles special characters in 2 ways:
            # 1) If there is only a single special character, it is joined with the preceding word
            # 2) If there are multiple special characters, they are treated as separate words
            # So if we see a special character, we need to check if the next token is also a special character
            # If it is, we increment the word index to start a new word, otherwise we do not.
            # If the current token is a special character and the previous token is also a special character,
            # we do not increment the word index.
            if not re.match(r"^[a-zA-Z0-9]*$", token):
                if not special_token_seen:
                    idx += 1
                if i + 1 < len(tokens) - 1 and not re.match(r"^[a-zA-Z0-9]*$", tokens[i+1]):
                    if not special_token_seen:
                        special_token_seen = True
                word_map.append(idx)
                continue

            if space_between_words and not special_token_seen:
                if not tokens[i-1] == "'":
                    idx += 1

            if special_token_seen:
                idx += 1

            word_map.append(idx)
            space_between_words = True
            special_token_seen = False

        return torch.tensor(word_map)

    def get_word_map_roberta(
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Get a list mapping each token to a word. This is necessary because the Roberta tokenizer splits words into subwords.
        The spaces are joined with the preceding word.
        Example:
        input_ids = [0, 1, 2, 3, 4, 5, 6, 7]


        Args:
            input_ids (torch.Tensor): Input ids to get word map for
            text_tokenizer (RobertaTokenizer): Text tokenizer to use

        Returns:
            list[int]: Word map of same length as input_ids
        """
        tokens = get_tokens(input_ids, text_tokenizer)
        word_map = []
        idx = 0
        space_between_words = False
        for token in tokens:
            if re.match(r"^<.*>$", token):
                if idx in word_map:
                    idx += 1
                word_map.append(idx)
                idx += 1
                space_between_words = False
                continue

            # if the token only contains Ċ or Ġ characters
            if re.match(r"^[ĊĠ]+$", token):
                space_between_words = True
                if not include_space:
                    idx += 1
                word_map.append(idx)
                continue


            if token.startswith("Ġ"):
                space_between_words = True
                
            if not re.match(r"^[a-zA-Z0-9]*$", token) and not token.startswith("Ġ"):
                idx += 1
                word_map.append(idx)
                space_between_words = True
                continue

            if space_between_words:
                idx += 1
                space_between_words = False

            word_map.append(idx)

        return torch.tensor(word_map)
    
    if is_roberta:
        return get_word_map_roberta
    else:
        return get_word_map_bert
    


if __name__ == "__main__":
    import datasets
    yelp = datasets.load_dataset("csv", data_files="yelp_polarity_test_small.csv", split="train")

    roberta_tokenizer = RobertaTokenizer.from_pretrained("VictorSanh/roberta-base-finetuned-yelp-polarity")
    bert_tokenizer = BertTokenizer.from_pretrained("textattack/bert-base-uncased-yelp-polarity")

    roberta_callable = get_word_map_callable(roberta_tokenizer, is_roberta=True)
    bert_callable = get_word_map_callable(bert_tokenizer)

    mishaps = 0
    for example in yelp:
        bert_ids = bert_tokenizer(example["text"])["input_ids"] 
        roberta_ids = roberta_tokenizer(example["text"])["input_ids"]
        bert_word_map = bert_callable(bert_ids)
        roberta_word_map = roberta_callable(roberta_ids)

        if set(bert_word_map.tolist()) != set(roberta_word_map.tolist()):
            mishaps += 1

    print(f"Total pass: {len(yelp) - mishaps}, Total mishaps: {mishaps}")