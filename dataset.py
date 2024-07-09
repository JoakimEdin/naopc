import torch
from datasets import Dataset
import torch.utils
import itertools
from utils.tokenizer import convert_word_map_to_dict

class PerturbDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset: Dataset, mask_token_id: int, pad_token_id: int, input_ids_column_name: str = "input_ids", word_map_callable: callable = None):
        self.dataset = dataset
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.input_ids_column_name = input_ids_column_name
        self.word_map_callable = word_map_callable


    def __iter__(self) -> tuple[int, str, torch.Tensor]:
        for example in self.dataset:
            input_ids = torch.tensor(example[self.input_ids_column_name])
            
            word_map = convert_word_map_to_dict(self.word_map_callable(input_ids))

            permutation = list(range(1, len(word_map.keys()) - 1))
            for i in range(len(permutation) + 1):
                for mask_indices in itertools.combinations(permutation, i):
                    try:
                        mask_indices_mapped = list(itertools.chain.from_iterable(word_map[x] for x in mask_indices))
                    except KeyError:
                        continue
                    temp = input_ids.clone()
                    if len(mask_indices) > 0:
                        temp[mask_indices_mapped] = self.mask_token_id
                    yield example["id"], str(sorted(mask_indices_mapped)), str(sorted(list(mask_indices))), temp
    
    def collate_fn(self, batch):
        ids, token_key, word_key, input_ids = zip(*batch)
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        attention_mask = (input_ids != self.pad_token_id).float()
        return list(ids), (token_key), (word_key), input_ids, attention_mask



        

