
from typing import * 
import torch 
from qdls.data import sequence_padding

from ..register import registers

@registers.collator.register("causal_collator")
class CausalCollator:
    """ 
        训练时right padding，测试时left padding
        fit for all causal language model
    """
    def __init__(self, tokenizer, mode='train') -> None:
        self.tokenizer = tokenizer
        self.mode = mode
        assert self.tokenizer.pad_token_id is not None, "tokenizer must have pad_token_id"

    def __call__(self, features) -> Any:
        if self.mode == 'test':
            input_ids= sequence_padding([x['input_ids'] for x in features ], padding=self.tokenizer.pad_token_id, padding_side='left')
            attention_mask= sequence_padding([ x['attention_mask'] for x in features ], padding=0, padding_side='left')
            labels = None 
        else:
            input_ids= sequence_padding([x['input_ids'] for x in features ], padding=self.tokenizer.pad_token_id)   #  
            attention_mask= sequence_padding([ x['attention_mask'] for x in features ])
            labels = sequence_padding([x['labels'] for x in features ], padding=self.tokenizer.pad_token_id)

        return {
            'input_ids' : torch.tensor(input_ids).long() ,                  # collate时的截断需要重新考虑 TODO
            'attention_mask' : torch.tensor(attention_mask).long(),
            'labels': torch.tensor(labels).long() if labels is not None else None ,
        }
        

@registers.collator.register("seq2seq_collator")
class Seq2seqCollator:
    """ 
        T5, Bart 等seq2seq模型的collator
    """
    def __init__(self, tokenizer, mode='train') -> None:
        self.tokenizer = tokenizer
        self.mode = mode

    def __call__(self, features) -> Any:
        if self.mode == 'test':
            input_ids= sequence_padding([x['input_ids'] for x in features ], padding=self.tokenizer.pad_token_id)
            attention_mask= sequence_padding([ x['attention_mask'] for x in features ], padding=0)
            labels = None 
        else:
            input_ids= sequence_padding([x['input_ids'] for x in features ], padding=self.tokenizer.pad_token_id)   #  
            attention_mask= sequence_padding([ x['attention_mask'] for x in features ], padding=0)
            labels = sequence_padding([x['labels'] for x in features ], padding=self.tokenizer.pad_token_id)

        return {
            'input_ids' : torch.tensor(input_ids).long(),
            'attention_mask' : torch.tensor(attention_mask).long(),
            'labels': torch.tensor(labels).long() if labels is not None else None ,
        }
    
