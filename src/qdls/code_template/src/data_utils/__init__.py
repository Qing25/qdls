
from .common import * 


if __name__ == '__main__':
    PLM = "/pretrains/pt/fb_bart"
    DATA = ""
    TOKENIZE_FN = None   # from .kqa import nl2cy_tokenization
    from qdls.data import load_json
    from qdls.utils import print_dict
    from qdls.reg.collators.common_collators import Seq2seqCollator
    
    import datasets 
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    tokenizer = AutoTokenizer.from_pretrained(PLM)
    tokenizer.pad_token_id = 0 
    

    data = load_json(DATA)[:100]
    for sample in data:
        r = TOKENIZE_FN(example=sample, tokenizer=tokenizer)
        print_dict(r)
        print(tokenizer.convert_ids_to_tokens(r['distill_labels']))
        input()
        break
    ds = datasets.Dataset.from_list(data)
    ds = ds.map(TOKENIZE_FN, num_proc=1, fn_kwargs={'tokenizer':tokenizer})
    loader = DataLoader(ds, batch_size=4, collate_fn=Seq2seqCollator(tokenizer))
    for batch in loader:
        print(batch)
        # print(tokenizer.batch_decode(batch['labels'], skip_special_tokens=False, clean_up_tokenization_spaces=False))
        # print(tokenizer.batch_decode(batch['distill_labels'], skip_special_tokens=False, clean_up_tokenization_spaces=False))
        import pdb;pdb.set_trace();
        break