# -*- coding: utf-8 -*-
# @File    :   test.py
# @Time    :   2023/03/09 16:07:55
# @Author  :   Qing 
# @Email   :   aqsz2526@outlook.com
######################### docstring ########################
'''
'''

import torch


class TerminalTestor:
    """
        方便进行逐条测试
        1. 继承此类，重写 forward_fn 和 __call__
    """
    def __init__(self) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    def input_fn(self):
        s = input(f"Type in your input data:\n")
        return s 
    

    def forward(self, model, td, tokenizer):
        pred_ids = model.generate(
            input_ids=td['input_ids'] , 
            attention_mask=td['attention_mask'], 
            max_length=500, num_beams=1
        )
        texts = tokenizer.batch_decode(pred_ids, clean_up_tokenization_spaces=True, skip_special_tokens=True)
        return texts


    def __call__(self, model, tokenizer, forward_fn):
        model = model.to(self.device)
        with torch.no_grad():
            while True:
                inputs = self.input_fn()
                td = tokenizer(inputs, return_tensors='pt').to(self.device)
                output = forward_fn(model, td, tokenizer)
                print(output)
                print("="*80)


def terminal_test(config, module):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = Wrapper.load_from_checkpoint(checkpoint_path=config.preds.ckpt_path, config=config).to(device)
    model = module(config).to(device)
    
    with torch.no_grad():
        while True:
            q = input("input your question:\n")
            q = model.tokenizer.bos_token + "Here is a question: " + q + " The CQL for this question is: "
            td = model.tokenizer(q, return_tensors='pt').to(device)
            res = model.model.generate(
                input_ids=td['input_ids'].to(device), attention_mask=td['attention_mask'].to(device), 
                max_length=500, num_beams=1, pad_token_id=50256)

            texts = model.tokenizer.batch_decode(res, clean_up_tokenization_spaces=True, skip_special_tokens=True)
            print(texts)