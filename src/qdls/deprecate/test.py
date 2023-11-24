# -*- coding: utf-8 -*-
# @File    :   test.py
# @Time    :   2023/03/09 16:07:55
# @Author  :   Qing 
# @Email   :   aqsz2526@outlook.com
######################### docstring ########################
'''
'''
import abc
import torch
from torch.utils.data import DataLoader
import datasets
from ..data import load_json, save_json

class GenerationPipeline():
    """ 传入model 和 tokenizer
        自行实现save_fn保存结果
        __call__ 进行预测和生成
    """
    def __init__(self, model, tokenizer) -> None:
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
        self.model = model.to(self.device)
        self.tokenizer = tokenizer

    @abc.abstractmethod
    def save_fn(self, ):

        pass 

    @abc.abstractmethod
    def __call__(self, data, save_fn=None):
        

        return 


class TerminalTestor:
    """
        方便进行逐条测试
        1. 继承此类，重写/传入 forward_fn 和 __call__
    """
    def __init__(self) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

    def input_fn(self):
        s = input(f"Type in your input data:\n")
        return s 
    

    def forward(self, model, td, tokenizer, generation_config=None):
        """
            model: torch model
            td: tokenized results from tokenizer
            tokenizer: transformers tokenizer
            generation_config: transformers generation config, transformers version >= 4.28
        """
        if generation_config is None:
            pred_ids = model.generate(
                input_ids=td['input_ids'] , 
                attention_mask=td['attention_mask'], 
                max_length=500, num_beams=1
            )
        else:
            pred_ids = model.generate(
                input_ids=td['input_ids'] , 
                attention_mask=td['attention_mask'], 
                generation_config=generation_config 
            )
        texts = tokenizer.batch_decode(pred_ids, clean_up_tokenization_spaces=True, skip_special_tokens=True)
        return texts


    def __call__(self, model, tokenizer, forward_fn):
        """调用方法

        Args:
            model: torch model
            tokenizer: transformers tokenizer
            forward_fn: 自行编写 or 使用 self.forward
        """
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

if __name__ == '__main__':
    pass 

