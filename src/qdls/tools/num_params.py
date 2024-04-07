# -*- coding: utf-8 -*-
# @File    :   num_params.py
# @Time    :   2024/04/06 15:18:43
# @Author  :   Qing 
# @Email   :   aqsz2526@outlook.com
######################### docstring ########################
'''
'''
import fire 

from transformers import AutoModel

def num_params(model_path):
    """ 
    model_path: str, huggingface 模型的路径
    Return: int, 模型的参数量(浮点数的个数)
    """
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型 {model_path} 的参数量为：{total_params}, {total_params/1000000000:.2f} B, {total_params/1000000:.2f} M")
    return total_params


if __name__ == '__main__':
    fire.Fire(num_params)