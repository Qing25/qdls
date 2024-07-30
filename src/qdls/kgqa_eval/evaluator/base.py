# -*- coding: utf-8 -*-
# @File    :   base.py
# @Time    :   2023/11/08 13:50:38
# @Author  :   Qing 
# @Email   :   aqsz2526@outlook.com
######################### docstring ########################
'''
'''


####################### imports ###########################

from abc import ABC, abstractmethod
import datasets 
from tqdm import tqdm 
from collections import defaultdict
from qdls.data import load_json, save_json
from qdls.utils import print_string
from qdls.kgqa_eval.evaluator.metric_fns import calc_metrics_per_sample

class BaseEvalutor(ABC):
    """ 
        实现 __init__ 和 normalize_sample 函数
    """
    def __init__(self, file=None, refer_key='cypher', nproc=1, **kwargs) -> None:
        self.refer_key = refer_key
        self.nproc = nproc
        self.metrics_to_calc = []
        self._manage_metrics_to_calc(kwargs)
        print_string(f"metrics to calc: {self.metrics_to_calc}")
        print_string(f"neo4j config: {kwargs.get('neo4j_config', '|| please set neo4j_config in kwargs ||')}")

        if file is not None:
            self.raw_data = load_json(file) if type(file) is str else file 
            
            self.data_to_eval = [ self.normalize_sample(s, refer_key) for s in self.raw_data ]
            self._pre_calc_metrics()
            self.id2sample = {sample['sample_id']:sample for sample in self.data_to_eval}

            

    @abstractmethod
    def normalize_sample(self, sample, *args, **kwargs):
        """ standardize the sample to a dict with keys 
            sample_id

        IMPORTANT: values should not be dict, or it will be extremely slow
        """
        return sample
    
    def _manage_metrics_to_calc(self, kwargs):
        """  
            增加在 init 函数通过 kwargs 传入新的 metrics_to_calc
            lang 是 cypher 和 sparql 默认增加 is_correct
        """
        # if self.refer_key in ['cypher', 'sparql']:
        #     self.metrics_to_calc.append("is_correct")  # 默认是要计算准确率的 TODO:但是对应的是 kqa 的计算，考虑删除
        if "metrics_to_calc" in kwargs:
            for metric in kwargs['metrics_to_calc']:
                if metric not in self.metrics_to_calc:
                    self.metrics_to_calc.append(metric)   

    def evaluate(self, verbose=True):
        """ 读取已经计算完毕的 metrics 数据，在数据集上计算 macro 指标，并打印出来"""
        scores = self.calc_marco_metrics()
        if verbose:
            for k,v in scores.items():
                print(f"{k:<20}: {v:.4f}")
        return scores
    
    def _pre_calc_metrics(self):
        """ 使用 datasets库 并行计算所有指标 """
        ds = datasets.Dataset.from_list(self.data_to_eval)
        print(ds)
        
        if self.nproc > 1:
            ds = ds.map(
                calc_metrics_per_sample, num_proc=self.nproc, 
                fn_kwargs={
                    'ref_key':self.refer_key, 
                    'metrics':self.metrics_to_calc,
                    'neo4j_config': self.neo4j_config
                }
            )
            self.data_to_eval = ds.to_list()
        else:
            self.data_to_eval = [] 
            for s in tqdm(ds):
                self.data_to_eval.append(
                    calc_metrics_per_sample(s, self.refer_key, self.metrics_to_calc, self.neo4j_config)
                )
    
    def calc_marco_metrics(self, ids=None):
        """ 根据预先计算好的metrics, 为ids中的样本计算平均值 """
        if ids is None:
            ids = self.id2sample.keys()
        L = len(ids)
        scores = defaultdict(list)
        for i in ids:
            sample = self.id2sample[i]    
            for m in self.metrics_to_calc:
                scores[m].append(float(sample.get(m, 0)))
        return {k:sum(v)/L for k,v in scores.items()}
    
    def save_with_results(self, target_path):
        save_json(self.data_to_eval, target_path)  
    
 