import os 
import pandas as pd 
from typing import Any
from loguru import logger
from qdls.data import load_json, save_json
from qdls.utils import print_string
from collections import defaultdict
from tqdm import tqdm
from argparse import Namespace
 
import datasets 

from .base import BaseEvalutor
from .metric_fns import calc_metrics_per_sample

example_neo4j_config = Namespace(
    neo4j_uri="neo4j://$IP:$PORT", neo4j_user="neo4j", neo4j_passwd="kqa", timeout=3
)

example_virtuoso_config = Namespace(
    virtuoso_address = "http://127.0.0.1:28890/sparql",
    virtuoso_graph_uri = 'kqaspcy'
)


class KqaAutoEvaluator(BaseEvalutor):
    def __init__(self, file=None, lang='cypher', nproc=1, **kwargs) -> None:
        self.lang = lang
        self.nproc = nproc
        self.neo4j_config = kwargs.get('neo4j_config', "please set neo4j_config in kwargs")
        self.metrics_to_calc = ['bleu', 'executable', 'exact_match']
        if lang == 'cypher':
            self.metrics_to_calc.extend(['is_correct'])

        print_string(f"metrics to calc: {self.metrics_to_calc}")
        print_string(f"neo4j config: {self.neo4j_config}")

        if file is not None:
            self.raw_data = load_json(file) if type(file) is str else file 
            self.length2sids = self.length_to_sampleids()
            
            self.data_to_eval = [ self.normalize_sample(s, lang) for s in self.raw_data ]
            self.__pre_calc_metrics()
            self.id2sample = {sample['sample_id']:sample for sample in self.data_to_eval}


        if lang == 'sparql':
            print_string(f"to evaluate for sparql generation, calling obj.eval_sparql()")
            self.eval_sparql()
            self.metrics_to_calc.extend(['is_correct'])

    def eval_sparql(self):
        from .sparql_utils import exec_one_sample 
        from multiprocessing import Pool 
        with Pool(self.nproc) as p:
            R = list(tqdm(p.imap(exec_one_sample, self.data_to_eval), total=len(self.data_to_eval), desc=f"executing sparql with {self.nproc} processes"))
        for sample, (is_correct, info) in zip(self.data_to_eval, R):
            sample['is_correct'] = is_correct
            sample['exec_info'] = info
        

    @classmethod
    def from_processed(cls, file, lang='cypher'):
        data = load_json(file)
        obj = cls(lang=lang)
        obj.data_to_eval = data
        obj.id2sample = {sample['sample_id']:sample for sample in obj.data_to_eval}
        obj.length2sids = obj.length_to_sampleids()
        return obj 

    def save_processed(self, target_path=None):
        if target_path is None:
            target_path = f"./eval_results_for_{self.lang}.json"
        save_json(self.data_to_eval, target_path)  

    def __pre_calc_metrics(self):
        """ 使用datasets库 并行计算 """
        ds = datasets.Dataset.from_list(self.data_to_eval)
        print(ds)
        
        if self.nproc > 1:
            ds = ds.map(
                calc_metrics_per_sample, num_proc=self.nproc, 
                fn_kwargs={
                    'ref_key':self.lang, 
                    'metrics':self.metrics_to_calc,
                    'neo4j_config': self.neo4j_config
                }
            )
            self.data_to_eval = ds.to_list()
        else:
            self.data_to_eval = [] 
            for s in tqdm(ds):
                self.data_to_eval.append(
                    calc_metrics_per_sample(s, self.lang, self.metrics_to_calc, self.neo4j_config)
                )
            
    def _get_sample_length(self, sample):
        return len(sample['program'])
    
    def length_to_sampleids(self):
        """ 按照program（KoPL）的长度，分组 sample_id
        2 452 0.051
        3 1890 0.266
        4 2414 0.541
        5 1072 0.662
        6 1438 0.826
        7 460 0.878
        8 742 0.963
        9 235 0.989
        10 7 0.990
        11 77 0.999
        12 6 1.000
        14 4 1.000
        """
        length2sids = defaultdict(list)
        for sample in self.raw_data:
            length2sids[self._get_sample_length(sample)].append(sample['sample_id'])
        length2sids = dict(sorted(length2sids.items(), key=lambda x:x[0]))
        return length2sids
    
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
    
    @staticmethod
    def complexity_fn(length2sids):
        """ 划分困难的标准 """
        Easy, Medium, Hard = [], [], [] 
        for k,v in length2sids.items():
            if k <=3:
                Easy.extend(v)
            elif k <= 6:
                Medium.extend(v)
            else:
                Hard.extend(v)

        return Easy, Medium, Hard
    
    def detailed_length_metrics(self):
        """ 将问题按照长度分类，计算每个类别的问题的指标 """
        Easy, Medium, Hard = self.complexity_fn(self.length2sids)
        Easy_metrics = self.calc_marco_metrics(Easy)
        Medium_metrics = self.calc_marco_metrics(Medium)
        Hard_metrics = self.calc_marco_metrics(Hard)
        return Easy_metrics, Medium_metrics, Hard_metrics

    def exec_results(self):
        num = sum([x['exec_acc'] for k,x in self.id2sample.items()])
        acc = num /len(self.id2sample)
        print(f"exec acc: {acc}, {num} ")