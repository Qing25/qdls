from argparse import Namespace
from qdls.data import load_json, save_json
from qdls.utils import print_string
import pandas as pd
import datasets 
from tqdm import tqdm
from collections import defaultdict
from neo4j import GraphDatabase

from qdls.kgqa_eval.evaluator.base import BaseEvalutor
from qdls.kgqa_eval.evaluator.metric_fns import calc_metrics_per_sample, metric_fns


def get_answer(query, neo4j_config):
    driver = GraphDatabase.driver(uri=neo4j_config.neo4j_uri, auth=(neo4j_config.neo4j_user, neo4j_config.neo4j_passwd))
    with driver.session() as session:
        result = session.run(query)
        data = result.data()
    return sum([list(d.values()) for d in data], [])

def unwind_list(nested_list):
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(unwind_list(item))
        else:
            result.append(str(item))
    return result

@metric_fns.register("metaqa_acc")
def metaqa_acc(sample, **kwargs):
    gold= sample['answers']
    try:
        answers = get_answer(sample['pred'], neo4j_config=kwargs['neo4j_config']) # 从neo4j中查询
    except: # 语法有误/ 执行错误直接视为[]
        answers = []
    pred = [_ for _ in answers if _ is not None]
    pred = unwind_list(pred)
    gold = unwind_list(gold)
    if set(gold).issubset(set(pred)):
        return True
    else:
        return False 
    

class MetaqaEval(BaseEvalutor):
    def __init__(self, file=None, lang='cypher', nproc=1, **kwargs) -> None:
        self.lang = lang
        self.nproc = nproc
        self.neo4j_config = kwargs.get('neo4j_config', "please set neo4j_config in kwargs")
        if kwargs.get("metrics_to_calc", None) is not None:
            self.metrics_to_calc = kwargs['metrics_to_calc']
        else:
            self.metrics_to_calc = ['bleu', 'executable', 'exact_match', "metaqa_acc"]
        # if lang == 'cypher':
        #     self.metrics_to_calc.extend(['is_correct'])

        print_string(f"metrics to calc: {self.metrics_to_calc}")
        print_string(f"neo4j config: {self.neo4j_config}")

        if file is not None:
            self.raw_data = load_json(file) if type(file) is str else file   
            for i, s in enumerate(self.raw_data):
                s['sample_id'] = i
            self.data_to_eval = [ self.normalize_sample(s, lang) for s in self.raw_data ]
            self.__pre_calc_metrics()
            self.id2sample = {sample['sample_id']:sample for sample in self.data_to_eval}
    
    def evaluate(self, verbose=True):
        scores = self.calc_marco_metrics()
        if verbose:
            for k,v in scores.items():
                print(f"{k:<20}: {v:.4f}")
        return scores
    
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

    
class MetaqaCausalEval(MetaqaEval):
    @staticmethod
    def _process_raw(query):
        cql = query.split('[query]')[-1]
        return cql.strip()
    
    def normalize_sample(self, sample, lang=None):
        d = {
            'sample_id': sample['sample_id'],
            'question': sample['question'],
            'answers': sample['answers'],
            'cypher': sample['cypher'],
            'pred': MetaqaCausalEval._process_raw(sample['pred']),
        }
        return d

class MetaqaSeq2seqEval(MetaqaEval):
    def normalize_sample(self, sample, lang=None):
        pred = sample['pred'].strip()
        d = {
            'sample_id': sample['sample_id'],
            'question': sample['question'],
            'answers': sample['answers'],
            'cypher': sample['cypher'],
            'pred': pred,
        }
        return d
