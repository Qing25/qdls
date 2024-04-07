# -*- coding: utf-8 -*-
# @File    :   example.py
# @Time    :   2023/11/23 13:46:39
# @Author  :   Qing 
# @Email   :   aqsz2526@outlook.com
######################### docstring ########################
'''
    to use the evaluator, you need to:
    implement the following functions:
        normalize_sample
        evaluate
    
'''
from qdls.data import load_json, save_json, load_jsonl
from argparse import Namespace
from evaluator import BaseEvalutor, KqaAutoEvaluator


class KqaSparqlICLEvaluator(KqaAutoEvaluator):

    @staticmethod
    def _process_vllm(query):
        return query.split("</s>")[0].strip()

    def normalize_sample(self, sample, lang='sparql'):
        d = {
            'sample_id': sample['sample_id'],
            'question': sample['question'],
            'answer': sample['answer'],
            'sparql': sample['sparql'],
            'pred': KqaSparqlICLEvaluator._process_vllm(sample['vllm']),
        }
        return d
    
    def evaluate(self):
        scores = obj.calc_marco_metrics()
        for k,v in scores.items():
            print(f"{k:<20}: {v:.4f}")

class KqaCypherICLEvaluator(KqaAutoEvaluator):

    @staticmethod
    def _process_vllm(query):
        return query.split("</s>")[0].strip()
    

    def normalize_sample(self, sample, lang='cypher'):
        d = {
            'sample_id': sample['sample_id'],
            'question': sample['question'],
            'answer': sample['answer'],
            'cypher': sample.get("cypher") if "cypher" in sample else sample['match_clause'],
            'pred': KqaCypherICLEvaluator._process_vllm(sample['vllm']),
        } 
        return d

    def evaluate(self):
        scores = self.calc_marco_metrics()
        for k,v in scores.items():
            print(f"{k:<20}: {v:.4f}")


class KqaCypherICLEvaluatorGPT(KqaAutoEvaluator):
    """ 
    """
    def normalize_sample(self, sample, lang=None):
        d = {
            'sample_id': sample['sample_id'],
            'question': sample['question'],
            'answer': sample['answer'],
            'cypher': sample.get("cypher"),
            'pred': sample['gpt-3.5-turbo-0613'],
        } 
        return d
    
    def evaluate(self):
        scores = self.calc_marco_metrics()
        for k,v in scores.items():
            print(f"{k:<20}: {v:.4f}")


if __name__ == '__main__':
    # data = load_json("/home/qing/raid/paperwork/ICL/src/cache/results/kqa_qwen14b_BM25KqaRunner_5icl.jsonl")
    # obj = KqaAutoEvaluator(data, lang='sparql', nproc=16)

    # data = load_jsonl("/home/qing/raid/paperwork/ICL/src/cache/results/kqa_qwen14b_MixKqaSparqlRunner_6icl.jsonl")
    """ 
    bleu                : 0.8521
    executable          : 0.9369
    exact_match         : 0.4193
    is_correct          : 0.5158
    ./eval_results_for_sparql.json saved with 11797 samples!
    """

    # data = load_jsonl("/home/qing/raid/paperwork/ICL/src/cache/results/kqa_qwen14b_BM25KqaRunner_3icl.jsonl")
    """ 
    bleu                : 0.8439
    executable          : 0.9164
    exact_match         : 0.4354
    is_correct          : 0.5224
    ./eval_results_for_sparql.json saved with 11797 samples!
    """

    # data = load_jsonl("/home/qing/raid/paperwork/ICL/src/cache/results/kqa_qwen14b_VectorKqaRunner_6icl.jsonl")
    """ 
    bleu                : 0.8819
    executable          : 0.9606
    exact_match         : 0.4608
    is_correct          : 0.5605
    """
    # neo4j_config = Namespace(
    #     neo4j_uri="neo4j://map:28892", neo4j_user="neo4j", neo4j_passwd="kqa", timeout=3
    # )

    # obj = KqaSparqlICLEvaluator(data, lang='sparql', nproc=32, neo4j_config=neo4j_config)
    # obj.neo4j_config = neo4j_config
    # obj.evaluate()
    # obj.save_processed()
    
    def _chatgpt_icl_example():

        neo4j_config = Namespace(
            neo4j_uri="neo4j://map:28892", neo4j_user="neo4j", neo4j_passwd="kqa", timeout=3
        )
        data = load_json("/home/qing/raid/paperwork/manage/chatgpt/cache/nl2cql_1k.json")
        obj = KqaCypherICLEvaluatorGPT(data, lang='cypher', nproc=32, neo4j_config=neo4j_config, metrics_to_calc=['exact_set_match'])
        obj.evaluate()
        obj.save_processed("/home/qing/raid/paperwork/manage/chatgpt/cache/nl2cql_1k_processed.json")
        """ 
        bleu                : 0.6302
        executable          : 0.9130
        exact_match         : 0.0230
        is_correct          : 0.1450
        exact_set_match     : 0.0520
        """

    _chatgpt_icl_example()