# -*- coding: utf-8 -*-
# @File    :   metric_fns.py
# @Time    :   2023/11/20 14:05:24
# @Author  :   Qing 
# @Email   :   aqsz2526@outlook.com
######################### docstring ########################
'''
    用于评估 GQL 生成模型的指标函数
'''
from qdls.reg.register import Register
from qdls.gql.cypher.utils.syntax import syntax_check as syntax_check_cypher
from qdls.gql.sparql.utils.syntax import syntax_check as syntax_check_sparql

from qdls.gql.cypher.utils.kqa_eval import exec_one_sample_cypher
from torchmetrics.functional import sacre_bleu_score

from pygments.lexers import get_lexer_by_name
from pygments.lexers.rdf import SparqlLexer
from pygments.token import *

sp_lexer = SparqlLexer()
cy_lexer = get_lexer_by_name('py2neo.cypher') # pip install py2neo to fix

metric_fns = Register('metric_fns')

@metric_fns.register("exact_match")
def match_query(pred, gold, lexer='cypher', verbose=False):
    """ 将cypher or sparql 语言解析为语义单元，比较是否匹配
        如果出现ERROR tag 则表示生成的语法错误， return 0 
    """
    lexer = sp_lexer if lexer == 'sparql' else cy_lexer
    pred_units = []
    try:
        for tag, substr in lexer.get_tokens(pred):
            if tag is Token.Error:
                return 0
            # print(tag, substr)
            if substr.strip() != "":
                pred_units.append(substr)
    except Exception as e:
        return 0 # 语法错误,无法lex
    gold_units = [ substr for tag, substr in lexer.get_tokens(gold) if substr.strip() != "" ]
    res = (gold_units == pred_units)
    if res:
        return 1
    else:
        if verbose:
            print(f'{" ".join(gold_units)} \n {" ".join(pred_units)}')
        return 0
    

def executable_fn_cypher(pred):
    """ 判断预测的是否可执行 """
    try:
        flag, tree, parser = syntax_check_cypher(pred)
    except:
        flag = False
    return 1.0 if flag else 0.0


def executable_fn_sparql(pred):
    """ 判断预测的是否可执行 """
    try:
        flag, tree, parser = syntax_check_sparql(pred)
    except:
        flag = False
    return 1.0 if flag else 0.0

@metric_fns.register("executable")
def executable_fn(pred, lang):
    if lang == 'sparql':
        return executable_fn_sparql(pred)
    elif lang in ['cql', 'cypher']:
        return executable_fn_cypher(pred) 
    else:
        raise Exception(f"{lang} is not supported!")


@metric_fns.register("bleu")
def bleu_fn(pred, refs):
    """ 默认是用 mteval-v13a 进行分词 return bleu-4 """
    return sacre_bleu_score([pred], [refs], tokenize='char').item()


def calc_metrics_per_sample(sample, ref_key, metrics=None, neo4j_config=None):
    """ 默认的预测 key 为 pred 
        sample normalized_sample()的结果
        ref_key 为参考答案的key: cypher,cql, sparql
        metrics 为要计算的指标， 默认为 ['bleu', 'executable', 'exact_match']
        neo4j_config 为neo4j的配置 
    """
    if metrics is None:
        metrics = ['bleu', 'executable', 'exact_match']
    else:
        assert isinstance(metrics, list), f"metrics should be a list of str, but got f{metrics}"

    if 'bleu' in metrics:
        bleu = bleu_fn(sample['pred'], [sample[ref_key]])
        sample['bleu'] = bleu
    
    if 'executable' in metrics:
        sample['executable'] = executable_fn(sample['pred'], ref_key)

    if 'exact_match' in metrics:
        sample['exact_match'] = match_query(sample['pred'], sample[ref_key], lexer=ref_key)

    if 'is_correct' in metrics or 'exec_info' in metrics:
        assert ref_key in ['cql', 'cypher'], f"only cql or cypher is supported, but got {ref_key}"
        try:
            is_correct, info, results = exec_one_sample_cypher(sample, neo4j_config, key='pred')
        except Exception as e:
            print(e)
            is_correct, info, results = False, "Timeout", '[]'
        sample['is_correct'] = is_correct
        sample['exec_info'] = info
        sample['exec_results'] = results

    # import pdb;pdb.set_trace();
    return sample

def nltk_bleu():
    import nltk
    reference = ['今天天气很好吗']

    # 候选翻译（机器翻译结果）
    candidate = '今日天气很好。'

    # 将参考翻译和候选翻译转换为标记化的句子列表
    reference_tokens = nltk.word_tokenize(reference[0], language='chinese')
    candidate_tokens = nltk.word_tokenize(candidate, language='chinese')

    # 计算BLEU分数
    bleu_score = nltk.translate.bleu_score.sentence_bleu([reference_tokens], candidate_tokens)

    print("BLEU score:", bleu_score)

if __name__ == '__main__':
    # print(bleu_fn('hello world', ['hello world']))
    nltk_bleu()