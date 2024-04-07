import re
import sys
import os
import json

current_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_directory)
from match_evaluate import find_matches
from where_evaluate import find_wheres
from with_evaluate import find_withs
from return_evaluate import find_returns
import pandas as pd
from pprint import pprint
from utils import special_sort


def text_parts(text):
    match_pattern = re.compile(r'match (.*?)(?=where|return|with|$)')
    where_pattern = re.compile(r'where (.*?)(?=with|return|$)')
    with_pattern = re.compile(r'with (.*?)(?=match|return|$)')
    return_pattern = re.compile(r'return (.*)$')

    match_part = [i.strip() for i in match_pattern.findall(text)]
    where_part = [i.strip() for i in where_pattern.findall(text)]
    with_part = [i.strip() for i in with_pattern.findall(text)]
    return_part = [i.strip() for i in return_pattern.findall(text)]

    parts = {
        'match': match_part,
        'where': where_part,
        'with': with_part,
        'return': return_part,
    }
    return parts


def combine_match2where(split):
    match_where = split['match']['match_where']
    split['where']['where_calculates'].extend(match_where)
    del split['match']['match_where']
    return split


def post_split(split):
    # sort+set+去空格
    for key, value in split.items():
        for k, v in value.items():
            split[key][k] = special_sort(set([vv.replace(' ', '') for vv in v if type(vv) is str]))
    return split


def combine_split(split):
    # 合并
    for key, value in split.items():
        if value:
            merged_dict = value[0]
            for d in value:
                for k, v in d.items():
                    merged_dict[k].extend(v)
            split[key] = merged_dict
    return split


def split(text):
    # print(text)
    text = text.lower()  # 小写化处理
    text = text.replace('\n', ' ')  # 去除换行
    text = text.replace('"', "'")  # 统一引号
    text = re.sub(r'\s+', ' ', text)  # 去除多个连续空格
    text = text.replace("( ", "(").replace(" )", ")").replace(". ", ".")  # 去除多余空格
    parts = text_parts(text)
    # print(parts)

    match_splits = [{'match_tags': [],'match_where': []}]
    where_splits = [{'where_calculates': [], 'where_temps': []}]
    with_splits = [{'with_rag_attrs': [], 'with_calculates': [], 'with_functions': [], 'with_temps': [], 'with_orderbys': []}]
    return_splits = [{'return_rag_attrs': [], 'return_calculates': [], 'return_functions': [], 'return_orderby': [], 'return_temps': [], 'return_limit': []}]

    for part in parts['match']:
        match_split = find_matches(part)
        match_splits.append(match_split)
        # print(1)
    for part in parts['where']:
        where_split = find_wheres(part)
        where_splits.append(where_split)
        # print(2)
    for part in parts['with']:
        with_split = find_withs(part)
        with_splits.append(with_split)
        # print(3)
    for part in parts['return']:
        return_split = find_returns(part)
        return_splits.append(return_split)
        # print(4)
    split = {
        'match': match_splits,
        'where': where_splits,
        'with': with_splits,
        'return': return_splits,
    }
    # pprint(split)
    split = combine_split(split)
    # pprint(split)
    split = combine_match2where(split)
    # pprint(split)
    split = post_split(split)
    return split


def eval_in(list1, list2):
    list1 = [i for i in list1 if 'date' not in i]
    list2 = [i for i in list2 if 'date' not in i]
    return all(elem in list2 for elem in list1)


def del_section(dict, key_list):
    for key in key_list:
        if key in dict:
            del dict[key]
    return dict


def evaluate_em(gold, pre, escape_part=[], escape_section=[]):
    '''
    EM-evaluation for cypher
    :param gold: cypher1
    :param pre: cypher2
    :param escape_part: string-list escape_part = ["match", "where", "with", "return"]
    :param escape_section: string-list
    escape_section = ["match_tags",
    "where_calculates", "where_temps",
    "with_rag_attrs", "with_calculates", "with_functions", "with_temps",
    "return_rag_attrs", "return_calculates", "return_functions", "return_orderby", "return_temps", "return_limit"]
    :return: evaluation_result, gold_split_dict, pre_split_dict, gold_split_dict_error, pre_split_dict_error
    '''
    splits_gold = split(gold)
    splits_pre = split(pre)

    for key in splits_gold.keys():
        if key in escape_part:
            continue
        del_section(splits_gold[key], escape_section)
        del_section(splits_pre[key], escape_section)
        if splits_gold[key] != splits_pre[key]:
            return False, splits_gold, splits_pre, splits_gold[key], splits_pre[key]
    return True, splits_gold, splits_pre, None, None


def read_json_data(source_file):
    with open(source_file, "r", encoding="utf8") as f:
        dataset = json.load(f)
    return dataset


if __name__ == '__main__':

    cypher = ("MATCH (s:stock{name: '[s]'})-[:has_stock_data]->(sd:stock_data{`date`: date('[d]')}) "
              "WITH sd.stock_data.volume AS temp "
              "MATCH (s:stock)-[:has_stock_data]->(sd:stock_data{`date`: date('[d]')}) "
              "WHERE sd.stock_data.volume == temp "
              "RETURN s.stock.name LIMIT 10")
    cypher1 = ("MATCH (s:stock{name: '[s]'})-[:has_stock_data]->(sd:stock_data{`date`: date('[d]')}) "
              "WITH sd.stock_data.volume AS temp "
              "MATCH (s:stock)-[:has_stock_data]->(sd:stock_data{`date`: date('[d]')}) "
              "WHERE sd.stock_data.volume == temp "
              "RETURN s.stock.name LIMIT 100")
    cypher2 = ("MATCH (pof1:public_offering_fund{name: '[f1]'})<-[:manage]-(fm:fund_manager)-[:manage]->(pof2:public_offering_fund{name: '[f2]'}) "
              "RETURN fm.fund_manager.name")


    splits_cypher = split(cypher)
    pprint(splits_cypher)

    splits_cypher = split(cypher1)
    pprint(splits_cypher)

    splits_cypher = split(cypher2)
    pprint(splits_cypher)

    result, _, _, _, _ = evaluate_em(cypher, cypher1, escape_section=['return_limit'])
    print(result)
    result, _, _, _, _ = evaluate_em(cypher, cypher2)
    print(result)