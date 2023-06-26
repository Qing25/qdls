# -*- coding: utf-8 -*-
# @File    :   kqa_eval.py
# @Time    :   2023/06/26 16:36:09
# @Author  :   Qing 
# @Email   :   aqsz2526@outlook.com
######################### docstring ########################
'''
    elvauate KQA Cypher
'''

import os 
import neo4j
from neo4j import GraphDatabase, Query

from qdls.data import load_json, save_json
from qdls.gql.cypher.utils.syntax import syntax_check

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from argparse import Namespace
from tqdm import tqdm

from qdls.utils import print_string

units = ['metre', 'pound', 'minute', 'United States dollar', 'mile', '1', 'centimetre', 'square metre', 'square kilometre', 
'hour', 'second', 'kilogram', 'inch', 'hertz', 'foot', 'Japanese yen', 'year', 'square mile', 'kilometre', 'act', 'audio track',
 'millimetre', 'Danish krone', 'hectare', 'euro', 'Argentine peso', 'beats per minute', 'Nigerian naira', 'barrels per day', 
 'kilogram per cubic metre', 'kilometre per hour', 'month', 'annum', 'Singapore dollar', 'cubic metre per second', 'Czech koruna', 
 'day', 'pound sterling', 'percentage', 'acre', 'workforce', 'square foot', 'South Korean won', 'Canadian dollar', 'Indian rupee', 
 'reservist', 'volt', 'Russian ruble', 'active duty military personnel', 'son', 'atomic mass unit', 'Hong Kong dollar', 
 'years old', 'drawing', 'book', 'gram per cubic centimetre', 'painting', 'gram per kilogram', 'sculpture', 'degree Celsius', 
 'cubic hectometre', 'photograph', 'yottagram', 'gram per mole', 'geographic map', 'astronomical unit', 'Australian dollar', 
 'degree', 'Philippine peso', 'chapter', 'Deutsche Mark', 'cubic kilometre', 'gigawatt', 'solar radius', 'kelvin', 
 'disability-adjusted life year', 'solar mass', 'degree Fahrenheit', 'millimeter of mercury', 'milligram per cubic meter', 
 'volume percent', 'electronvolt', 'metre per second', 'hectopascal', 'joule per mole-kelvin', 'sol', 'daughter', 
 'kilojoule per mole', 'watt per meter kelvin', 'gigapascal', 'cubic metre per ton', 'picometre', 'amperes per volt meter', 
 'millipascal-second', 'Caribbean guilder', 'megajoule per kilogram', 'liter per kilogram', 'grams per cubic meter', 
 'Lebanese pound', 'gram per 100 gram of solvent', 'square versta', 'gram', 'tonne', 'week', 'joule per tesla', 'country', 
 'Geary–Khamis dollar', 'cases per 100000 person-years', 'European Currency Unit', 'Swiss franc', 'hryvnia', 'Swedish krona', 
 'crore', 'Thousands', 'gigabyte', 'torr', 'movie theater', 'academic term']

def serialize_result(res):
    """现在的方法比较日期有BUG，使用此函数转换后再比较  """
    if type(res) is neo4j.time.Date:
        return res.isoformat()
    elif type(res) is dict:
        return {k:serialize_result(v) for k,v in res.items()}
    elif type(res) is list:
        return [serialize_result(_) for _ in res]
    else:
        return str(res)


def exec_one_sample(sample, config, key):
    driver = GraphDatabase.driver(uri=config.neo4j_uri, auth=(config.neo4j_user, config.neo4j_passwd))

    # query = sample['pred'] if 'pred' in sample else sample['cypher']
    query = sample[key]
    # 不是很需要这个了
    # query = fix_illeal_relation(query)    # 对生成的语句进行后处理，In case `` is not generated
    if not syntax_check(query):
        return False, "syntax pre-check failed"
    ans = sample.get('answer', None)
    unit  = None                           # 答案中可能有单位
    if ans is not None:
        if query.strip()[-7:] == "type(p)" or query.strip()[-10:] == "type ( p )": #问的是关系
            if ans == "position played on team / speciality":
                ans = '_speciality'
            if " " in ans:
                ans = ans.replace(" ", "_")
            if "-" in ans or "(" in ans or "\xa0" in ans or "." in ans or "'" in ans:
                ans = f"`{ans}`"
        # elif any(u in ans for u in units):  # 答案中带有关系
        #     try:
        #         n, *unit_parts = ans.split(" ")
        #         ans = f'{float(n)}'
        #         unit = " ".join(unit_parts)
        #     except:
        #         pass 
        elif any(u in ans for u in units) and " " in ans:  # 答案中包含单位, fix_bug:有单位则须有空格
            if "1" in ans and ans.replace(" ", "").isnumeric():  # ISBN 等如 '0000 0000 4361 9306'
                pass 
            else:
                try:
                    n, *unit_parts = ans.split(" ")
                    unit = " ".join(unit_parts)
                    if unit in units:
                        ans = f'{float(n)}'
                except:
                    # print(f"execution answer units converting failed: {ans}, {[ u for u in units if u in ans]}")
                    pass 
    # 开始执行
    matched = None
    with driver.session(database='neo4j') as session:
        try:
            q = Query(query, timeout=config.timeout)
            res = session.run(q).data()

        except neo4j.exceptions.CypherSyntaxError as e:
            # print(query, " ||| ", e)
            res = "syntax: " + query 
            return False,  res
        except neo4j.exceptions.DatabaseError as e:
            return False, "db error incomplete: "+ query
        
        result_str = str(serialize_result(res))
        
        if len(res) == 0:
            if ans == 'no':
                matched = True 
            else:
                # return "no result: " + query, matched
                matched = False
        elif ans is None:
            # 没有提供答案，只返回查询结果
            return  None, res 
        else:
            # res  = str(res[0]) + "query: "+ query
            if ans in result_str:
                if unit is not None and unit not in result_str:
                    matched = False
                else:
                    matched = True
            elif ans == 'no' and ('False' in result_str or "None" in result_str):  # 值节点不存在该属性名，则对应null
                matched = True 
            elif ans == 'yes' and 'True' in result_str:
                matched = True 
            else:
                matched = False
    driver.close()
    # 返回查询结果，以及是否匹配成功（结果正确）
    return matched, result_str


def process_execute(queries, nproc=8, config=None, key='cypher'):
    
    assert config is not None, f"config is None, please set config first"

    Results = [] 
    if nproc == 1:
        print_string("sequential execution")
        for sample in tqdm(queries):
            Results.append(exec_one_sample(sample, config, key))
    else:
        with Pool(nproc) as pool:
            R = {}
            for sample in queries:
                future = pool.apply_async(exec_one_sample, (sample, config, key))
                R[future] = sample
            
            for future in tqdm(R):
                try:
                    res = future.get()
                except Exception as e:
                    print(e)
                    print(R[future])
                    print("+"*80)
                    res = False, f"Neo4jError: {R[future]['pred']}"
                Results.append(res)
    assert len(Results) == len(queries), f"{len(R)} != {len(queries)}"
    return Results


def cypher_exec_eval(path, nproc=16, key='pred', resave=False, post_process_fn=None, config=None):
    """ 
        path: 可以是字符串（读取文件）也可以是读取后的对象list 
        key: 每一个sample字典中, 预测的sparql语句的key
        nproc: 并行处理的进程数
        resave: bool or string  是否重新保存结果, 保存在原目录, 文件名前加上exec_
        post_process_fn: 对每一个sample进行后处理的函数，返回一个新的sample
    """
    data = load_json(path) if type(path) is str else path 
    # queries = [ s['generated'] for s in data]
    # Results = threads_execution(data, nthreads=32)
    if post_process_fn is None:
        print_string("post_process_fn is None, make sure `key` do not need to be processed")
    if post_process_fn is not None:
        data = [ post_process_fn(s) for s in data]
    Results = process_execute(data, nproc=nproc, config=config, key=key)
    
    is_right = [ 1 for match, res_str in Results if match is True ]
    acc = 100 * len(is_right)/ len(Results)
    print(f"{len(is_right)} of {len(Results)} is correct, {acc:.3f}")

    if resave:
        if type(resave) is not str:
            assert type(path) is str, "resave=True, path should be a string, or set resave=filepath"
        else:
            path = resave

        for sample, exec_res in zip(data, Results):
            sample['exec'] = exec_res
        print_string(f"resaving to {resave}")
        new_filename = "exec_" + os.path.basename(resave)
        new_path = os.path.join(os.path.dirname(resave), new_filename)
        save_json(data, new_path)

    return acc, Results



if __name__ == '__main__':

    config = Namespace(neo4j_uri="neo4j://127.0.0.1:28892", neo4j_user="neo4j", neo4j_passwd="kqa", timeout=10)
    pass 
