import neo4j
from neo4j.time import DateTime
from neo4j import GraphDatabase, Query
import neo4j.exceptions

from tqdm import tqdm
import datetime
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor

from argparse import Namespace

import signal
class TimeoutError(Exception):
    pass

# 如果内置的timeout不好用，可以使用这个
def timeout(seconds):
    def decorator(func):
        def handler(signum, frame):
            raise TimeoutError("Function call timed out")

        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(seconds)

            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)  # 重置信号

            return result

        return wrapper

    return decorator    

def query_neo4j(query, driver, timeout=10):
    with driver.session(database='neo4j') as session:
        q = Query(query, timeout=timeout)
        res = session.run(q).data()
        return res 

def batch_query(queries, uri, user, passwd, nthreads=8, timeout=10):
    """并行执行查询，需要传入neo4j参数，用于给每个线程创建连接

    Args:
        queries: a list of query strings
        uri: _description_
        user: _description_
        passwd: _description_
        nthreads: _description_. Defaults to 8.
        timeout: _description_. Defaults to 10.

    Returns:
        a list of results
    """
    n = len(queries)
    config = Namespace(
        neo4j_uri=uri, neo4j_user=user, neo4j_passwd=passwd, timeout=timeout
    )
    if n == 1:
        return [exec_single_query(queries[0], config)]
    else:
        return threads_execution(queries, config, nthreads if n > nthreads else n)


def exec_single_query(query, config):
    """ 
        neo4j.time.DateTime 无法序列化，因此转换为python的datetime.datetime
        
        neo4j 的 driver 是线程安全的，通过传入的config参数在每个线程创建一个driver连接
            ref: https://community.neo4j.com/t5/drivers-stacks/python-generator-already-executing/m-p/40421

        return:
            Exception 或 list
    """
    driver = GraphDatabase.driver(uri=config.neo4j_uri, auth=(config.neo4j_user, config.neo4j_passwd))
    with driver.session(database="neo4j") as session:
        q = Query(query, timeout=config.timeout)
        res = session.run(q).data()
        return res

def threads_execution(queries, nthreads=16, config=None):
    """
    Args:
        queries: a list of query strings
        config: 连接参数

    Returns:
        执行结果
    """
    if config is None:
        raise Exception(f'config is None!!! expected: argparse.Namespace(neo4j_uri="neo4j://ip:port", neo4j_user="neo4j", neo4j_passwd="xxx", timeout=10)')

    
    thread_pool = ThreadPoolExecutor(nthreads) 
    R = []
    result_iter = thread_pool.map(exec_single_query, queries, [config]*len(queries))
    for res in tqdm(result_iter, total=len(queries)):
        R.append(res)
    assert len(R) == len(queries), f"{len(R)} != {len(queries)}"
    return R 


def process_execute(queries, nproc=8, config=None):
    """多进程执行查询，速度比多线程快(3000,32测试快3倍)

    Args:
        queries: List, 要处理的数据,
        nproc: 进程数目. Defaults to 8.
        config: Neo4j连接配置. Defaults to None.

    Raises:
        Exception: config 

    Returns:
        List, 执行结果
    """
    if config is None:
        raise Exception(f'config is None!!! expected: argparse.Namespace(neo4j_uri="neo4j://ip:port", neo4j_user="neo4j", neo4j_passwd="xxx", timeout=10)')

    Results = [] 
    with Pool(nproc) as pool:
        R = {}
        for sample in queries:
            future = pool.apply_async(exec_single_query, (sample, config))
            R[future] = sample
        
        for future in tqdm(R):
            res = future.get()
            Results.append(res)
    assert len(Results) == len(queries), f"{len(R)} != {len(queries)}"
    return Results

    
def neo4j_data_serializable(obj):
    if type(obj) is DateTime:
        dt = obj
        return datetime.datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute,
                        int(dt.second), int(dt.second * 1000000 % 1000000))
    elif type(obj) is neo4j.time.Date:
        return obj.isoformat()
    elif type(obj) is dict:
        return { k : neo4j_data_serializable(v) for k,v in obj.items()}
    elif type(obj) is list:
        return [ neo4j_data_serializable(_) for _ in obj]
    else:
        return obj  
    
