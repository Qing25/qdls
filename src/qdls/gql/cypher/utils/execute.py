from neo4j.time import DateTime
from neo4j import GraphDatabase, Query
import neo4j.exceptions

from tqdm import tqdm
import datetime
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor

from argparse import Namespace
    

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
        try:
            q = Query(query, timeout=config.timeout)
            res = session.run(q).data()
        except Exception as e:
            return e
        return res

def threads_execution(queries, config, nthreads):
    """

    Args:
        queries: a list of query strings
        config: 连接参数

    Returns:
        执行结果
    """
    
    thread_pool = ThreadPoolExecutor(nthreads) 
    R = []
    for res in thread_pool.map(exec_single_query, queries, [config]*len(queries)):
        R.append(res)
    assert len(R) == len(queries), f"{len(R)} != {len(queries)}"
    return R 

    
def make_neo4j_date_serializale(obj):
    if type(obj) is DateTime:
        dt = obj
        return datetime.datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute,
                        int(dt.second), int(dt.second * 1000000 % 1000000))
    elif type(obj) is dict:
        return { k : make_neo4j_date_serializale(v) for k,v in obj.items()}
    elif type(obj) is list:
        return [ make_neo4j_date_serializale(_) for _ in obj]
    else:
        return obj  
    

