
import re 


def postprocess_sparql_tokens(sparql_query):
    sparql_query = re.sub(r'(\?[a-zA-Z0-9_]+)', r' \1 ', sparql_query)
    # 去除多余的空格
    sparql_query = re.sub(r'\s+', ' ', sparql_query)
    return sparql_query


def detect_query_language(query_string):
    # 判断是否包含SPARQL关键词
    sparql_keywords = ['SELECT', 'DISTINCT', 'WHERE', 'FILTER', 'OPTIONAL', 'ORDER BY', 'LIMIT', 'ASK']
    has_sparql_keywords = any(keyword in query_string.upper() for keyword in sparql_keywords)

    # 判断是否包含Cypher关键词
    cypher_keywords = ['MATCH', 'OPTIONAL MATCH', 'WHERE', 'RETURN', 'ORDER BY', 'LIMIT']
    has_cypher_keywords = any(keyword in query_string.upper() for keyword in cypher_keywords)

    if has_sparql_keywords and not has_cypher_keywords:
        return 'sparql'
    elif has_cypher_keywords and not has_sparql_keywords:
        return 'cypher'
    else:
        raise Exception(f"unknown query language `{query_string}`")

