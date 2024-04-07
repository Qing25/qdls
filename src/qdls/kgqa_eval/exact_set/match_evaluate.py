import re


def match2where(cypher):
    cypher = cypher.replace(' ', '')
    pattern = re.compile(r'\([^)]*\)')
    matches = pattern.findall(cypher)
    wheres = []
    for match in matches:
        if '{' in match and '}' in match and ':' in match:
            wheres.append(match)
        elif '{' in match and '}' not in match and ':' in match:
            wheres.append(match + '}')
    match2where = []
    for where in wheres:
        tag = where.split(':')[1].split('{')[0]
        prop = where.split('{')[1].split(':')[0]
        value = where.split('{')[1].split(':')[1].split('}')[0]
        temp = tag + '.' + prop + ' == ' + value
        match2where.append(temp)
    return match2where


def special_match_find(cypher):
    pattern = re.compile(r'\{.*?\}')
    cypher = re.sub(pattern, '', cypher)
    return cypher


def find_matches(text):
    match_where = match2where(text)  # 取出match语句中的{name: '[s]'}特殊表示方式，并转为where条件
    text = special_match_find(text)  # 删除match语句中的{name: '[s]'}特殊表示方式
    text = re.sub(r'\s+', ' ', text)  # 去除多个连续空格
    text = text.lower().replace('match ', '')  # 去除match字符
    matches = re.findall(r':[^\)\]]+', text)
    # print(matches)
    matches = sorted([i.replace(' ', '')[1:] for i in matches])
    # print(matches)
    splits = {
        'match_tags': matches,
        'match_where': match_where
    }
    return splits


if __name__ == '__main__':
    # gold = 'MATCH (t:trade)<-[bt:belong_to]-(s:stock)-[hsd:has_stock_data]->(sd:stock_data)'
    # gold = 'MATCH (pof:public_offering_fund)'
    # gold = 'MATCH (s:stock)<-[ico:is_chairman_of]-(cm:chairman)'
    # gold = 'MATCH (s:stock)-[bt:belong_to]->(t:trade)'
    # gold = 'MATCH (sd:stock_data)<-[hsd:has_stock_data]-(s:stock)-[bt:belong_to]->(t:trade)'
    gold = "MATCH (t:trade)<-[bt:belong_to]-(s:stock{name: '[s]'})-[:has_stock_data]->(sd:stock_data{`date`: date('[d]')})"
    print(gold)
    splits = find_matches(gold)
    print(splits)
