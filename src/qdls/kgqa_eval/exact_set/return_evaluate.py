import os, sys 
current_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_directory)

import re
from utils import OPERATOR, FUNCTION, find_functions, find_calculates, find_tag_attrs, find_orderby


def find_temps(text):
    if not re.findall(r'[\w`]+\.[\w`]+', text):
        return ['temp']
    else:
        return []



def find_limit(text):
    match = re.findall(r'\blimit\s+(\d+)', text)
    if match:
        limit_value = [int(i) for i in match]
        # print(limit_value)
        return limit_value
    else:
        return []


# def find_orderby(text):
#     # print(text)
#     word_list = text.split(' ')
#     order_index = word_list.index('order')
#     try:
#         order_attr = '.'.join(word_list[order_index - 1].split('.')[1:])
#         # print('order_attr', order_attr)
#         # order_attr_var1 = word_list[order_index - 1]
#         # order_attr_var2 = word_list[order_index + 2]
#         try:
#             order_sort = word_list[order_index + 3]
#         except:
#             order_sort = 'asc'
#         orderby = (order_attr, order_sort)
#         return orderby
#     except:
#         try:
#             order_sort = word_list[order_index + 2]
#         except:
#             order_sort = 'asc'
#         orderby = (word_list[order_index + 2], order_sort)
#         return orderby


def has_english_characters(s):
    return bool(re.search(r'[a-zA-Z]', s))


def is_numeric_using_regex(s):
    return bool(re.match(r'^-?\d+(\.\d+)?$', s))




def find_returns(text):
    text = re.sub(r'\s+', ' ', text)  # 去除多个连续空格
    text = text.lower().replace('return ', '')  # 去除return字符
    text = re.sub(r'\s+as\s+\w+\s*', ' ', text)  # 去除自命名
    rag_attrs = []
    calculates = []
    functions = []
    orderbys = []
    temps = []
    limits = []

    # 首先找到limit
    if ' limit ' in text:
        match = re.search(r'\blimit\s+(\d+)\s*(.*)$', text)
        if match:
            limit = match.group(1).strip()
            limits.append(limit)
            # print(limit)
            text = re.sub(r'\s*limit\s+\d+\s*.*$', '', text)
        else:
            pass

    # print(text)
    sents = text.split(',')
    sents = [i.strip() for i in sents]
    # print(sents)

    for sent in sents:
        # print(sent)
        # 涉及orderby
        if 'order by' in sent:
            orderby = find_orderby(sent)
            orderbys.append(orderby)
            # print(orderby)
            continue
        # 涉及计算
        if any(substring in sent for substring in [' ' + i + '' for i in OPERATOR]):
            # print(0)
            calculate = find_calculates(sent)
            calculates.append(calculate)
            continue
        if any(substring in sent for substring in [i + '(' for i in FUNCTION]):
            function = find_functions(sent)
            functions.append(function)
            continue
        # 定义temp
        if not re.findall(r'[\w`]+\.[\w`]+', text):
            temps.append('temp')
            continue
        rag_attr = find_tag_attrs(sent)
        rag_attrs.append(rag_attr)
        # print(rag_attr)
    splits = {
        'return_rag_attrs': sorted(rag_attrs),
        'return_calculates': sorted(calculates),
        'return_functions': sorted(functions),
        'return_orderby': sorted(orderbys),
        'return_temps': sorted(temps),
        'return_limit': sorted(limits),
    }
    return splits


def return_evaluate(gold, pre):
    # math.isclose(10.00, 10)
    splits_gold = find_returns(gold)
    splits_pre = find_returns(pre)
    for key in splits_gold.keys():
        if splits_gold[key] != splits_pre[key]:
            return False
    return True


if __name__ == '__main__':
    gold = 'RETURN sd.stock_data.`date`, sd.stock_data.closing_price, h.position_ratio'
    gold = 'RETURN pof.public_offering_fund.name , pof.public_offering_fund.net_worth, pof.public_offering_fund.management_fee LIMIT 50'
    # gold = 'RETURN sd.stock_data.closing_price , sd.stock_data.opening_price , avg(td.trade_data.closing_price) AS avg_closing_price'
    gold = 'RETURN distinct(sd.stock_data.closing_price), sd.stock_data.maximum_price as SSDCP ORDER BY SSDCP ASC LIMIT 1'
    # gold = 'RETURN (sd.stock_data.closing_price - p1) / p1 as sum_range'
    # gold = 'RETURN pof.public_offering_fund.management_fee > 10.00'
    # gold = 'RETURN temp'

    pre = 'RETURN sd.stock_data.`date`, h.position_ratio, sd.stock_data.closing_price'
    pre = 'RETURN pof.public_offering_fund.name , pof.public_offering_fund.net_worth, pof.public_offering_fund.management_fee LIMIT 50'
    # pre = 'RETURN sd.stock_data.closing_price , avg(dgsdf.trade_data.closing_price) , sd.stock_data.opening_price'
    pre = 'RETURN sd.stock_data.closing_price, ORDER BY sd.stock_data.maximum_price   LIMIT 1'
    # pre = 'RETURN (sd.stock_data.closing_price - ddd) / ddd as s'
    # pre = 'RETURN avg(pof.public_offering_fund.management_fee) > 10'
    # pre = 'RETURN temp1'

    splits = find_returns(gold)
    print(splits)
    splits = find_returns(pre)
    print(splits)

    result = return_evaluate(gold, pre)
    print(result)
