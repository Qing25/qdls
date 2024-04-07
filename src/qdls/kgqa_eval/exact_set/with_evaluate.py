import os, sys 
current_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_directory)
import re
from utils import OPERATOR, FUNCTION, find_functions, find_calculates, find_tag_attrs, find_orderby


def find_withs(text):
    rag_attrs = []
    calculates = []
    functions = []
    temps = []
    orderbys = []
    limits = []
    text = re.sub(r'\s+', ' ', text)  # 去除多个连续空格
    text = text.lower().replace('with ', '')  # 去除with字符
    text = re.sub(r'\s+as\s+\w+\s*', ' ', text)  # 去除自命名
    if ' limit ' in text:
        match = re.search(r'\blimit\s+(\d+)\s*(.*)$', text)
        if match:
            limit = match.group(1).strip()
            limits.append(limit)
            # print(limit)
            text = re.sub(r'\s*limit\s+\d+\s*.*$', '', text)
        else:
            pass
    sents = text.split(', ')
    sents = [i.strip() for i in sents]
    # print(sents)
    for sent in sents:
        # print(sent)
        if 'order by' in sent and "row_number()" not in sent:
            orderby = find_orderby(sent)
            orderbys.append(orderby)
            continue
        elif any(substring in sent for substring in [i + '(' for i in FUNCTION]):
            function = find_functions(sent)
            functions.append(function)
            continue
        elif any(substring in sent for substring in [' ' + i + '' for i in OPERATOR]):
            # print('sent', sent)
            calculate = find_calculates(sent)
            calculates.append(calculate)
            continue
        # 定义temp
        if not re.findall(r'[\w`]+\.[\w`]+', text):
            temps.append('temp')
            continue
        rag_attr = find_tag_attrs(sent)
        rag_attrs.append(rag_attr)

    splits = {
        'with_rag_attrs': sorted(rag_attrs),
        'with_calculates': sorted(calculates),
        'with_functions': sorted(functions),
        'with_temps': sorted(temps),
        'with_orderbys': sorted(orderbys)
    }
    return splits




if __name__ == '__main__':
    gold = "WITH c.chairman.sex AS temp1, WITH c.chairman.name AS temp2, avg(td.trade_data.closing_price) AS avg_closing_price, c.chairman.name - 1 AS temp3"
    gold = "WITH s.stock.name AS sname, sd.stock_data.closing_price AS SSDR ORDER BY SSDR DESC LIMIT 1"

    splits = find_withs(gold)
    print(splits)




