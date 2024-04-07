import os, sys 
current_directory = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_directory)
import re
from utils import OPERATOR, UNDERAND, find_calculates, remove_quotes, add_spaces_around_operators


def find_underand(text):
    text = remove_quotes(text)
    # print('text', text)
    for i in UNDERAND:
        if ' ' + i + '' in text:
            contents = []
            sents = text.split(i)
            sents = [i.strip() for i in sents]
            # print('sents', sents)
            for sent in sents:
                content = find_calculates(sent)
                contents.append(content)
            return (i, sorted(contents))
    return ''


def find_wheres(text):
    calculates = []
    temps = []
    # text = add_spaces_around_operators(text)
    text = re.sub(r'\s+', ' ', text)  # 去除多个连续空格
    text = text.lower().replace('where ', '')  # 去除where字符
    # print(text)
    sents = text.split('and')
    sents = [i.strip() for i in sents]
    # print(sents)
    for sent in sents:
        # print('sent', sent)
        if any(substring in sent for substring in [' ' + i + '' for i in UNDERAND]):
            calculate = find_underand(sent)
            # print(calculate)
            calculates.append(calculate)
            continue
        if any(substring in sent for substring in [' ' + i + '' for i in OPERATOR]):
            # print(sent)
            calculate = find_calculates(sent)
            # print(calculate)
            calculates.append(calculate)
            # print('calculate', calculate)
            continue
        if not re.findall(r'[\w`]+\.[\w`]+', text):
            temps.append('temp')
            continue

    splits = {
        'where_calculates': sorted(calculates, key=lambda x: x[0] if isinstance(x, tuple) else x),
        'where_temps': sorted(temps)
    }
    return splits


def where_evaluate(gold, pre):
    splits_gold = find_wheres(gold)
    splits_pre = find_wheres(pre)
    for key in splits_gold.keys():
        if splits_gold[key] != splits_pre[key]:
            return False
    return True


if __name__ == '__main__':
    gold = "WHERE s.stock.name == '[s]'"
    gold = "WHERE s.stock.name == '天力锂能' AND sd.stock_data.`date` == date('2023-8-31')"
    splits = find_wheres(gold)
    print(splits)
