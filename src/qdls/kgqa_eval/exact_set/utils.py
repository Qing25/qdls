import re
from datetime import datetime

OPERATOR = ['==', '!=', '<>', '>', '>=', '<', '<=', 'is null', 'is not null', 'is empty', 'is not empty',
            '+', '-', '*', '/', '%', '!',
            'and', 'or', 'not', 'xor',
            'union', 'union all', 'intersect', 'minus',
            'contains', 'in', 'starts with', 'ends with', '', '']
FUNCTION = ['abs', 'floor', 'ceil', 'round', 'sqrt', 'cbrt', 'hypot', 'pow', 'exp', 'exp2', 'log', 'log2', 'log10',
            'sin', 'asin', 'cos', 'acos', 'tan', 'atan', 'rand', 'rand32', 'rand64', 'size', 'range', 'sign',
            'avg', 'count', 'max', 'min', 'collect', 'std', 'sum',
            'date',
            'distinct']
UNDERAND = ['or', 'not', 'xor']
SPLIT = ['match', 'where', 'with', 'return']


def has_english_characters(s):
    return bool(re.search(r'[a-zA-Z]', s))


def is_numeric_using_regex(s):
    return bool(re.match(r'^-?\d+(\.\d+)?$', s))


# def is_valid_date_string(date_str):
#     print('date_str', date_str)
#     date_str = remove_quotes(date_str)
#     date_object = datetime.strptime(date_str, "%Y-%m-%d")
#     formatted_date_string = date_object.strftime("%Y-%m-%d")
#     # pattern = re.compile(r'\b\d{4}-\d{2}-\d{2}\b')
#     match = pattern.match(date_str)
#     return bool(match)


def add_spaces_around_operators(sentence):
    for op in ['<=', '>=', '==', '<', '>', '+']:
        pattern = re.compile(rf'(?<=[^\s]){re.escape(op)}(?=[^\s])')
        sentence = pattern.sub(f' {op} ', sentence)
    return sentence


def is_datetime_format(input_string, date_formats=["%Y-%m-%d", "%Y", "%Y-%m"]):
    input_string = remove_quotes(input_string)
    for date_format in date_formats:
        try:
            datetime_object = datetime.strptime(input_string, date_format)
            date_str = datetime_object.strftime("%Y-%m-%d")
            return date_str
        except ValueError:
            continue
    return False


def remove_quotes(input_string):
    if len(input_string) >= 2 and ((input_string[0] == "'" and input_string[-1] == "'") or
                                   (input_string[0] == "(" and input_string[-1] == ")")):
        return input_string[1:-1]
    else:
        return input_string


def find_calculates(text):
    sents = text.split(' ')
    # print('sents', sents)
    temp = []
    for sent in sents:
        if sent in OPERATOR:
            temp.append(sent)
            continue
        if is_numeric_using_regex(sent):
            temp.append("{:.2f}".format(float(sent)))
            continue
        if any(substring in sent for substring in [i + '(' for i in FUNCTION]):
            f = find_functions(sent)
            temp.append(f[0] + '(' + f[1] + ')')
            continue
        if '.' in sent:
            temp.append('.'.join(sent.split('.')[1:]))
            continue
        # match = re.search(r'\b(\w+)\.(\w+)\.(\w+)\b', sent)
        # if match:
        #     print(1)
        #     temp.append('.'.join([match.group(2), match.group(3)]))
        #     continue
        # match = re.search(r'(\w+)\.(\w+)', sent)
        # if match:
        #     print(2)
        #     temp.append(match.group(2))
        #     continue
        if sent not in OPERATOR and has_english_characters(sent) and '[' not in sent:
            temp.append('[MASK]')
            continue
        temp.append(sent)
    calculate = ' '.join(temp)
    return calculate


def find_functions(text):
    # print(text)
    for i in FUNCTION:
        if i + '(' in text:
            # print(text)
            pattern = r'\b(?:' + i + r')\s*\((.*?)\)'
            match = re.search(pattern, text)
            if match:
                content = match.group(1)
                if is_numeric_using_regex(content):
                    return (i, content)
                if is_datetime_format(content):
                    return (i, is_datetime_format(content))
                content = '.'.join(content.split('.')[1:])
                # print('content', content)
                return (i, content)
            else:
                # print(text)
                return (i, text.replace(i, '')[1:-1])
    return ''


def find_tag_attrs(text):
    tag_attrs = '.'.join(text.split('.')[1:])
    return tag_attrs


def read_dict(dict):
    for key, value in dict.items():
        print(key, ': ', value)


def find_orderby(text):
    # print(text)
    word_list = text.split(' ')
    order_index = word_list.index('order')
    try:
        order_attr = '.'.join(word_list[order_index - 1].split('.')[1:])
        # print('order_attr', order_attr)
        # order_attr_var1 = word_list[order_index - 1]
        # order_attr_var2 = word_list[order_index + 2]
        try:
            order_sort = word_list[order_index + 3]
        except:
            order_sort = 'asc'
        orderby = (order_attr, order_sort)
        return orderby
    except:
        try:
            order_sort = word_list[order_index + 2]
        except:
            order_sort = 'asc'
        orderby = (word_list[order_index + 2], order_sort)
        return orderby


def special_sort(l):
    return sorted(l, key=lambda x: x[0] if isinstance(x, tuple) else x)


if __name__ == '__main__':
    content = '2023'
    s = is_datetime_format(content)
    print(s)
