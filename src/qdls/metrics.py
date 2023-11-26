import os
import re
import collections
import string
import numpy as np
try:
    from sklearn.metrics import classification_report
except Exception as e:
    print("Try ` pip install scikit-learn `")
    pass 
from tqdm import tqdm 

from .data import load_json

def classification_matrix(gold_ids, pred_ids, label_names, save_file):
    """
        gold_ids = [0,1,2]
        pred_ids = [1,1,2]
        label_names = ['a','b','c']
    """
    report = classification_report(gold_ids, pred_ids, labels=list(range(len(label_names))), target_names=label_names)
    with open(save_file, 'w', encoding='utf8') as f:
        f.write(report)


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    if s == '':
        return s
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def prec_at_k(output, target, top_k=(1,)):
    """Computes the precision@k for the specified values of k
        Taken from  https://github.com/lyakaap/pytorch-template
    """
    max_k = max(top_k)
    batch_size = target.size(0)

    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    if len(res) == 1:
        res = res[0]

    return res



if __name__ == '__main__':
    f1 = compute_f1("你好？hh", "NO go away 不好")
    f1 = compute_f1("你好？", "你好")
    print(f1)
    a = [1,1,1,0,0]
    b = [1,1,0,0,0]

    gold_ids = [0,1,2]
    pred_ids = [1,1,2]
    label_names = ['a','b','c']
    classification_matrix(gold_ids, pred_ids, label_names, 'tmp.txt')
