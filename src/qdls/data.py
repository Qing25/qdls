import os
try:
    import torch
except:
    pass
import json
import yaml
import math
import time
import hashlib
from collections import defaultdict
import numpy as np
import pandas as pd

from tqdm import tqdm
import pickle

def timeit(f):
    """ 函数装饰器，定义函数时调用 """
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        print('func:%r args:[%r, %r] took: %2.4f sec' % (f.__name__, args, kw, te-ts))
        return result

    return timed

def save_json(R, path, **kwargs):
    """ Obj, path """
    with open(path, 'w', encoding='utf8') as f:
        json.dump(R, f, indent=2, ensure_ascii=False, **kwargs)
    print(f"{path} saved with {len(R)} samples!")

def load_json(path):
    with open(path, 'r', encoding='utf8') as f:
        obj = json.load(f)
    print(f"{path} loaded with {len(obj)} samples!")
    return obj

def load_jsonl(path):
    R = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            R.append(json.loads(line))
    print(f"{path} loaded with {len(R)} samples!")
    return R


def load_yaml(path):
    with open(path) as fin:
        return yaml.safe_load(fin)


def save_pickle(R, path):
    """ save some objects that json not support, like DateTime """
    with open(path, 'wb') as f:
        pickle.dump(R, f)
    print(f"{path} saved with {len(R)} samples!")
    
def load_pickle(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    print(f"{path} loaded with {len(obj)} samples!")
    return obj


class BaseData:    
    def cpu(self):
        for k, v in self.__dict__.items():
            if type(v) is torch.Tensor:
                setattr(self, k, v.cpu())
        return self
    
    def to(self, device):
        for k,v in self.__dict__.items():
            if type(v) is torch.Tensor:
                setattr(self, k, v.to(device))
        return self
    
    def __getitem__(self,index):
        return self.__dict__[index]
    
    def todict(self):
        return self.__dict__
    
    def tolist(self):
        return [ v for k,v in self.__dict__.items()]

    @property
    def size_info(self):
        size = { 
            k: v.size() if type(v) is torch.Tensor else 
            (len(v) if v is not None else None)
            for k,v in self.__dict__.items()
        }
        return size
    


def span_decode(start_logits, end_logits, cls_logits=None, max_a_len=512, samples=None, offset_mappings=None, use_cls=True, no_answer=""):
    """

    Args:
        start_logits  (torch.Tensor) :  (bsz,seqlen)
        end_logits (torch.Tensor) :   (bsz,seqlen)
        cls_logits (torch.Tensor) :  (bsz, num_classes)
        max_a_len ( int ): 限制答案文本的最大长度
        samples (MRCSample ): 该条数据的所有信息
        offset_mappings ([type]): tokenizer返回的
        use_cls (bool, optional): 是否使用预测的有无答案的概率，False则一定会返回预测的span文本. Defaults to True.
        no_answer (str, optional): Squad和DuReader要求的无答案形式不同，. Defaults to "".

    Returns:
        Dict : {qid: pred_text, ...}
    """
    se_sum = end_logits[:,None,:] + start_logits[:,:,None]
    # 限制值的范围是s<e, 最大长度为 max_a_len        
    mask = torch.tril(torch.triu(torch.ones_like(se_sum), 0), max_a_len)   
    r = (mask * se_sum).masked_fill_((1-mask).bool(), float('-inf'))    # score值全是负的，导致0 > score，选出来s>e了
    start_max, end_max = r.max(2)[0].max(1)[1], r.max(1)[0].max(1)[1]
    answerable = cls_logits.argmax(-1) if cls_logits is not None else torch.zeros(start_logits.size(0),)
    R = {}
    for s, e, a, sample, mapping in zip(start_max, end_max, answerable, samples, offset_mappings):
        if a == 1 and use_cls:
            R[sample.qid] = no_answer
        else:
            s_char, e_char = mapping[s][0], mapping[e][-1]
            pred_text = sample.context[s_char:e_char]
            pred_text = no_answer if pred_text == "" else pred_text
            R[sample.qid] = pred_text
    return R


def nbest_span_decode(start_scores, end_scores, batch, max_a_len=512, nbest=5):
    # se_sum = end_scores[:,None,:] + start_scores[:,:,None]
    # r = torch.tril(torch.triu(torch.ones_like(se_sum), 0), max_a_len) * se_sum   # (0,0) is necessary !!!!!!!!!!!!
    se_sum = end_scores[:,None,:] + start_scores[:,:,None]
    # 限制值的范围是s<e, 最大长度为 max_a_len        
    mask = torch.tril(torch.triu(torch.ones_like(se_sum), 0), max_a_len)   
    r = (mask * se_sum).masked_fill_((1-mask).bool(), float('-inf'))

    pred_answers = defaultdict(dict)
    for mat, c, m, qid, a in zip(r, batch['context'], batch['offset_mapping'], batch['qids'], batch['gold']):
        pred_answers[qid]['gold'] = a
        pred_answers[qid]['nbest'] = []
        v, i = torch.topk(mat.flatten(), int(nbest))
        v, i = v.cpu(), i.cpu()
        for (s,e), p in zip(np.array(np.unravel_index(i.numpy(), mat.shape)).T, v):
            s_char, e_char =  m[s][0], m[e][-1]
            pred_text = c[s_char:e_char]
            pred_text = "no answer" if pred_text == "" else pred_text
            pred_answers[qid]['nbest'].append({"prob":f"{p.item():.3f}", "text": pred_text})
    return pred_answers

def gen_uid(string):
    """
    根据字符串生成对应的Md5码, 用与给数据生成ID
    Args:
        string (str): 需要是能标识该对象的独特字符串

    Returns:
        str: 十六进制字符串
    """
    return hashlib.md5(string.encode("utf8")).hexdigest()


def sequence_padding(inputs, length=None, force=False, padding=0, padding_side='right'):
    """Numpy函数，将序列padding到同一长度
    Args:
        inputs (list of lists): [description]
        length (int, optional): 指定最大长度. Defaults to None.
        force (bool, optional): 如果为True则一定Padding到length长度，否则padding到最长的list的长度. Defaults to False.
        padding (int, optional): Padding的值. Defaults to 0.
        padding_side: GPT2做预测的时候，padding to left

    Returns:
        np.array: padding后的序列
    """
    _length = max([len(x) for x in inputs])
    if length is None:
        length = _length
    else:
        length = min(length, _length) if not force else length

    outputs = np.array([
        np.concatenate(
            [x, [padding] * (length - len(x))] if padding_side == 'right' else [[padding] * (length - len(x)), x] 
        ) if len(x) < length else x[:length]
        for x in inputs
    ])

    return outputs 


def dict2list(td):
    """
        输入的是tokenizer一次处理多条数据的结果， 返回list, 其中每个dict是每条数据的结果
    Args:
        td (dict): {'input_ids': [[...],[...],...], 'attention_mask':...}

    Returns:
        [list]: [{'input_ids':[], 'attention_mask':[]},{},...]
    """
    df = pd.DataFrame.from_dict(td, orient='index').T
    return df.to_dict(orient='records')



class SegmentUtility():
    def __init__(self, seg_func) -> None:
        """
        Args:
            seg_func (function):  eg: lambda s: seg.cut(s) || jieba.lcut(s)
        """
        self.seg = seg_func

    def seg_start_end(self, sentence):
        """
            返回句子分词的结果，以及每个词对应的字符start end
        """
        words = self.seg(sentence)
        s,e = 0,0
        R = [[(s:=e, e:=s+len(word))] for word in words]
        return words, sum(R, [])

    @staticmethod
    def generate_seg_ids(tokenizer, input_ids, offset_mapping, se):
        """对于一个句子的input_ids， 根据该句子的中文分词结果，对于每个词的tokens给一个id
        从[CLS]是0开始，第一个词是1,...
        Args:
            input_ids (list): Tokenizer 返回的input_ids
            offset_mapping (list): Tokenizer 返回的 offset_mapping
            se (list): [(0,2),(2,6),...,(s,e)]

        Returns:
            list : [0,1,1,2,2,2,2,...]
        """
        R, ptr, word_ptr = [], 0, 0
        for token_id, (s,e) in zip(input_ids, offset_mapping):
            if (s,e) == (0,0) and token_id != tokenizer.pad_token_id:
                R.append(ptr)
                ptr += 1
            elif token_id == 0:  # padding
                R.append(ptr)
            elif se[word_ptr][0] <= s and e < se[word_ptr][1]:
                R.append(ptr)
            elif e == se[word_ptr][1]:
                R.append(ptr)
                word_ptr += 1
                ptr += 1
        assert len(R) == len(input_ids), f"{(len(R), R)},{(len(input_ids), input_ids)}"
        return R

    def prepare_inputs(self, tokenizer, first, second=None):
        """[summary]

        Args:
            tokenizer ([type]): [description]
            first (str): 第一个句子
            second (str, optional): 第二个句子 Defaults to None.

        Returns:
            [type]: [description]
        """
        words_1 = self.seg(first)
        s,e = 0,0
        se_1 = [[(s:=e, e:=s+len(word))] for word in words_1]
        
        if second is not None:
            words_2 = self.seg(second)
            s,e = 0,0
            se_2 = [[(s:=e, e:=s+len(word))] for word in words_2]
            
        td = tokenizer(first, second, return_offsets_mapping=True)
        td['se'] =  sum(se_1 if second is None else se_1 + se_2, [])
        return td

    def prepare_batch_inputs(self, tokenizer, first, second=None):
        """[summary]

        Args:
            tokenizer ([type]): [description]
            first (list):  字符串的列表
            second (list, optional): 字符串的list. Defaults to None.

        Returns:
            dict: 分词器分词后的结果，增加了words = [], se = []
        """
        td = tokenizer(first, second, return_offsets_mapping=True, padding=True)
        SE = []
        Seg = []
        if second is not None:
            for ids, mapping, a, b in zip(td.input_ids, td.offset_mapping, first, second):
                words_1 = self.seg(a)
                s,e = 0,0
                se_1 = [[(s:=e, e:=s+len(word))] for word in words_1]

                words_2 = self.seg(b)
                s,e = 0,0
                se_2 = [[(s:=e, e:=s+len(word))] for word in words_2]

                se = sum(se_1 + se_2, [])
                SE.append(se)
                Seg.append((words_1, words_2))
        else:
            for ids, mapping, a in zip(td.input_ids, td.offset_mapping, first):
                words_1 = self.seg(a)
                s,e = 0,0
                se_1 = [[(s:=e, e:=s+len(word))] for word in words_1]
                se = sum(se_1, [])
                SE.append(se)
                Seg.append(words_1)
        td.se = SE
        td.words = Seg
        return td

######################
#   IO  functions 


def recursive_file_find(root):
    """ return all file contained in dir 'root' and it's subdirs
    """
    R = []
    for file in os.listdir(root):
        abs_path = os.path.join(root, file)
        if os.path.isfile(abs_path):
            R.append(abs_path)
        else:
            R.extend(recursive_file_find(abs_path))
    R = sorted(R)
    return R


if __name__ == '__main__':
    R = recursive_file_find("..")
    print(f"{len(R)} files found! ")

    
   
