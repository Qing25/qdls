# -*- coding: utf-8 -*-
# @File    :   pretrains.py
# @Time    :   2021/11/02 19:54:48
# @Author  :   Qing 
# @Email   :   sqzhao@stu.ecnu.edu.cn
"""
    本模块使用方法：
        执行saved_pretrains() 查看本地已经保存的模型
        想要的模型没有保存，可以通过models[index].save()保存到本地
"""
import os
import time
from dataclasses import dataclass
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, AutoConfig
from transformers import AutoModelWithLMHead, AutoModelForSeq2SeqLM
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForTokenClassification
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertForQuestionAnswering, BertForPreTraining
from transformers import T5ForConditionalGeneration, MT5ForConditionalGeneration, PegasusForConditionalGeneration, PegasusTokenizer, T5Tokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import RoFormerPreTrainedModel, RoFormerTokenizer, RoFormerForMaskedLM
from transformers import ElectraModel, ElectraTokenizer, ElectraForPreTraining
from transformers import XLNetLMHeadModel 
from transformers import LongformerModel
from transformers import BartForConditionalGeneration, BartTokenizer


ROOT = "/pretrains/pt"

@dataclass
class ModelCard:
    web_dirname : str                        # hfl/bert-base-wwm
    model : object = AutoModel
    tokenizer : object = AutoTokenizer
    save_dir : str = None                    # example: /pretrians/pt/hfl-bert-base-wwm

    @property
    def local_dirname(self):
        return self.web_dirname.replace("/", "-", 1)     # hfl-bert-base-wwm
    
    @property
    def web_url(self):
        return "https://huggingface.co/"+self.web_dirname

    @property
    def local_url(self):
        if self.save_dir is None:
            return os.path.join(ROOT, self.local_dirname)
        else:
            return self.save_dir

    def save(self):
        t1 = time.time()
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t1)))
        print("="*20, f"  start saving {self.web_url} !  ", "="*20)
        config = AutoConfig.from_pretrained(self.web_dirname)
        if self.model != AutoModel:
            self.m = self.model.from_pretrained(self.web_dirname)
        else:
            self.m = self.model.from_config(config)
        self.t = self.tokenizer.from_pretrained(self.web_dirname)
        # self.t = self.tokenizer(config)
        t2 = time.time()
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t2)), f"---- Downloading finished in {(t2-t1):.2f} seconds.")
        self.m.save_pretrained(self.local_url)
        self.t.save_pretrained(self.local_url)
        print(f"saved to {self.local_url}; please check if warning info exists \n "
                f"Or you'd better re-save the model with correct model head ! {self.web_url}")

    def is_saved(self):
        return os.path.exists(self.local_url)

models = [
    
    ModelCard('hfl/chinese-roberta-wwm-ext-large', model=BertForPreTraining),
    ModelCard('hfl/chinese-bert-wwm-ext', model=BertForPreTraining),
    ModelCard('hfl/chinese-bert-wwm', model=BertForPreTraining),
    ModelCard('hfl/chinese-macbert-base', model=BertForPreTraining),
    ModelCard('hfl/chinese-macbert-large', model=BertForPreTraining),
    ModelCard('hfl/chinese-electra-180g-large-discriminator', model=ElectraForPreTraining, tokenizer=ElectraTokenizer),
    ModelCard('hfl/chinese-legal-electra-large-discriminator', model=ElectraForPreTraining, tokenizer=ElectraTokenizer),
    ModelCard('hfl/chinese-xlnet-mid', model=XLNetLMHeadModel),
    ModelCard('hfl/chinese-xlnet-base', model=XLNetLMHeadModel),
    ModelCard('hfl/rbt3', model=BertForPreTraining),           # re-trained 3-layer RoBERTa-wwm-ext model
    # QA
    ModelCard('luhua/chinese_pretrain_mrc_roberta_wwm_ext_large', model=BertForQuestionAnswering, tokenizer=BertTokenizer),
    ModelCard('luhua/chinese_pretrain_mrc_macbert_large', model=BertForQuestionAnswering, tokenizer=BertTokenizer),
    ModelCard('uer/roberta-base-chinese-extractive-qa', model=BertForQuestionAnswering, tokenizer=BertTokenizer),
    # ModelCard('mrm8488/spanbert-large-finetuned-squadv2', model=BertForQuestionAnswering, tokenizer=BertTokenizer),
    ModelCard('schen/longformer-chinese-base-4096', model=LongformerModel, tokenizer=BertTokenizer),
    # https://github.com/CLUEbenchmark/CLUEPretrainedModels
    # ModelCard('clue/roberta_chinese_clue_tiny', model=BertModel),
    # ModelCard('clue/roberta_chinese_pair_tiny', model=BertModel, tokenizer=BertTokenizer),
    # ModelCard('clue/roberta_chinese_3L768_clue_tiny', model=BertModel),
    # ModelCard('clue/roberta_chinese_3L312_clue_tiny' , model=BertModel),
    # ModelCard('clue/roberta_chinese_clue_large' , model=BertModel),
    # ModelCard('clue/roberta_chinese_pair_large', model=BertModel),
    # ModelCard('clue/xlnet_chinese_large', model=XLNetLMHeadModel),
    # https://github.com/renmada/t5-pegasus-pytorch
    ModelCard('imxly/t5-pegasus',  model=T5ForConditionalGeneration, tokenizer=BertTokenizer),
    ModelCard('imxly/t5-pegasus-small',  model=T5ForConditionalGeneration, tokenizer=BertTokenizer),
    ModelCard('algolet/mt5-base-chinese-qg',  model=MT5ForConditionalGeneration, tokenizer=T5Tokenizer),
    ModelCard('IDEA-CCNL/Randeng-T5-77M',  model=MT5ForConditionalGeneration, tokenizer=T5Tokenizer),
    ModelCard('IDEA-CCNL/Randeng-T5-784M',  model=MT5ForConditionalGeneration, tokenizer=T5Tokenizer),
    ModelCard('IDEA-CCNL/Randeng-MegatronT5-770M',  model=MT5ForConditionalGeneration, tokenizer=T5Tokenizer),  # something error
    ModelCard('IDEA-CCNL/Randeng-Pegasus-238M-Summary-Chinese',  model=PegasusForConditionalGeneration, tokenizer=PegasusTokenizer),
    ModelCard('IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese',  model=PegasusForConditionalGeneration, tokenizer=PegasusTokenizer),
    ModelCard('IDEA-CCNL/Randeng-Pegasus-523M-Chinese',  model=PegasusForConditionalGeneration, tokenizer=AutoTokenizer),
    ModelCard('IDEA-CCNL/Randeng-BART-139M',  model=BartForConditionalGeneration, tokenizer=AutoTokenizer),
    ModelCard('IDEA-CCNL/Randeng-BART-139M-SUMMARY',  model=BartForConditionalGeneration, tokenizer=AutoTokenizer),
    ModelCard('IDEA-CCNL/Wenzhong2.0-GPT2-3.5B',  model=GPT2LMHeadModel, tokenizer=GPT2Tokenizer),
    ModelCard('IDEA-CCNL/Wenzhong-GPT2-110M',  model=GPT2LMHeadModel, tokenizer=GPT2Tokenizer),
    ModelCard('Langboat/mengzi-t5-base',  model=T5ForConditionalGeneration, tokenizer=T5Tokenizer),

    # https://huggingface.co/uer
    ModelCard('uer/chinese_roberta_L-2_H-128', model=BertForMaskedLM, tokenizer=BertTokenizer),
    ModelCard('uer/chinese_roberta_L-2_H-768', model=BertForMaskedLM, tokenizer=BertTokenizer),
    ModelCard('uer/t5-base-chinese-cluecorpussmall',  model=T5ForConditionalGeneration, tokenizer=BertTokenizer),
    ModelCard('uer/t5-small-chinese-cluecorpussmall',  model=T5ForConditionalGeneration, tokenizer=BertTokenizer),
    ModelCard('uer/bart-base-chinese-cluecorpussmall', BartForConditionalGeneration, BertTokenizer),
    ModelCard('uer/gpt2-chinese-poem', model=GPT2LMHeadModel, tokenizer=BertTokenizer),
    ModelCard('uer/gpt2-base-chinese-cluecorpussmall', model=GPT2LMHeadModel, tokenizer=BertTokenizer),
    ModelCard('uer/gpt2-distil-chinese-cluecorpussmall', model=GPT2LMHeadModel, tokenizer=BertTokenizer),
    ModelCard('uer/pegasus-base-chinese-cluecorpussmall', model=PegasusForConditionalGeneration, tokenizer=BertTokenizer),

    ModelCard('voidful/albert_chinese_xxlarge', model=BertModel, tokenizer=BertTokenizer),
    # ModelCard('wptoux/albert-chinese-large-qa'),
    ModelCard("junnyu/roformer_chinese_small", model=RoFormerForMaskedLM, tokenizer=RoFormerTokenizer),
    ModelCard("junnyu/roformer_chinese_base", model=RoFormerForMaskedLM, tokenizer=RoFormerTokenizer),
    ModelCard("junnyu/roformer_chinese_char_base", model=RoFormerForMaskedLM, tokenizer=RoFormerTokenizer),
    ModelCard("ckiplab/gpt2-base-chinese", model=AutoModelForMaskedLM, tokenizer=BertTokenizer),     # https://huggingface.co/ckiplab/gpt2-base-chinese
    ModelCard('fnlp/bart-base-chinese', BartForConditionalGeneration, BertTokenizer),
    # 古文
    ModelCard('ethanyt/guwenbert-base'),         # https://huggingface.co/ethanyt/guwenbert-base
    ModelCard('ethanyt/guwenbert-large'),
    ModelCard('SIKU-BERT/sikubert'),
    ModelCard('SIKU-BERT/sikuroberta'),
    ModelCard('uer/gpt2-chinese-ancient', model=GPT2LMHeadModel, tokenizer=BertTokenizer),
    # English
    ModelCard('prajjwal1/bert-tiny'),
    ModelCard('bert-base-uncased'),
    ModelCard('bert-large-uncased'),
    ModelCard('xlm-roberta-large'),
    ModelCard('distilroberta-base'),
    ModelCard('gpt2-large', model=GPT2LMHeadModel),
    ModelCard('gpt2', model=GPT2LMHeadModel),
    ModelCard('distilgpt2', model=GPT2LMHeadModel),
    ModelCard('microsoft/DialoGPT-large'),
    ModelCard('facebook/bart-base', BartForConditionalGeneration, BartTokenizer),
    ModelCard('allenai/longformer-base-4096'),
    ModelCard('allenai/t5-small-squad11', model=T5ForConditionalGeneration),
    ModelCard('allenai/longformer-large-4096-finetuned-triviaqa'),
    ModelCard('allenai/unifiedqa-t5-small', model=T5ForConditionalGeneration),
    ModelCard('allenai/unifiedqa-t5-11b', model=T5ForConditionalGeneration),
    ModelCard('allenai/unifiedqa-t5-large', model=T5ForConditionalGeneration),
    ModelCard('allenai/t5-small-squad2-question-generation', model=T5ForConditionalGeneration),
    ModelCard('SpanBERT/spanbert-base-cased'),
    ModelCard('SpanBERT/spanbert-large-cased'),
    ModelCard('tuner007/pegasus_paraphrase', model=PegasusForConditionalGeneration, tokenizer=PegasusTokenizer),    #https://huggingface.co/tuner007/pegasus_paraphrase
    ModelCard('t5-large', model=T5ForConditionalGeneration),
    ModelCard('t5-small', model=T5ForConditionalGeneration),
    ModelCard('t5-base', model=T5ForConditionalGeneration),
    ModelCard('facebook/bart-large',BartForConditionalGeneration, BartTokenizer),
    # https://huggingface.co/sshleifer
    ModelCard('sshleifer/tinier_bart', BartForConditionalGeneration, BartTokenizer),
    ModelCard('sshleifer/distilbart-xsum-1-1', BartForConditionalGeneration, BartTokenizer),
    ModelCard('sshleifer/distilbart-xsum-12-1', BartForConditionalGeneration, BartTokenizer),
    ModelCard('sshleifer/distilbart-xsum-12-6', BartForConditionalGeneration, BartTokenizer),
    ModelCard('sshleifer/distilbart-xsum-6-6', BartForConditionalGeneration, BartTokenizer),

    ModelCard('google/bigbird-roberta-base'),
    ModelCard('google/bigbird-roberta-large'),
    ModelCard('google/byt5-small',model=T5ForConditionalGeneration,),
    ModelCard('google/byt5-base',model=T5ForConditionalGeneration),
    ModelCard('google/byt5-large',model=T5ForConditionalGeneration),
    ModelCard('google/byt5-xl',model=T5ForConditionalGeneration),
    ModelCard('google/byt5-xxl',model=T5ForConditionalGeneration),
    ModelCard('google/mt5-small',model=MT5ForConditionalGeneration),
    ModelCard('google/mt5-base' ,model=MT5ForConditionalGeneration),
    ModelCard('google/mt5-large',model=MT5ForConditionalGeneration),
    ModelCard('google/mt5-xxl',model=MT5ForConditionalGeneration),
    ModelCard('google/roberta2roberta_L-24_cnn_daily_mail'),
    ModelCard('google/bigbird-pegasus-large-arxiv'),
    # ModelCard(''),
] 


def get_model_url(string):
    """
    如果该预训练模型已经被保存到本地了，则返回本地路径
    否则，如果在本文件的模型集合中，则下载后保存，如果不在，则直接保存到本地； 最后返回本地路径
    Args:
        string (str): 预训练模型的huggingface标识， 如 hfl/roberta_chinese_wwm

    Returns:
        str : 本地路径
    """
    tmp_model = ModelCard(string)
    if tmp_model.is_saved():
        return tmp_model.local_url
    else:
        for model in models:
            if model.web_dirname == string:
                model.save()
                return model.local_url
            else:
                tmp_model.save()
                return tmp_model.local_url

def saved_info():
    print("\n".join([ f"{_.is_saved()}\t{i}\t{_.local_url}"  for i, _ in enumerate(models)]))

def download(url, model_cls=AutoModel, tokenizer_cls=AutoTokenizer, path="./hgf_model", **kwargs):
    """手动下载模型
    Args:
        url: _description_
        model_cls: _description_. Defaults to AutoModel.
        tokenizer_cls: _description_. Defaults to AutoTokenizer.
        path: _description_. Defaults to "./hgf_model".
    """
    model = model_cls.from_pretrained(url, **kwargs)
    tok = tokenizer_cls.from_pretrained(url)
    if not os.path.exists(path):
        os.makedirs(path)
    model.save_pretrained(path)
    tok.save_pretrained(path)
    print(f"{url} saving to {path}!")
    

def save_model(name, model, tokenizer):
    """
        huggingface model url
        model class
        model tokenizer class 
    """
    tmp_model = ModelCard(name, model, tokenizer)
    tmp_model.save()


def hf_download(repo_id, local_dir):
    """ 
        调用官方接口，将模型下载到本地目录
    """
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=repo_id, 
        allow_patterns=["*.md", "*.txt", "*.json", "*.bin", "*.model"],  # pytorch 
        ignore_patterns=None,
        local_dir=local_dir,
        local_dir_use_symlinks=False
    )


if __name__ == '__main__':
    # for _ in models:
        # print(_.local_url)
    saved_info()
    