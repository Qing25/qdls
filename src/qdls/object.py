import os
import sys
import json
import yaml
import argparse
from  argparse import Namespace
import omegaconf
from omegaconf import OmegaConf as oc

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import multiprocessing
from colorama import Fore, Back, Style
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# from tensor_ops import safesoftmax
# from q_snippets.tensor_ops import safesoftmax

from abc import ABC, abstractmethod
from pytorch_lightning.loggers import TensorBoardLogger


class RoPE(pl.LightningModule):
    def __init__(self, dim, max_seq_len=513, device=None):
        """
            ref: https://kexue.fm/archives/8265
            创建最大长度，维度固定的pos_emebddings
        """
        super().__init__()
        self.dim = dim
        self.seq_len = max_seq_len
        # self.device = device
        self.embeddings = self._create_pos_embedding()

    def _create_pos_embedding(self):
        position_ids = torch.arange(0, self.seq_len, dtype=torch.float, device=self.device)[None]  # (1, seqlen)
        indices = torch.arange(0, self.dim//2, dtype=torch.float, device=self.device)  # odd not work?
        indices = torch.pow(10000.0, -2*indices/self.dim)
        
        embeddings = torch.einsum("bn,d->bnd", position_ids, indices)
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = torch.reshape(embeddings, (-1, self.seq_len, self.dim))
        return embeddings
    
    def add_pos_embedding(self,qw):
        """
            输入向量序列(bsz,seq_len,dim)，返回乘上了RoPE的结果
        """
        bsz, seq_len, dim = qw.size()
        pos = self.embeddings[:, seq_len, :]
        cos_pos = torch.repeat_interleave(pos[..., 1::2], 2, dim=-1).to(self.device)
        sin_pos = torch.repeat_interleave(pos[..., ::2], 2, dim=-1).to(self.device)
        qw2 = torch.stack([-qw[...,1::2], qw[...,::2]], dim=-1)
        qw2 = torch.reshape(qw2, qw.shape)
        # print(self.device, qw.device, qw2.device, cos_pos.device)
        qw = qw*cos_pos + qw2*sin_pos
        return qw


class TrainerProcess(multiprocessing.Process, ABC):
    """ Only for Cross validation """
    def __init__(self, config, module, dm, k=None):
        """

        Args:
            config (Config):  配置信息
            module (LightningModule):  要运行的 LightningModule
            dm (LightningDataModule):     DataModule
            k (int, optional): 第几个fold. Defaults to None.
        """
        super().__init__()
        self.k = k
        config.sub_dir = os.path.join(config.version, f"fold_{k}" if k is not None else k) 
        self.config = self._prepare_config(config)
        self.model = module(config)
        self.dm = dm
        
        self.logger = self._set_logger()
        self.callbacks = self._set_callbacks()
        self.trainer = self._set_trainer()
    
    @property
    def dirname(self):
        return os.path.join(os.getcwd(), "lightning_logs", self.config.sub_dir)

    @abstractmethod
    def _prepare_config(self, config):
        # logger and model_ckpt require same directory
        
        return config

    @abstractmethod
    def _set_callbacks(self):
        """设置保存模型的路径及参数, 第一个最好是ModelCheckPointCallback
        set self.callbacks = 
            a list of callbacks
        """
        pass
        
    @abstractmethod
    def _set_trainer(self):
        """set self.trainer = 
        """
        pass


    def _set_logger(self):
        """set self.logger here
        """
        logger = TensorBoardLogger(
            save_dir="./lightning_logs/",
            name=None,    # 指定experiment, ./lightning_logs/exp_name/version_name
            version=self.dirname,    # 指定version, ./lightning_logs/version_name
        )
        return logger
        
        
    def run_train(self):
        """进程执行的主函数

        """
        self.trainer.fit(self.model, self.dm)
        # return self.callbacks[0].best_model_path
        self.callbacks[0].to_yaml(os.path.join(self.dirname, "best_k_path.yaml"))

    def run(self):
        self.run_train()


class RNNWrapper(nn.Module):
    """
    Args:
        rnn (nn.Module):  RNN to run 
    """
    def __init__(self, rnn):
        super().__init__()
        self.rnn = rnn
        
    def forward(self, input_sequence, attention_mask):
        """
            输出dim与RNN期待的输出维度相同
        Args:
            input_sequence (Tensor): (bsz,seq_len,dim)
            attention_mask (Tensor): (bsz,seq_len),  0 for masking

        Returns:
            tuple : output => (bsz,seq_len,dim),  padding的位置是0向量
                    hidden => (num_layers*bidirectional, bsz, dim) 如果是LSTM有两个:(last_hidden,cell_state) 不是batch_first
        """
        pack = pack_padded_sequence(input_sequence, attention_mask.sum(1), batch_first=True, enforce_sorted=False)
        o, h = self.rnn(pack)
        output, seq_lens = pad_packed_sequence(o, batch_first=True)
        return output, h


class MultiHeadAttnWrapper(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.atten = nn.MultiheadAttention(dim, num_heads)
        
    def forward(self, q, k, v, q_mask=None, k_mask=None):
        """  1 (bool()转成True)的位置被mask掉

        Args:
            q (Tensor): (bsz,seqlen_target,dim)
            k (Tensor): (bsz,seqlen_source,dim)
            v (Tensor): (bsz,seqlen_source,dim)
            q_mask (Tensor, optional):  (bsz, seqlen_target, optional dim), 1 for masking. Defaults to None.
            k_mask (Tensor, optional): (bsz, seqlen_source, optional dim), 1 for masking. Defaults to None.

        Returns:
            tuple:
                attended_q         (bsz, seqlen_target, dim)    q中的每个向量都是v的加权求和
                attention_weight   (bsz, seqlen_target, seqlen_source)
        """
        if q_mask is not None and k_mask is not None:
            q_mask = q_mask[...,None] if len(q_mask.size()) == 2 else q_mask
            k_mask = k_mask[...,None] if len(k_mask.size()) == 2 else k_mask
            try:
                mask = q_mask.long() & k_mask.permute((0,2,1)).long()              # (bsz, seqlen_target, seqlen_source)
            except:
                mask = q_mask @ k_mask.permute((0,2,1))
            mask = mask.repeat_interleave(self.num_heads, dim=0).bool()
        else:
            mask = None
        atten, attention_weight = self.atten(
            q.permute((1,0,2)),                       # (seqlen_target, bsz, dim)
            k.permute((1,0,2)),v.permute((1,0,2)),    # (seqlen_source, bsz, dim)
            attn_mask=mask                            # (bsz*nheads, seqlen_target, seqlen_source)
        )
        return atten.permute((1,0,2)), attention_weight

# class Config(argparse.Namespace):
class Config(oc):
    def __init__(self):
        pass

    def __repr__(self):
        return json.dumps(self.__dict__, indent=2, default=lambda x: x.__dict__, ensure_ascii=False) 

    def as_dict(self):
        return eval(json.dumps(self.__dict__, indent=2, default=lambda x: x.__dict__, ensure_ascii=False))

    def save_to_json_file(self, to_file):
        with open(to_file, "w") as fout:
            json.dump(self.__dict__, fout, indent=2, default=lambda x: x.__dict__, ensure_ascii=False)
            print(f"----- config saved to {to_file} -------")

    @classmethod
    def from_json_file(cls, in_file):
        with open(in_file) as fin:
            config_dict = json.load(fin)
            return cls.from_dict(config_dict)

    @classmethod
    def from_yaml_file(cls, in_file):
        """ 可以直接使用Config.load(yaml_path), 调用的是OmegaConf的load """
        try:
            config_dict = yaml.safe_load(open(in_file))
        except:
            config_dict = yaml.load(open(in_file), Loader=yaml.Loader)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict):
        config = cls()
        to_remove = []
        for key, value in config_dict.items():
            if isinstance(value, dict):
                value = cls.from_dict(value)
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
            else:
                setattr(config, key, value)

        if to_remove:
            print("Config override keys: %s" % ",".join(to_remove))

        return config       
    
    def save_to_yaml_file(self, to_file):
        """保存OmegaConf到yaml文件

        Args:
            config (OmegaConf): config 对象
            path (str): 目标yaml
        """
        with open(to_file, 'w') as f:
            config = oc.create(self.as_dict())            
            oc.save(config, f)
            print(f"----- config saved to {to_file} -------")

    @staticmethod
    def from_argparse(parser: argparse.ArgumentParser):
        """返回从
            一个ArgumentParser对象获取的命令行参数，
            没出现在命令行中设定好默认值的参数
           两个OmegaConf对象。
        ！命令行设置时，不要加=， 而是直接空格隔开 !! 可以用=了

        Args:
            parser (argparse.ArgumentParser): 入口文件中定义的parser

        Returns:
            tuple: (OmegaConf, OmegaConf)
        """
        #  {'data.train_bsz': '--train_bsz'}
        dest_to_arg = {v.dest: k for k, v in parser._option_string_actions.items()}

        all_args = vars(parser.parse_args())
        provided_args = {}
        default_args = {}
        sys_argv = [ s.split("=")[0] for s in sys.argv[1:]]  # 除去 =
        for k, v in all_args.items():
            if dest_to_arg[k] in sys_argv:
                provided_args[k] = v
            else:
                default_args[k] = v
        # print(provided_args,"\n---------------\n", default_args)
        provided_dotlist = [ f"{k}={v}" for k,v in provided_args.items()] if provided_args != {} else []
        provided = oc.from_dotlist(provided_dotlist)
        default_dotlist = [ f"{k}={v}" for k,v in default_args.items()]
        defaults = oc.from_dotlist(default_dotlist)

        return provided, defaults  


# def print_string(string):
#     """ 两边是==，输入字符串居中，两边各有5个空格，输出字体颜色为黄色 """
#     print(Fore.YELLOW,"="*25 + string.center(len(string)+10,) + "="*25)
#     print(Style.RESET_ALL)

# def print_config(config):
#     """ 打印字典，config等， 字体颜色为绿色 """
#     print("="*80)
#     if type(config) is omegaconf.dictconfig.DictConfig:
#         config_dict = oc.to_object(config)
#     else:
#         config_dict = namespace2dict(config)
#     s = json.dumps(config_dict, ensure_ascii=False, indent=2)
#     print(Fore.GREEN, s)
#     print(Style.RESET_ALL)
#     print("="*80)

def resume_config(config, path):
    """ 
    pytorch lightning save_hyperparameters() 保存的yaml文件 在预测的时候需要更新
    从原先保存的yaml恢复，重写新的 mode max_epochs, data and preds 
        config: 当前的config
        path: yaml 路径

    """

    new_config = Config.load(path)
    # new_config = oc.merge(old_config, config)
    new_config.mode = config.mode 
    new_config.max_epochs = config.max_epochs

    new_config.data.test_path = config.data.test_path
    new_config.data.test_bsz = config.data.test_bsz
    new_config.data.force_reload = config.data.force_reload

    new_config.preds.ckpt_path = config.preds.ckpt_path
    new_config.preds.result_path = config.preds.result_path
    print_string(f"config file resumed from {path}")
    return new_config



def build_config(d):
    """ 嵌套的 argparse.Namespace 
        TODO: move to utils
    """
    config = Namespace()
    for k,v in d.items():
        if type(v) == dict:
            setattr(config,k, build_config(v))
        else:
            setattr(config, k, v)
    return config 
    
def namespace2dict(config):
    """
     TODO: move to utils, rename to config2dict
    """
    d = {}
    for k,v in config.__dict__.items():
        if type(v) is Namespace:
            d[k] = namespace2dict(v)
        else:
            d[k] = v 
    return d 


if __name__ == '__main__':
    # print( os.getcwd())
    # config = Config.from_dict({'a':23, 'b':{'c':'we'}})
    # print(config)
    # config.save_to_json_file("tmp.json")
    # c1 = Config.from_json_file("tmp.json")
    # print("c1", c1)
    # c1.save_to_yaml_file("tmp.yaml")
    # c2 = Config.from_yaml_file("tmp.yaml")
    # print("c2", c2)
    # print(hasattr(c2, 'model_specific'))
    # c2.model_specific = {'k':10, 'n':50}
    # print(hasattr(c2, 'model_specific'), c2)
    # print_info("This config:")


    # c2 = Config.load("./tmp.yaml")
    # print_config(c2)
    # print_string("THis is good"*5)
    
    def test_namespace():
        d = {
            'name': {
                '1': 1,
                '2':2
            },
            'list': [1,2,3]
        }
        c = build_config(d)
        print(c)
        print(namespace2dict(c))

    test_namespace()