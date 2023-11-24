# -*- coding: utf-8 -*-
# @File    :   utils.py
# @Time    :   2023/05/21 16:47:34
# @Author  :   Qing 
# @Email   :   aqsz2526@outlook.com
######################### docstring ########################
'''
    常用的，不依赖于不常见模块的函数（使用时直接from qdls.utils import *）
'''
import json 
import sys
import omegaconf

from termcolor import colored, cprint
from argparse import Namespace

def print_dict(d):
    s = json.dumps(d, indent=4, ensure_ascii=False)
    print(s)


def build_config(d):
    """ 嵌套的 argparse.Namespace """
    config = Namespace()
    for k,v in d.items():
        if type(v) == dict:
            setattr(config,k, build_config(v))
        else:
            setattr(config, k, v)
    return config 


def config2dict(config):
    """ 将 argparse.Namespace 转换为 dict
        TODO: 支持更多类 dataclass 等 
    """
    if type(config) is omegaconf.dictconfig.DictConfig:
        d = omegaconf.OmegaConf.to_object(config)
        return d
    d = {}
    for k,v in config.__dict__.items():
        if type(v) is Namespace:
            d[k] = config2dict(v)
        else:
            d[k] = v 
    return d 


def print_string(string, color="yellow"):
    """ 在命令行以黄色字体打印 string 
        eg: ====================  string  ====================
    """
    if type(string) is not str:
        try:
            string = str(string)
        except:
            raise Exception("Input must be str or can be converted to str")

    s = "="*20 + string.center(len(string)+10,) + "="*20
    cprint(s, color, attrs=["bold"])


def print_config(config, title=None):
    """在命令行以 json 格式打印 config 绿色字体;   

    Args:
        config:  Config object  
        title: config 的名字. Defaults to None.
    """
    if title is None:
        print("="*80)
    else:
        print_string(title)

    config_dict = config2dict(config)
    config_string = json.dumps(config_dict, indent=4, ensure_ascii=False)
    cprint(config_string, "green")

    print_string("END OF CONFIG")


def namespace2dict(config):
    """ 将 argparse.Namespace 转换为 dict
    """
    d = {}
    for k,v in config.__dict__.items():
        if type(v) is Namespace:
            d[k] = namespace2dict(v)
        else:
            d[k] = v 
    return d 



