# -*- coding: utf-8 -*-
# @File    :   loggers.py
# @Time    :   2023/03/26 10:43:18
# @Author  :   Qing 
# @Email   :   aqsz2526@outlook.com
######################### docstring ########################
'''
    封装一下常用的logger
'''
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger


def get_wandb_logger(config):
    
    logger = WandbLogger(project=config.project, name=config.version, job_type=config.job_type)

    return logger 


def get_tfb_logger(config):
    logger = TensorBoardLogger(
        save_dir="./lightning_logs/",
        name=None,                # 指定experiment, ./lightning_logs/exp_name/version_name
        version=config.version,   # 指定version, ./lightning_logs/version_name
    )
    return logger