import os 
import sys 
os.environ["TOKENIZERS_PARALLELISM"] = "true"

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = FILE_DIR[:FILE_DIR.index('src')]
sys.path.append(PROJ_DIR)

import torch 
torch.set_float32_matmul_precision('high')
import pytorch_lightning as pl
pl.seed_everything(3407)

import fire
from omegaconf import OmegaConf 
from qdls.utils import print_config, print_string
from qdls.reg.register import import_all_modules_for_register
import_all_modules_for_register()

from frame import train_model, predict_model, data_dev

def main(config_file=None, mode=None, version='default_version', **kwargs):
    if config_file is None:
        raise Exception(f"must specify a configuration file to start!")
    
    config = OmegaConf.load(config_file)
    config.version = version 
    if mode is not None:
        config.mode = mode 
    
    cli_str = [ f"{k}={v}" for k,v in kwargs.items() ]
    config = OmegaConf.unsafe_merge(config, OmegaConf.from_cli(cli_str))
    print_config(config)
 
    if config.mode == 'train':
        train_model(config)
    elif config.mode in ['predict']:
        predict_model(config)
    elif config.mode == 'data_dev': # for dataloader development
        data_dev(config)
    else:
        raise Exception(f"mode `{config.mode}` is not implemented yet!")


if __name__ == '__main__':
    
    fire.Fire(main)