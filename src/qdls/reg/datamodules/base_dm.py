
import os 
import sys 

import datasets 
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer 
import pytorch_lightning as pl
from typing import * 

from ...utils import print_string, print_config
from .dataset_builder import DatasetBuilder
from ..register import registers


@registers.datamodule.register("base_dm")
class BaseDataModule(pl.LightningDataModule):
    """ 
        config:
            pretrained
            data:
                train_bsz
                val_bsz
                test_bsz
                padding_side
                collator_name
                tokenize_fn_name
                cache_dir
                force_reload
        需要先在registers中注册collator和process_function
        DataBuilder 需要 train_path val_path test_path 或者 dataset_name
    """
    def __init__(self, config, **kwargs) -> None:
        super().__init__() 

        self.config = config 
        print_string("configuration of datamodule")
        print_config(self.config.data)

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.pretrained,
            trust_remote_code=True, 
            use_fast=getattr(config.data, 'use_fast', False),
            padding_side = config.data.padding_side
        )
        
        # 有些模型的tokenizer没有设置 pad token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id 


        self.collator_cls = registers.collator.get(self.config.data.collator_name) 
        self.tokenize_fn = registers.process_function.get(self.config.data.tokenize_fn_name)

        self.num_workers = kwargs.get('num_workers', 4)

    def prepare_data(self, mode='train') -> None:
        """设置缓存文件的路径，如果缓存文件不存在则调用DatasetBuilder进行构建

        Args:
            mode: train or test. Defaults to 'train'.

        """
        self.cached_train = os.path.join(self.config.data.cache_dir, f"train_{self.config.data.tokenize_fn_name}")
        self.cached_val = os.path.join(self.config.data.cache_dir, f"val_{self.config.data.tokenize_fn_name}")
        self.cached_test = os.path.join(self.config.data.cache_dir, f"test_{self.config.data.tokenize_fn_name}")

        if not os.path.exists(self.config.data.cache_dir):
            os.makedirs(self.config.data.cache_dir, exist_ok=True)

        # 强制 or 训练时没有self.cached_train or 预测时没有self.cached_test
        if self.config.data.force_reload or ( (not os.path.exists(self.cached_train)) and mode=='train') \
            or ( (not os.path.exists(self.cached_test)) and mode =='test'):
            ds = DatasetBuilder(self.config)
            if mode == 'train':
                trainset = ds.build('train', self.tokenizer, self.tokenize_fn)
                valset =  ds.build('val', self.tokenizer, self.tokenize_fn)
                
                if trainset is not None:
                    trainset.save_to_disk(self.cached_train)
                if valset is not None:
                    valset.save_to_disk(self.cached_val)
            elif mode == 'test':
            
                testset = ds.build('test', self.tokenizer, self.tokenize_fn)
                assert testset is not None 
                testset.save_to_disk(self.cached_test)
            else:
                raise Exception(f'mode {mode} not imple')
            
            print_string("Data cache re-generated!")

    def setup(self, stage: Optional[str] = None) -> None:
        
        if stage == 'fit' or stage is None:
            self.trainset = datasets.load_from_disk(self.cached_train)
            self.valset = datasets.load_from_disk(self.cached_val)

        if stage == 'test' or stage is None:
            self.testset = datasets.load_from_disk(self.cached_test)
        
        print_string("Datasets setup finished!")

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.config.data.train_bsz,
            shuffle=True,
            collate_fn=self.collator_cls(self.tokenizer, mode='train'),
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size=self.config.data.val_bsz, 
            shuffle=False,
            collate_fn=self.collator_cls(self.tokenizer, mode='val'),
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(self.testset, batch_size=self.config.data.test_bsz, 
            shuffle=False,
            collate_fn=self.collator_cls(self.tokenizer, mode='test'),
            num_workers=self.num_workers
        )


