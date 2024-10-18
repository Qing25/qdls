
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


from sklearn.model_selection import KFold, StratifiedKFold

@registers.datamodule.register("kfold_dm")
class KFoldDataModule(BaseDataModule):
    """ 
    """
    def __init__(self, config, **kwargs) -> None:
        super().__init__(config, **kwargs)

        

    def set_kfold(self, k, label_key=None, current_k=0):
        """ 在 prepare_data 之后调用"""
        if label_key is None:
            kf = KFold(n_splits=k, shuffle=True, random_state=42)
            y = None 
        else:
            y = self.trainset[label_key]
            kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

        assert hasattr(self, 'num_samples'), "num_samples not found, please remove cache and call prepare_data first"
        for i, (train_idxs, val_idxs) in enumerate(kf.split(range(self.num_samples), y=y)):
            if current_k == i:
                self.train_idxs = train_idxs
                self.val_idxs = val_idxs
                return 
        raise Exception(f"current_k {current_k} not found in kfold range {list(range(k))}")

    def prepare_data(self, mode='train') -> None:
        """设置缓存文件的路径，如果缓存文件不存在则调用DatasetBuilder进行构建
           KFold 要求有一个 train_path （推荐）
           如果 train_path 和 val_path 都存在，则合并之; test_path 独立存在
           用于KFold的所有数据都在 trainset 并缓存到 disk， 通过 set_kfold 设置当前的索引，并在 setup('fit') 时划分数据
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
                if valset is not None:
                    trainset = datasets.concatenate_datasets([trainset, valset])

                if trainset is not None:
                    self.num_samples = len(trainset)
                    trainset.save_to_disk(self.cached_train)

            elif mode == 'test':
            
                testset = ds.build('test', self.tokenizer, self.tokenize_fn)
                assert testset is not None 
                testset.save_to_disk(self.cached_test)
            else:
                raise Exception(f'mode {mode} not imple')
            
            print_string("Data cache re-generated!")

        self.num_samples = len(datasets.load_from_disk(self.cached_train))

    def setup(self, stage: Optional[str] = None) -> None:
        
        if stage == 'fit' or stage is None:
            full_dataset = datasets.load_from_disk(self.cached_train)

            self.trainset = full_dataset.select(self.train_idxs)
            self.valset = full_dataset.select(self.val_idxs)

        if stage == 'test' or stage is None:
            self.testset = datasets.load_from_disk(self.cached_test)
        
        print_string("Datasets setup finished!")
        
