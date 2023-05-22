
from qdls.data import load_json
import datasets 


class DatasetBuilder():
    def __init__(self, config) -> None:
        """
           应当支持初始化两种方式
            1.huggingface 在线dataset
            2.本地文件 train_path val_path test_path
        """
        if not hasattr(config.data, 'dataset_name'):
            assert hasattr(config.data, 'train_path') and \
                    hasattr(config.data, 'val_path') and \
                    hasattr(config.data, 'test_path'), \
            f"train_path, val_path, test_path should be provided when dataset_name is not provided"
            self._init_from_paths(config)
        else:
            assert hasattr(config.data, 'dataset_name'), \
            f"you should provide dataset_name since no train_path, val_path, test_path is provided"
            self._init_from_hf(config)

        self.config = config 

    
    def _init_from_paths(self, config):
        """ 从本地文件加载数据 """
        self.trainset = load_json(config.data.train_path)
        self.valset = load_json(config.data.val_path)
        self.testset = load_json(config.data.test_path)

        
    def _init_from_hf(self, config, proportion=0.8):
        """ 如果只有两个字段（大概率train test）则将train划分为train val """
        ds = datasets.load_dataset(config.data.dataset_name)
        if len(ds) == 2:
            train_ds = ds['train'].train_test_split(test_size=1-proportion)
            self.trainset = train_ds['train']
            self.valset = train_ds['test']

        self.testset = ds['test'] 
 
    
    def build(self, mode, tokenizer, tokenize_fn):
        """ 
            mode: train, val, test
        """
        if mode == 'train':
            data  = self.trainset
        elif mode == 'val':
            data  = self.valset 
        elif mode == 'test':
            data  = self.testset
        else:
            raise Exception(f"{mode} is not implemented!")
        if type(data) is list:                              # 本地加载的数据是list
            dataset = datasets.Dataset.from_list(data)
        else:
            dataset = datasets.Dataset.from_dict(data)
        dataset = dataset.map(
            tokenize_fn, 
            batched=False,
            num_proc=self.config.data.num_proc, 
            fn_kwargs={'tokenizer': tokenizer, 'mode': mode}
        )
        return dataset