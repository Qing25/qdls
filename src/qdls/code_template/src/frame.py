import os,sys 
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ_DIR = FILE_DIR[:FILE_DIR.index('src')]
sys.path.append(PROJ_DIR)

from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from qdls.data import load_json, save_json
from qdls.utils import print_dict
from qdls.reg.register import registers
from qdls.reg.datamodules.base_dm import BaseDataModule


from src.models import *
from src.pl_callbacks import *
from src.data_utils import *


def train_model(config):
    dm = BaseDataModule(config)
    dm.prepare_data()
    dm.setup("fit")
    print_dict(dm.trainset[0])
    # dm.valset = dm.valset.shard(index=0, num_shards=10)
    # print(f"len dm.valset: {len(dm.valset)}")

    model = registers.model.get(config.model_type)(config)

    logger = TensorBoardLogger(
        save_dir="./lightning_logs/",
        name=None,                # 指定experiment, ./lightning_logs/exp_name/version_name
        version=config.version,   # 指定version, ./lightning_logs/version_name
    )
    CUR_DIR = os.getcwd()
    dirname = os.path.join(CUR_DIR, "./lightning_logs/", config.version)
    ckpt_callback = HuggingfaceModelCKPTCallback(
        dirpath=dirname,
        filename="{epoch}_{val_loss:.4f}",   # 模型保存名称， epoch信息以及验证集分数
        monitor='val_loss',
        mode='min',
        save_top_k=2,
        verbose=True,
    )
    config.total_steps = len(dm.train_dataloader()) * config.trainer.max_epochs \
        if 'max_epochs' in config.trainer else config.trainer.max_steps
    es = EarlyStopping(monitor='val_loss', mode='min', patience=3)

    trainer = pl.Trainer(
        accelerator='gpu', 
        num_nodes=config.trainer.num_nodes,                # 一台机器
        devices=config.trainer.devices,                  # 两张显卡
        strategy=config.trainer.strategy,
        # strategy='deepspeed_stage_2_offload', # does not support AdamW
        precision=config.trainer.get('precision', '32-true'),
        logger=logger,
        log_every_n_steps=10,
        callbacks=[ckpt_callback,es, BatchEvalCallback(), ],
        accumulate_grad_batches=config.trainer.accumulate_grads,
        val_check_interval=config.trainer.get('val_check_interval', 1.0),
        check_val_every_n_epoch=config.trainer.get('check_val_every_n_epoch', None),  # If you want to validate based on the total training batches,
        max_steps=config.trainer.get('max_steps', -1),
        max_epochs=config.trainer.get('max_epochs', None),
        # limit_train_batches=60,
        # limit_val_batches=5
    )

    trainer.fit(model, datamodule=dm)


def predict_model(config):
    dm = BaseDataModule(config)
    dm.prepare_data('test')
    dm.setup("test")
    print_dict(dm.testset[0])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.pretrained = config.pred.ckpt_path if config.pred.ckpt_path is not None else config.pretrained
    print_string(f"ckpt model {config.pretrained} set to config.pretrained")
    model = registers.model.get(config.model_type)(config).to(device) 
    T = [] 
    with torch.no_grad():
        for batch in tqdm(dm.test_dataloader()):
            batch = { k:v.to(device) if type(v) is torch.Tensor else v for k,v in batch.items()}
            pred_ids = model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                num_beams=1, do_sample=False,
                max_new_tokens=config.pred.get('max_new_tokens', 256),
            )
            texts = dm.tokenizer.batch_decode(pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            T.extend(texts)
    R = []
    for sample, pred in zip(dm.testset, T):
        d = {
            k:v for k,v in sample.items() if k not in ['input_ids', 'attention_mask', 'labels']
        }
        d['pred'] = pred 
        R.append(d)
    save_json(R, config.pred.result_path)


def data_dev(config):
    pass 
