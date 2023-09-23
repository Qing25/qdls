import torch 
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from tqdm import tqdm
from qdls.utils import print_string

class BatchEvalCallback(Callback):
    """ 
        在训练过程中，validataion epoch 结束后，使用训练好的模型生成一些文本打印出来，用于人工评估模型效果
    """
    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pl_module.model.eval()
        with torch.no_grad():
            try:
                s = trainer.datamodule.trainset[0]['input_text']
            except:
                s = 'This is the default prompt for evalation. We only calc val_loss during validation because it will be too slow to generate text for all valset.' + \
                    'So we use this default prompt to generate only one sentence for evaluation.'
            td = trainer.datamodule.tokenizer(s, return_tensors='pt').to(pl_module.model.device)
            pred_ids = pl_module.model.generate(
                input_ids=td.input_ids, attention_mask=td.attention_mask,
                max_new_tokens=384, do_sample=False, num_beams=1,
            )
            print_string("eval: pred_examples:")
            print(trainer.datamodule.tokenizer.batch_decode(pred_ids, clean_up_tokenization_spaces=True, skip_special_tokens=True)," ||| ", s)
        
        pl_module.model.train()



from weakref import proxy
class HuggingfaceModelCKPTCallback(ModelCheckpoint):
    """ 
        修改自 pytorch_lightning.callbacks.ModelCheckpoint 使得其使用 save_pretrained 保存模型权重
        适用于 huggingface transformers 模型 以及 peft 模型
        
    """
    def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
        
        # trainer.save_checkpoint(filepath, self.save_weights_only)
        trainer.model.model.save_pretrained(filepath)
        trainer.datamodule.tokenizer.save_pretrained(filepath)

        self._last_global_step_saved = trainer.global_step

        # notify loggers
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))