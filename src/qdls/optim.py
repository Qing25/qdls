

def get_grouped_params(self):
    no_decay = ["bias", "LayerNorm.weight"]

    # Group parameters to those that will and will not have weight decay applied
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters

def get_finetune_param_groups(self):
    no_decay = ["bias", "LayerNorm.weight"]
    backbone_no_decay = {
        'params': [ p for n,p in self.model.named_parameters() if n.startswith("encoder") and any(nd in n for nd in no_decay) ],
        'weight_decay' : 0.0,
        'lr': 1e-5
    }
    backbone = {
        'params': [ p for n,p in self.model.named_parameters() if n.startswith("encoder") and not any(nd in n for nd in no_decay) ],
        'weight_decay' : 0.01,
        'lr': 1e-5
    }
    new_initialized = {
        'params': [ p for n,p in self.model.named_parameters() if not n.startswith("encoder") ],
        'lr': 3e-5
    }
    return [backbone_no_decay, backbone, new_initialized]


