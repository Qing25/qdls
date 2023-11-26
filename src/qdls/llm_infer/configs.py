from dataclasses import dataclass

@dataclass
class Config:
    model_path: str 
    vocab_size: int
    tensor_parallel_size: int 
    temperature: float = 0.0
    max_tokens: int = 64
    logprobs: int = None 

    def to_dict(self):
        return self.__dict__



qwen_7b_config = Config("/sshfs/pretrains/Qwen/Qwen-7B-Chat", 151936, 1)
qwen_14b_config = Config("/sshfs/pretrains/Qwen/Qwen-14B-Chat", 152064, 1)

chatglm2_6b_config = Config("/sshfs/pretrains/THUDM/chatglm2-6b", 65024, 1)
chatglm3_6b_config = Config("/sshfs/pretrains/THUDM/chatglm3-6b", 65024, 1)

llama2_7b_config = Config("/sshfs/pretrains/meta-llama/Llama-2-7b-chat-hf", 32000, 1)
llama2_70b_config = Config("/sshfs/pretrains/meta-llama/Llama-2-70b-chat-hf", 32000, 4)

baichuan_7b_config = Config("/sshfs/pretrains/baichuan-inc/baichuan-7b-sft", 64000, 1)

baichuan2_7b_config = Config("/sshfs/pretrains/baichuan-inc/Baichuan2-7B-Chat",  125696, 1)
baichuan2_13b_config = Config("sshfs/pretrains/baichuan-inc/Baichuan2-13B-Chat", 125696, 2)

MODEL_DICT = {
    'qwen_7b': qwen_7b_config,
    'chatglm2_6b': chatglm2_6b_config,
    'chatglm3_6b': chatglm3_6b_config,
    'llama2_7b': llama2_7b_config,
    'llama2_70b': llama2_70b_config,
    'baichuan_7b': baichuan_7b_config,
    'baichuan2_7b': baichuan2_7b_config,
    'baichuan2_13b': baichuan2_13b_config,
}