
import os 
import json 
from tqdm import tqdm 
from abc import ABC, abstractmethod

from qdls.data import load_json, save_json, load_jsonl
from vllm import LLM, SamplingParams

class VLLMBaseRunner:
    """ 
    config: OmegaConf
        version: str
        model_path: str
        temperature: float
        max_tokens: int
        stop: str
        tensor_parallel_size: int

        实现build_prompt_fn
        cache 文件夹是生成缓存
        prompts 文件夹是 prompt 缓存, 有些 prompt 生成速度较慢
    """
    def __init__(self, config) -> None:
        self.config = config 

        self.version = config.version 
        self.cache_file = self._handle_cache()
        tensor_parallel_size = getattr(config, 'tensor_parallel_size', 1)
        stop_token = getattr(config, 'stop', '</s>')

        self.test_data = load_json(config.test_path)
        self.prompts = self._handle_prompt_cache()

        self.llm = LLM(
            model=config.model_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True
        )
        

        self.sampling_params = SamplingParams(
                temperature=config.temperature, 
                max_tokens=config.max_tokens, 
                stop=stop_token
            )


    def _handle_cache(self):
        """ define cache file name """
        if not os.path.exists("./cache/results"):
            os.makedirs("./cache/results")
        self.cache_file = f"./cache/results/{self.version}.jsonl"


        # 从缓存中加载 TODO: 目前没有用，不是调 API 计费的情况下，不太需要缓存
        if os.path.exists(self.cache_file):
            print(f"cache file {self.cache_file} exists, loading...")
            self.processed = load_jsonl(self.cache_file)
            print(f"loaded {len(self.processed)} chats")

        return open(self.cache_file, 'a')


    def _handle_prompt_cache(self):
        if not os.path.exists("./cache/prompts"):
            os.makedirs("./cache/prompts", exist_ok=True)

        self.prompt_cache_file = f"./cache/prompts/{self.version}.json"

        if os.path.exists(self.prompt_cache_file):
            print(f"cache file {self.prompt_cache_file} exists, loading...")
            prompts = load_json(self.prompt_cache_file)
            print(f"loaded {len(prompts)} chats")
        else:
            prompts = self.build_prompts(self.test_data)
            save_json(prompts, self.prompt_cache_file)

        return prompts

    @abstractmethod
    def build_prompt_fn(sample):
        """ 为一条数据构建prompt 
            应该是一个 staticmethod
        """
        pass 


    def build_prompts(self, samples, nproc=32):
        """ if use_tqdm 是数据量非常多的情况下，可以使用tqdm显示进度条; 否则是为一个batch构建prompt """

        if nproc == 1:
            R = [] 
            for sample in tqdm(samples, desc="Building prompts"):
                prompt = self.build_prompt_fn(sample)
                R.append(prompt)
            return R 
        else:
            from multiprocessing import Pool 
            with Pool(nproc) as p:
                R = list(tqdm(p.imap(self.build_prompt_fn, samples), total=len(samples), desc=f"Building prompts with {nproc} processes"))
            return R
        


    def generate_from_samples(self, data, bsz=8):
        """ 
        从原始的数据自行分batch进行生成，生成一个 batch 存储一下，如此 cache file 才有意义
        不足：构建 prompt 比较耗费时间的话，这样会导致生成速度变慢
        Args:
            data: list of samples 
            bsz: batch size
        
        """

        for i in tqdm(range(len(data)//bsz+1)):
            batch = data[i*bsz:(i+1)*bsz]
            if batch == []:
                break

            prompts = self.build_prompts(batch)

            outputs = self.llm.generate(prompts, self.sampling_params, use_tqdm=False)

            # Print the outputs.
            for sample, output in zip(batch, outputs):
                prompt = output.prompt
                generated_text = output.outputs[0].text
                sample['vllm'] = generated_text 

                self.cache_file.write(json.dumps(sample, ensure_ascii=False) + '\n')

        self.cache_file.close()


    def generate_from_prompts(self):
        """ 先构建好所有的prompts 交给vllm生成 
            生成完之后，再存储
        """
 
        outputs = self.llm.generate(self.prompts, self.sampling_params, use_tqdm=True)
        for sample, output in zip(self.test_data, outputs):
            generated_text = output.outputs[0].text
            sample['vllm'] = generated_text
            sample['prompt'] = output.prompt
            self.cache_file.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        self.cache_file.close()

        