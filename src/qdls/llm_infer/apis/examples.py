# -*- coding: utf-8 -*-
# @File    :   examples.py
# @Time    :   2024/06/26 15:14:02
# @Author  :   Qing 
# @Email   :   aqsz2526@outlook.com
######################### docstring ########################
'''
    调用 API 示例
'''
from loguru import logger
import os 
try:
    import chattool
except Exception as e:
    logger.warning(f"chattool not found: {e}, run `pip install -U chattool`")
    

TONGYI_API_KEY = os.environ.get('TONGYI_API_KEY', None) 
TONGYI_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

CHATANY_API_KEY = os.environ.get('CHATANY_API_KEY', None)
CHATANY_API_URL = os.environ.get('CHATANY_API_URL' ,"https://api.chatanywhere.tech/v1")

KIMI_API_KEY = os.environ.get('KIMI_API_KEY', None)
KIMI_API_URL = os.environ.get('KIMI_API_URL', "https://api.moonshot.cn/v1")


ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", None)

def openai_demo(prompt):
    """Openai 官方库调用通义API示例
        json format 的使用示例
    Args:
        prompt: str
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=TONGYI_API_KEY,  # 替换成真实DashScope的API_KEY
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope服务endpoint
    )

    completion = client.chat.completions.create(
        model="qwen-long",
        messages=[
            {
                'role': 'system',
                'content': 'You are a helpful assistant.'
            },
            {
                'role': 'user',
                'content': prompt
            },

        ],
        stream=False,
        temperature=0.0,
        # response_format={ "type": "json_object" } # 返回json格式式 prompt 中要有 json 这个关键字
    )
    print(completion)
    # print(completion.choices[0].message.content)
    return completion



def zhipu_demo():
    """ 
    https://open.bigmodel.cn/dev/api#language
    """
    from zhipuai import ZhipuAI
    client = ZhipuAI(api_key=ZHIPU_API_KEY)
    response = client.chat.completions.create(
    model="glm-4",  # 填写需要调用的模型名称
    messages=[
            {"role": "user", "content": "作为一名营销专家，请为智谱开放平台创作一个吸引人的slogan"},
            {"role": "assistant", "content": "当然，为了创作一个吸引人的slogan，请告诉我一些关于您产品的信息"},
            {"role": "user", "content": "智谱AI开放平台"},
            {"role": "assistant", "content": "智启未来，谱绘无限一智谱AI，让创新触手可及!"},
            {"role": "user", "content": "创造一个更精准、吸引人的slogan"}
        ],
    )
    print(response.choices[0].message)
    return  

def kimi_demo():
    """
    https://platform.moonshot.cn/docs/intro#%E4%B8%BB%E8%A6%81%E6%A6%82%E5%BF%B5 
    兼容 Openai API
    "moonshot-v1-8k"
    """
    pass


def chattool_demo(prompt="hello"):
    import chattool
    API_URL = "https://api.chatanywhere.tech/v1"
    API_KEY = os.environ.get('CHATANY_API_KEY', None) # local vllm set to "" 
    chattool.api_base = API_URL
    chattool.api_key = API_KEY
    c = chattool.Chat(model="gpt-4-turbo")
    # c.system()
    c.user(prompt)
    c.getresponse(
        temperature=0.0, 
        response_format={ "type": "json_object" }
        )
    print(c.chat_log)
    return c 

def chattool_parallel_demo():
    import chattool

    chattool.base_url=CHATANY_API_URL; chattool.api_key = CHATANY_API_KEY
    prompts = [] 
    model = 'gpt-4o-mini'; version= 'test_demo'
    ckpt_path = f"./cache/{model}_{version}.jsonl"
    def data2chat(prompt):
        c = chattool.Chat()
        c.user(prompt)
        return c

    chattool.async_chat_completion(
        prompts,
        model=model,
        chkpoint=ckpt_path,
        nproc=8,
        data2chat=data2chat,
        timeinterval=1, max_tries=3, timeout=3,
        stop="</s>"
    )
    chats = chattool.load_chats(ckpt_path)

    print(len(chats), "loaded! ")

    for chat in chats:
        output = chat.chat_log[-1]['content']


def langchain_demo(prompt="hello"):
    """ 
     langchain OpenAI 调用本地模型
    """
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model="/sshfs/pretrains/THUDM/glm-4-9b-chat",
        temperature=0,
        openai_api_key='EMPTY',
        openai_api_base=f'http://127.0.0.1:8000/v1', # sparql only proto
        max_tokens=1024,
        verbose=True
    )
    # to pass stop_token_ids to vllm 
    extra_body={
        "stop_token_ids": [151329, 151336, 151338]
        }
    print(llm.invoke(prompt, extra_body=extra_body))

if __name__ == '__main__':
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    prompt = [SystemMessage(), HumanMessage("你好")]
    langchain_demo(prompt)
    pass 



