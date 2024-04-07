

from abc import ABC, abstractmethod
from typing import List, Any,  Union, Callable
from jieba import cut
from rank_bm25 import BM25Okapi
from qdls.gql.cypher.utils.parse import split_cypher # TODO: unified interface for cypher sparql, etc 

from .base_retriever import Retriever

class LanguageProcessor(ABC):
    @abstractmethod
    def tokenize(self, document: str) -> List[str]:
        pass

class EnglishProcessor(LanguageProcessor):
    def tokenize(self, document: str) -> List[str]:
        return document.split(" ")

class ChineseProcessor(LanguageProcessor):
    def tokenize(self, document: str) -> List[str]:
        return list(cut(document))

class CypherProcessor(LanguageProcessor):
    def tokenize(self, document: str) -> List[str]:
        return split_cypher(document)

class BM25Retriever(Retriever):
    def __init__(self, data_source: List[Any], key: Union[str, Callable] = None, lang: str = 'en') -> None:
        """
        初始化一个处理文本数据的实例。

        参数:
        - data_source: List[Any], 数据源列表，每个元素可以是任何类型，但必须包含可处理的文本信息。
        - key: str, 用于从数据源中提取文本信息的键。此键应指向包含文本数据的字段。
        - lang: str, 文本的语言，默认为'en'表示英语。

        返回值:
        - 无
        """
        # 确保key已经被指定
        assert key is not None, "key is not specified"
        # 确保data_source是一个列表
        assert isinstance(data_source, list), "data_source must be a list"
        self.data_source = data_source
        # 使用指定的key处理数据源，提取需要的文本项
        self.items4retrieval = self._process_key(key)
        self.lang = lang
        # 初始化语言处理器，用于处理特定语言的文本
        self.language_processor = self._get_language_processor()
        # 对数据源进行分词处理
        self.tokenized_corpus = self._tokenize_data()
        # 初始化BM25索引，用于文本的检索
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _process_key(self, key: Union[str, Callable]) -> List[Any]:
        """
        根据提供的键对数据源中的每个样本进行处理。
        
        如果键是一个可调用对象，则将对每个样本应用该函数，并返回结果列表。
        如果键是一个字符串，则将它作为样本字典的键来访问，并返回每个键对应的值的列表。
        
        参数:
        key: str - 用于处理数据源样本的键。可以是函数，也可以是字符串。
        
        返回值:
        List[Any] - 根据提供的键处理数据源样本后的结果列表。
        """
        if callable(key):
            # 如果key是可调用的，则对每个样本应用该函数
            return [key(sample) for sample in self.data_source]
        elif isinstance(key, str):
            # 如果key是字符串，则用它来访问每个样本的值
            return [sample[key] for sample in self.data_source]
        else:
            # 如果key既不是可调用的也不是字符串，则抛出异常
            raise Exception(f"key should be dict key or callable function")

    def _get_language_processor(self) -> LanguageProcessor:
        if self.lang == 'en':
            return EnglishProcessor()
        elif self.lang == 'zh':
            return ChineseProcessor()
        elif self.lang == 'cypher':
            return CypherProcessor()
        else:
            raise ValueError(f"lang must be 'en', 'zh', or 'cypher', but got {self.lang}")

    def _tokenize_data(self) -> List[List[str]]:
        return [self.language_processor.tokenize(doc) for doc in self.items4retrieval]

    def get_topk_samples(self, query: str, topk: int = 5) -> List[Any]:
        """
        获取与查询字符串最相关的前k个样本。
        
        参数:
        - query: str，用户输入的查询字符串。
        - topk: int，返回的最相关样本数量，默认为5。
        
        返回值:
        - List[Any]，包含最相关样本的列表。
        """
        # 对查询字符串进行分词处理
        tokenized_query = self.language_processor.tokenize(query)
        # 使用BM25算法从数据源中获取与分词后的查询最相关的前topk个样本
        topk_samples = self.bm25.get_top_n(tokenized_query, self.data_source, n=topk)
        return topk_samples
