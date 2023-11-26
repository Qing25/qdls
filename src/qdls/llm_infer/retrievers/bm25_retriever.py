
import jieba 
from rank_bm25 import BM25Okapi
from qdls.gql.cypher.utils.parse import split_query # TODO: unified interface for cypher sparql, etc

from .base_retriever import Retriever

class BM25Retriever(Retriever):
    def __init__(self, data_source, key=None, lang='en') -> None:
        """ 
            data_source: list of samples
            key: key to retrieve
        """
        self.data_source = data_source
        assert key is not None, "key is not specified"
        if type(key) is str:
            self.items4retrieval = [ sample[key] for sample in data_source]
        elif callable(key):
            # 以构建更多样的检索形式，而非简单的使用sample[key]作为检索项
            self.items4retrieval = [ key(sample) for sample in data_source]
        else:
            raise Exception(f"key should be dict key or callable function")

        self.lang = lang
        if lang == 'en':
            tokenized_corpus = [doc.split(" ") for doc in self.items4retrieval]
        elif lang == 'zh':
            tokenized_corpus = [ list(jieba.cut(doc)) for doc in self.items4retrieval]
        elif lang == 'cypher':
            tokenized_corpus = [ split_query(doc) for doc in self.items4retrieval]
        else:
            raise ValueError(f"lang must be 'en' or 'zh', but got {lang}")
        self.bm25 = BM25Okapi(tokenized_corpus)


    def get_topk_samples(self, query, topk=5):
        """ 将输入query分词，返回self.topk个 sample """
        if self.lang == "en":
            tokenized_query = query.split(" ")
        elif self.lang == 'zh':
            tokenized_query = list(jieba.cut(query))
        elif self.lang == 'cypher':
            tokenized_query = split_query(query)
        else:
            raise ValueError(f"lang must be 'en' or 'zh', but got {self.lang}")
        topk_samples = self.bm25.get_top_n(tokenized_query, self.data_source, n=topk)
        return topk_samples
    
