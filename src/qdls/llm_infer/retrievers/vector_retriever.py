
import os 
try:
    import faiss
except ImportError:
    raise ImportError("Please install faiss by running `pip install faiss-gpu`")
from loguru import logger
try:
    from sentence_transformers import SentenceTransformer  
    from langchain_huggingface import HuggingFaceEmbeddings  # requires sentence-transformers 
except ImportError:
    raise ImportError("Please install langchain-huggingface by running `pip install langchain-huggingface`")

try:
    from langchain_chroma import Chroma 
    from langchain_core.documents import Document
except ImportError:
    logger.warning("Please install langchain-chroma by running `pip install langchain-chroma`\nOr use VectorRetriever backended by faiss")

from qdls.llm_infer.retrievers.base_retriever import Retriever

class VectorRetriever(Retriever):
    def __init__(self, data_source, key=None, embedding_model_path=None, cache_file=None, **kwargs):
        """
        Args:
            data_source: List[Dict] or List[Object].
            key: str or callable function. fetch data from each sample in data_source.
            embedding_model_path: model used to encode text. 
            cache_file: file path to save the cached Vector/Index

        """
        self.data_source = data_source
        assert key is not None, "key is not specified"
        self.cache_file = cache_file      
        if type(key) is str:
            self.items4retrieval = [ sample[key] for sample in data_source]
        elif callable(key):
            # 以构建更多样的检索形式，而非简单的使用sample[key]作为检索项
            self.items4retrieval = [ key(sample) for sample in data_source]
        else:
            raise Exception(f"key should be dict key or callable function")
        self.key = key 
        self.model = self._init_encoding_model(embedding_model_path)

        # 
        self._handle_cache( self.encode_fn if hasattr(self, 'encode_fn') else None)

        # default args 
        self.bsz = kwargs.get('bsz', 128)
    

    def _init_encoding_model(self, model_name_or_path=None):
        if model_name_or_path is None:
            model_name_or_path = "/sshfs/pretrains/sentence-transformers/all-MiniLM-L6-v2"
            # option models: /sshfs/pretrains/Salesforce/SFR-Embedding-Mistral
        model = SentenceTransformer(model_name_or_path, trust_remote_code=True)
        return model 
            
    def _handle_cache(self, encode_fn=None):
        os.makedirs("cached_index", exist_ok=True)
        if self.cache_file is not None:
            cache_file = os.path.join(".", "cached_index",f"{self.cache_file}_index.bin")
        else:
            cache_file = f"{ self.key if type(self.key) is str else self.key.__name__ }_index.bin"
            cache_file = os.path.join(".", "cached_index", cache_file)

        if encode_fn is None:
            def encode_fn(model, text, bsz=1):
                return model.encode(text, batch_size=bsz, show_progress_bar=True)

        if os.path.exists(cache_file):
            self.index = faiss.read_index(cache_file)
            logger.info(f"index loaded from {cache_file}")
        else:
            # vectors = self.model.encode(self.items4retrieval, batch_size=128, show_progress_bar=True)
            vectors = encode_fn(self.model, self.items4retrieval)
            self.index = faiss.IndexFlatIP(vectors.shape[-1])
            logger.info(f"index built, dimension: {self.index.d} vs {vectors.shape}, ntotal: {self.index.ntotal}")
            self.index.add(vectors)
            faiss.write_index(self.index, cache_file)
            logger.info(f"index built and saved to {cache_file}")

    
    def get_topk_samples(self, query, topk=5):
        """ 将输入query分词，返回self.topk个 sample """
        if type(query) is str:
            query_vector = self.model.encode(query)
            D, I = self.index.search(query_vector.reshape(1,-1), topk)
        else:
            raise Exception("query should be str")
        topk_samples = [ self.data_source[i] for i in I[0]]
        return topk_samples


class VectorRetrieverInstruct(VectorRetriever):
    """ 
        一些模型是带着 instruction 训练的
        ref: https://huggingface.co/Salesforce/SFR-Embedding-Mistral
    """
    def encode_fn(self, model, text, bsz):
        task_description = 'Given a web search query, retrieve relevant passages that answer the query'
        text = [f'Instruct: {task_description}\nQuery: {query}' for query in text]
        return model.encode(text, batch_size=bsz, show_progress_bar=True)


class VectorRetrieverLangChain(VectorRetriever):

    def __init__(self, data_source, key=None, embedding_model_path=None, cache_file=None):
        """
        Args:
            data_source: List[Dict] or List[Object].
            key: str or callable function. fetch data from each sample in data_source.
            embedding_model_path: model used to encode text. 
            cache_file: file path to save the cached Vector/Index

        """
        self.data_source = data_source
        assert key is not None, "key is not specified"
        self.cache_file = cache_file
        if type(key) is str:
            # use metadata property to store the original sample
            self.items4retrieval = [Document(sample[key], metadata=sample) for sample in data_source]
        elif callable(key):
            # 以构建更多样的检索形式，而非简单的使用sample[key]作为检索项
            self.items4retrieval = [Document(key(sample), metadata=sample) for sample in data_source]
        else:
            raise Exception(f"key should be dict key or callable function")
        self.key = key 
        self.model = self._init_encoding_model(embedding_model_path)
        self._handle_cache()


    def _init_encoding_model(self, model_name_or_path=None):
        if model_name_or_path is None:
            model_name_or_path = "/sshfs/pretrains/BAAI/bge-large-zh-v1.5"
        model_kwargs = {'device': 'cuda',  }
        encode_kwargs = {'normalize_embeddings': True}    
        model = HuggingFaceEmbeddings(
                    model_name=model_name_or_path,
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs,
                    show_progress=True,
                )
        return model
    
    def _handle_cache(self):
        os.makedirs("cached_index", exist_ok=True)
        if self.cache_file is not None:
            persist_directory = self.cache_file + '_index'
        else:
            persist_directory = f"{ self.key if type(self.key) is str else self.key.__name__ }_index"
        persist_directory = os.path.join(".", "cached_index", persist_directory)
        

        collection_name = "default"
        
        if os.path.exists(persist_directory):
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=self.model,
                persist_directory=persist_directory,
            )
            logger.info(f"Vectorstore loaded from {persist_directory}")
        else:
            chunks = self.items4retrieval
            vectorstore = Chroma.from_documents(
                documents=chunks, 
                collection_name=collection_name,
                embedding=self.model,
                persist_directory=persist_directory,
            )
            logger.info(f"Vectorstore built and saved to {persist_directory}") 
        
        self.retriever = vectorstore

    def get_topk_samples(self, query:str, topk=5):

        r = self.retriever.similarity_search(query, k=topk)
        return [ doc.metadata for doc in r]


if __name__ == '__main__':
    data = [
        {"id": 1, "text": "今天天气不错"},
        {"id": 2, "text": "今天天气不好"},
        {"id": 3, "text": "今天天气很差"},
    ]
    # r = VectorRetrieverLangChain(data_source=data, key="text")
    # r = VectorRetriever(data_source=data, key="text", embedding_model_path="/sshfs/pretrains/Salesforce/SFR-Embedding-Mistral", bsz=2)
    r = VectorRetriever(data_source=data, key="text", embedding_model_path="/sshfs/pretrains/Alibaba-NLP/gte-Qwen2-7B-instruct", bsz=2)
    query = "坏天气"
    print(r.get_topk_samples(query, topk=1))