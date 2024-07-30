
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
    def __init__(self, data_source, key=None, embedding_model_path=None):
        self.data_source = data_source
        assert key is not None, "key is not specified"
        if type(key) is str:
            self.items4retrieval = [ sample[key] for sample in data_source]
        elif callable(key):
            # 以构建更多样的检索形式，而非简单的使用sample[key]作为检索项
            self.items4retrieval = [ key(sample) for sample in data_source]
        else:
            raise Exception(f"key should be dict key or callable function")
        self.key = key 
        self.model = self._init_encoding_model(embedding_model_path)
        self._build_index()
    

    def _init_encoding_model(self, model_name_or_path=None):
        if model_name_or_path is None:
            model_name_or_path = "/sshfs/pretrains/sentence-transformers/all-MiniLM-L6-v2"
        model = SentenceTransformer(model_name_or_path)
        return model 
            
    def _build_index(self):
        cache_file = f"{ self.key if type(self.key) is str else self.key.__name__ }_index.bin"

        if os.path.exists(cache_file):
            self.index = faiss.read_index(cache_file)
            print(f"index loaded from {cache_file}")
        else:
            self.index = faiss.IndexFlatIP(384)
            vectors = self.model.encode(self.items4retrieval, batch_size=128, show_progress_bar=True)
            self.index.add(vectors)
            faiss.write_index(self.index, cache_file)
            print(f"index built and saved to {cache_file}")

    
    def get_topk_samples(self, query_vector, topk=5):
        """ 将输入query分词，返回self.topk个 sample """
     
        D, I = self.index.search(query_vector.reshape(1,-1), topk)
        topk_samples = [ self.data_source[i] for i in I[0]]
        return topk_samples
    
class VectorRetrieverLangChain(VectorRetriever):

    def __init__(self, data_source, key=None, embedding_model_path=None):
        self.data_source = data_source
        assert key is not None, "key is not specified"
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
        self._build_index()


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
    
    def _build_index(self):
        
        persist_directory = f"{ self.key if type(self.key) is str else self.key.__name__ }_index"
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

    def get_topk_samples(self, query, topk=5):

        r = self.retriever.similarity_search(query, k=topk)
        return [ doc.metadata for doc in r]


if __name__ == '__main__':
    data = [
        {"id": 1, "text": "今天天气不错"},
        {"id": 2, "text": "今天天气不好"},
        {"id": 3, "text": "今天天气很差"},
    ]
    r = VectorRetrieverLangChain(data_source=data, key="text")
    query = "坏天气"
    print(r.get_topk_samples(query, topk=1))