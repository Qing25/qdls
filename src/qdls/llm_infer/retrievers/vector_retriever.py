
import os 
import faiss
from sentence_transformers import SentenceTransformer 

from .base_retriever import Retriever

class VectorRetriever(Retriever):
    def __init__(self, data_source, key=None):
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
        self.model = self._init_encoding_model()
        self._build_index()
    

    def _init_encoding_model(self):
        model = SentenceTransformer("/sshfs/pretrains/sentence-transformers/all-MiniLM-L6-v2")
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