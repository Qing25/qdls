

from abc import abstractmethod, ABC

class Retriever(ABC):
    
    @abstractmethod
    def get_topk_samples(self, query, topk=5):
        """ 
            return topk samples according to query
        """
        raise NotImplementedError