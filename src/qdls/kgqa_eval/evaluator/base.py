# -*- coding: utf-8 -*-
# @File    :   base.py
# @Time    :   2023/11/08 13:50:38
# @Author  :   Qing 
# @Email   :   aqsz2526@outlook.com
######################### docstring ########################
'''
'''


####################### imports ###########################

from abc import ABC, abstractmethod

class BaseEvalutor(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def normalize_sample(self, sample):
        """ standardize the sample to a dict with keys 
            sample_id

        IMPORTANT: values should not be dict, or it will be extremely slow
        """
        return sample

    @abstractmethod
    def evaluate(self):
        """ 

        """
        raise NotImplementedError

    
 