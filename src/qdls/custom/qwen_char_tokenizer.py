# -*- coding: utf-8 -*-
# @File    :   qwen_chat_tokenizer.py
# @Time    :   2024/06/26 16:03:04
# @Author  :   Qing 
# @Email   :   aqsz2526@outlook.com
######################### docstring ########################
'''
    自定义 Qwen2Tokenizer 对于中文词汇分词时分到 char 级别
    原理: bpe 时遇到中文字符停止合并
'''


import regex as re
from transformers import Qwen2Tokenizer



# Copied from transformers.models.gpt2.tokenization_gpt2.get_pairs
def get_pairs(word):
    """
    Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

class QwenCharTokenizer(Qwen2Tokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.chinese_char_range = (0x4E00, 0x9FFF)
        self.chinese_chars_ids = self._get_chinese_char_ids()

    def _get_chinese_char_ids(self):
        """ 找出 qwen 词表中所有的中文单字的 id"""
        R = set()
        for i in range(0, len(self)):
            text = self.decode([i])
            # if len(text) != 1:
                # print(i, text)
            if re.match(r'\p{Han}', text):
                if len(text) == 1:
                    R.add(i)
        return R

    
    def is_chinese_char(self, token):
        """判断 token 是否是中文字符"""
        token_id = self._convert_token_to_id(token)
        if token_id in self.chinese_chars_ids:
            return True 
        else:
            return False

    
    # Copied from transformers.models.gpt2.tokenization_gpt2.GPT2Tokenizer.bpe
    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram

            # Check if we are combining Chinese characters and stop if they are
            if self.is_chinese_char(first) or self.is_chinese_char(second):
                break

            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def _tokenize(self, text):
        """Tokenize a string."""
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )  
            # here are byte-level ascii characters 0-255

            # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" ")) 
            # 合并 byte 直到 byte pair不在词表中，bpe_tokens 就是合并后剩下的字符
        return bpe_tokens
    

if __name__ == '__main__':
    
    PATH = "/sshfs/pretrains/Qwen/Qwen2-0.5B-Instruct/"
    # 使用自定义的 CharTokenizer
    char_tokenizer = QwenCharTokenizer.from_pretrained(PATH)

    # 测试
    test_sentence = "醉春风"
    tokens = char_tokenizer.tokenize(test_sentence)
    token_ids = char_tokenizer.convert_tokens_to_ids(tokens)

    print("Tokens:", tokens)
    print("Token IDs:", token_ids)
    print()