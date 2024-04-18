import numpy as np
from typing import Union, List


class BasicTokenizer():
    def __init__(self, max_seq_length):
        self.char_dic = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:’'/\\|_@#$%ˆ&*˜‘+-=<>()[]{}\n"
        self.nb_char_dic = 70
        assert len(self.char_dic) == self.nb_char_dic
        self.max_seq_length = max_seq_length

    def tokenize(self, urls: Union[str, list]) -> np.array:
        if isinstance(urls, str):
            return self._tokenize(urls)
        elif isinstance(urls, list):
            return self._tokenize_batch(urls)
        else:
            raise ValueError("Input must be a string or a list of strings, but is a {}".format(type(urls)))

    def _tokenize(self, url: str) -> np.array:
        """
        Input : string of the url
        Output : numpy array (nb_char_dic, max_seq_length)
        """
        sparse_vector = np.zeros(( self.nb_char_dic, self.max_seq_length))
        nb_accepted_characters = 0
        for c in url:
            if c in self.char_dic:
                sparse_vector[self.char_dic.index(c), nb_accepted_characters] = 1
                nb_accepted_characters += 1
            elif c.lower() in self.char_dic:
                sparse_vector[self.char_dic.index(c.lower()), nb_accepted_characters] = 1
                nb_accepted_characters += 1
            if nb_accepted_characters == self.max_seq_length:
                break
        return sparse_vector
	
    def _tokenize_batch(self, urls: List[str]) -> np.array:
        """
        Input : list of urls
        Output : numpy array (len(urls), max_seq_length, nb_char_dic)
        """
        return np.array([self._tokenize(url) for url in urls])