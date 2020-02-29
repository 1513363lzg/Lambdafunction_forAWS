
"""
import all the modlue needed

@author: Nikki
"""
import sys
sys.path.append(r'../pre_processing/')
import zipfile
from os.path import abspath
from os.path import join
from os.path import dirname



class WordEmbedding():
    """

    Class to load Embedding dictionary and process token lists

    """

    def __init__(self, max_dictionary_size=50000):
        self.embedding_dictionary = {}
        self.max_dictionary_size = max_dictionary_size
        self.dictionary_path = join(
            dirname(abspath(__file__)),
            "..","resources", "golve_index.txt")

    def load_embedding_dictionary(self, file_path):
        """

        Loading a dictionary from GloVe Embeddings
        Values are the index of the embedding

        """
        self.embedding_dictionary = {}
        embeddings = []

        if ".zip/" in file_path:
            archive_path = abspath(file_path)
            split = archive_path.split(".zip/")
            archive_path = split[0] + ".zip"
            path_inside = split[1]
            archive = zipfile.ZipFile(archive_path, "r")
            embeddings = archive.read(path_inside).decode("utf-8").split("\n")

        else:
            embeddings = open(file_path, "r", encoding="utf-8").read().split("\n")

        for index, row in enumerate(embeddings):
            split = row.split(" ")
            if index >= self.max_dictionary_size:
                return

            self.embedding_dictionary[split[0]] = index

    def replace_tokens_with_index(self, token_list):
        """
        Replace each token with it's index in the embedding dictionary

        """

        index_list = []

        for token in token_list:
            token_index = self.embedding_dictionary.get(token, 1)
            index_list.append(token_index)

        return index_list
         