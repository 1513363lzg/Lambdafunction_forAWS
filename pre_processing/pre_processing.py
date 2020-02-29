"""
import library needed for text preprocessing

@author: Nikki
"""


from pre_processing.word_embedding import WordEmbedding
from pre_processing.text_processing import TextProcessor



class PreProcessor():
    """
            Run the pre processing end to end

    """

    def __init__(self, padding_size=20, max_dictionary_size=500000):

        self.text_processor = TextProcessor()
        self.embedding = WordEmbedding(max_dictionary_size=max_dictionary_size)

        self.embedding.load_embedding_dictionary(self.embedding.dictionary_path)

        self.padding_size = padding_size

    def pre_process_text(self, text):
        """
        Run the pre processing end to end

        """

        cleaned_text = self.text_processor.clean_text(text)
        tokens = self.text_processor.tokenize_text(cleaned_text)

        embedding_indexs = self.embedding.replace_tokens_with_index(tokens)

        padded_index = self.pad_sequence(embedding_indexs)

        return padded_index

    def pad_sequence(self, input_squence):
        """
        Add 0 padding to a seqence

        """

        sequence = input_squence[-self.padding_size:]

        if len(sequence) < self.padding_size:

            pad_sequence = [0] * (self.padding_size - len(sequence))
            sequence = sequence + pad_sequence

        return sequence
