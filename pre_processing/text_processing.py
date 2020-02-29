'''
all module needed for text preprocessing

@author: Nikki
'''
import sys
sys.path.append(r'../pre_processing/')
import re
from nltk.tokenize import TweetTokenizer

class TextProcessor():
    """
    Pre Processor class
    """
    def __init__(self):
        """
        setup the pre processor with URL regex and Twitter Pre Processor

        """

        p_1 = r'https?:\/\/[\S\/]*'
        p_2 = r'#[\S]*'
        p_3 = r'@[\S]*'
        p_4 = r'\s?rt'
        p_5 = r'\n+'
        p_6 = r'\s?[0-9]\s?'
        self.text_regex = f'({p_1})|({p_2})|({p_3})|({p_4})|({p_5})|({p_6})'
        self.tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True, preserve_case=False)

    def clean_text(self, input_text):
        """
        Clean input text

        """
        cleaned_text = re.sub(self.text_regex, '', input_text, flags=re.MULTILINE)

        return cleaned_text

    def tokenize_text(self, input_text):
        """
        Tokenize the input_Text with class tokenizer

        """
        tokens = self.tokenizer.tokenize(input_text)

        return tokens
