"""
This contains all relevant I/O operations.
"""

import json

import gensim.parsing.preprocessing
import ipdb
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_json(fpath, return_data="abstract", append_title=False):
    """ Loading the gathered json files """
    with open(fpath) as f:
        data = json.load(f)
        data_df = pd.json_normalize(data["papers"])
        if return_data == "abstract":
            corpus = data_df["abstract"]
            if append_title:
                corpus = data_df["title"] + " " + corpus
        elif return_data == "keywords":
            corpus = data_df["keywords"]

    return corpus


class Preprocessing:
    def __init__(
        self,
        text,
        lib="gensim",
        stemming=False,
        lemmatization=False,
        min_word_len=2,
        max_word_len=15,
        custom_stopwords=[],
        datatype="abstract",
    ):
        self.datatype = datatype
        self.custom_stopwords = custom_stopwords
        self.max_word_len = max_word_len
        self.min_word_len = min_word_len
        self.lemmatization = lemmatization
        self.stemming = stemming
        self.lib = lib
        self.text = text

        if self.stemming:
            from nltk.stem.snowball import SnowballStemmer

            self.stemmer = SnowballStemmer(language="english")

    def apply_preprocessing(self):
        """
        Perform tokenization, stop word removal and optionally stemming, lemmatization.
        These steps are crucial to get meaningful clusters.

        args:
        ---
        text [pd.Series]: pandas series of paper abstracts or key_words
        lib [str]: library to use for processing. options: spacy, nltk

        returns:
        ---
        tokenized [pd.DataFrame]: Data frame with both abstracts and related tokens
        """
        assert isinstance(
            self.text, pd.Series
        ), "Please pass panda data series for text"
        self.text.replace("", np.nan, inplace=True)

        num_nan = self.text.isna().sum()
        print("Dropping %d entries of corpus, due to nan ..." % num_nan)
        self.text.dropna(inplace=True)
        self.text = self.text.reset_index(drop=True)

        if self.lib == "spacy":
            tokenized = self.spacy_preprocess()

        elif self.lib == "nltk":
            tokenized = self.nltk_preprocess()
        else:
            raise Exception("Invalid library choice!")

        tokenized = pd.Series(tokenized)
        df_text = pd.concat([self.text, tokenized], axis=1)
        df_text.columns = ["abstract", "token"]

        return df_text

    def spacy_preprocess(self):
        import en_core_web_sm
        import spacy

        def process_spacy(data):
            data = gensim.parsing.preprocessing.strip_non_alphanum(data)
            data = nlp(data)

            if self.lemmatization:
                tokens = [doc.lemma_ for doc in data]
            else:
                tokens = [doc.text for doc in data]
            if self.stemming:
                tokens = [self.stemmer.stem(token) for token in tokens]

            tokens = self.remove_words(tokens, STOPWORDS)

            return tokens

        nlp = en_core_web_sm.load()

        STOPWORDS = spacy.lang.en.stop_words.STOP_WORDS
        STOPWORDS.update(self.custom_stopwords)

        tokenized = self.process(process_spacy)
        return tokenized

    def nltk_preprocess(self):
        import nltk

        nltk.download("stopwords")
        nltk.download("punkt")
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        from nltk.tokenize import word_tokenize

        def process_nltk(data):
            data = gensim.parsing.preprocessing.strip_non_alphanum(data).lower()
            text_tokens = word_tokenize(data)

            if self.stemming:
                text_tokens = [self.stemmer.stem(word) for word in text_tokens]
            if self.lemmatization:
                lemmatizer = WordNetLemmatizer()
                text_tokens = [lemmatizer.lemmatize(word) for word in text_tokens]

            tokens = self.remove_words(text_tokens, STOPWORDS)
            return tokens

        STOPWORDS = set(stopwords.words("english"))
        STOPWORDS.update(self.custom_stopwords)

        tokenized = self.process(process_nltk)
        return tokenized

    def remove_words(self, tokens, STOPWORDS):
        tokens = [word for word in tokens if word not in STOPWORDS]
        tokens = [word for word in tokens if not len(word) < self.min_word_len]
        tokens = [word for word in tokens if not len(word) > self.max_word_len]
        return tokens

    def process(self, process):
        tokenized = []
        print("Starting tokenization ...")
        if self.datatype == "abstract":
            for _, abstract in tqdm(enumerate(self.text)):
                tokens = process(abstract)
                tokenized.append(tokens)
        elif self.datatype == "keywords":
            for _, kword_list in tqdm(enumerate(self.text)):
                token_kwords = []
                # NOTE kword does not mean a single word, but a specific combination
                for kword in kword_list:
                    tokens = process(kword)
                    token_kwords.append(tokens)
                tokenized.append(token_kwords)
        return tokenized
