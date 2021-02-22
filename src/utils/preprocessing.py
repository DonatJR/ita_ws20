import gensim.parsing.preprocessing
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.helper import get_logger
from utils.config import PreprocessingLib, DimReduction, ClusteringMethod


class Preprocessing:
    def __init__(
        self,
        text,
        logger,
        config,
        datatype="abstract"
    ):
        self.__datatype = datatype
        self.__custom_stopwords = config.custom_stopwords
        self.__max_word_len = config.max_word_len
        self.__min_word_len = config.min_word_len
        self.__lemmatization = config.use_lemmatization
        self.__stemming = config.use_stemming
        self.__lib = config.lib
        self.__text = text
        self.__logger = logger

        if self.__stemming:
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
            self.__text, pd.Series
        ), "Please pass panda data series for text"

        self.__text.replace("", np.nan, inplace=True)

        num_nan = self.__text.isna().sum()
        self.__logger.info(
            "Dropping %d entries of corpus, due to nan ..." % num_nan)
        self.__text.dropna(inplace=True)
        self.__text = self.__text.reset_index(drop=True)

        if self.__lib == PreprocessingLib.SPACY:
            tokenized = self.__spacy_preprocess()
        elif self.__lib == PreprocessingLib.NLTK:
            tokenized = self.__nltk_preprocess()
        else:
            raise Exception("Invalid library choice!")

        tokenized = pd.Series(tokenized)
        df_text = pd.concat([self.__text, tokenized], axis=1)
        df_text.columns = ["abstract", "token"]

        return df_text

    def __spacy_preprocess(self):
        import en_core_web_sm
        import spacy

        def process_spacy(data):
            data = gensim.parsing.preprocessing.strip_non_alphanum(data)
            data = nlp(data)

            if self.__lemmatization:
                tokens = [doc.lemma_ for doc in data]
            else:
                tokens = [doc.text for doc in data]
            if self.__stemming:
                tokens = [self.stemmer.stem(token) for token in tokens]

            tokens = self.__remove_words(tokens, STOPWORDS)

            return tokens

        nlp = en_core_web_sm.load()

        STOPWORDS = spacy.lang.en.stop_words.STOP_WORDS
        STOPWORDS.update(self.__custom_stopwords)

        tokenized = self.__process(process_spacy)
        return tokenized

    def __nltk_preprocess(self):
        import nltk

        nltk.download("stopwords")
        nltk.download("punkt")
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        from nltk.tokenize import word_tokenize

        def process_nltk(data):
            data = gensim.parsing.preprocessing.strip_non_alphanum(
                data).lower()
            text_tokens = word_tokenize(data)

            if self.__stemming:
                text_tokens = [self.stemmer.stem(word) for word in text_tokens]
            if self.__lemmatization:
                lemmatizer = WordNetLemmatizer()
                text_tokens = [lemmatizer.lemmatize(
                    word) for word in text_tokens]

            tokens = self.__remove_words(text_tokens, STOPWORDS)
            return tokens

        STOPWORDS = set(stopwords.words("english"))
        STOPWORDS.update(self.__custom_stopwords)

        tokenized = self.__process(process_nltk)
        return tokenized

    def __remove_words(self, tokens, STOPWORDS):
        tokens = [word for word in tokens if word not in STOPWORDS]
        tokens = [word for word in tokens if not len(
            word) < self.__min_word_len]
        tokens = [word for word in tokens if not len(
            word) > self.__max_word_len]
        return tokens

    def __process(self, process):
        tokenized = []
        self.__logger.info("Starting tokenization ...")
        if self.__datatype == "abstract":
            for _, abstract in tqdm(enumerate(self.__text)):
                tokens = process(abstract)
                tokenized.append(tokens)
        elif self.__datatype == "keywords":
            for _, kword_list in tqdm(enumerate(self.__text)):
                token_kwords = []
                # NOTE kword does not mean a single word, but a specific combination
                for kword in kword_list:
                    tokens = process(kword)
                    token_kwords.append(tokens)
                tokenized.append(token_kwords)
        return tokenized
