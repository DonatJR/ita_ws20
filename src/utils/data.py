"""
This contains all relevant I/O operations.
"""

import json

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


def preprocessing(
    text,
    lib="gensim",
    stemming=False,
    lemmatization=False,
    min_word_len=2,
    max_word_len=15,
    custom_stopwords=[],
    datatype="abstract",
):
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
    import gensim.parsing.preprocessing

    def spacy_preprocess(
        text,
        datatype,
        stemming=False,
        lemmatization=False,
        min_word_len=2,
        max_word_len=15,
        custom_stopwords=[],
    ):
        def process(data):
            data = gensim.parsing.preprocessing.strip_non_alphanum(data)
            data = nlp(data)
            tokens = []

            if lemmatization:
                tokens = [doc.lemma_ for doc in data]
            else:
                tokens = [doc.text for doc in data]
            if stemming:
                tokens = [stemmer.stem(token) for token in tokens]

            tokens = [word for word in tokens if word not in STOPWORDS]
            tokens = [word for word in tokens if not len(word) < min_word_len]
            tokens = [word for word in tokens if not len(word) > max_word_len]
            return tokens

        # NOTE this is the only import that works for Jessica
        import en_core_web_sm
        import spacy

        nlp = en_core_web_sm.load()
        #        nlp = spacy.load("en_core_web_sm")

        # all_stopwords = nlp.Defaults.stop_words
        STOPWORDS = spacy.lang.en.stop_words.STOP_WORDS
        STOPWORDS.update(custom_stopwords)

        tokenized = []
        print("Starting tokenization ...")
        if datatype == "abstract":
            for _, abstract in tqdm(enumerate(text)):
                tokens = process(abstract)
                tokenized.append(tokens)
        elif datatype == "keywords":
            for _, kword_list in tqdm(enumerate(text)):
                token_kwords = []
                # NOTE kword does not mean a single word, but a specific combination
                for kword in kword_list:
                    tokens = process(kword)
                    token_kwords.append(tokens)
                tokenized.append(token_kwords)
        return tokenized

    def nltk_preprocess(
        text,
        datatype,
        stemming=False,
        lemmatization=False,
        min_word_len=2,
        max_word_len=15,
        custom_stopwords=[],
    ):
        import nltk

        nltk.download("stopwords")
        nltk.download("punkt")
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        from nltk.tokenize import word_tokenize

        STOPWORDS = set(stopwords.words("english"))
        STOPWORDS.update(custom_stopwords)

        def process(data):
            data = gensim.parsing.preprocessing.strip_non_alphanum(data)
            text_tokens = word_tokenize(data)

            if stemming:
                text_tokens = [stemmer.stem(word) for word in text_tokens]
            if lemmatization:
                lemmatizer = WordNetLemmatizer()
                text_tokens = [lemmatizer.lemmatize(
                    word) for word in text_tokens]

            tokens = [word for word in text_tokens if word not in STOPWORDS]
            tokens = [word for word in tokens if not len(word) < min_word_len]
            tokens = [word for word in tokens if not len(word) > max_word_len]
            return tokens

        tokenized = []
        print("Starting tokenization ...")
        print("Starting tokenization ...")
        if datatype == "abstract":
            for _, abstract in tqdm(enumerate(text)):
                tokens = process(abstract)
                tokenized.append(tokens)
        elif datatype == "keywords":
            for _, kword_list in tqdm(enumerate(text)):
                token_kwords = []
                # NOTE kword does not mean a single word, but a specific combination
                for kword in kword_list:
                    tokens = process(kword)
                    token_kwords.append(tokens)
                tokenized.append(token_kwords)
        return tokenized

    assert isinstance(
        text, pd.Series), "Please pass panda data series for text"
    text.replace("", np.nan, inplace=True)
    num_nan = text.isna().sum()
    print("Dropping %d entries of corpus, due to nan ..." % num_nan)
    text.dropna(inplace=True)
    text = text.reset_index(drop=True)

    if stemming:
        from nltk.stem.snowball import SnowballStemmer

        stemmer = SnowballStemmer(language="english")

    if lib == "spacy":
        tokenized = spacy_preprocess(
            text,
            stemming=stemming,
            lemmatization=lemmatization,
            min_word_len=2,
            max_word_len=15,
            custom_stopwords=custom_stopwords,
            datatype=datatype,
        )

    elif lib == "nltk":
        tokenized = nltk_preprocess(
            text,
            stemming=stemming,
            lemmatization=lemmatization,
            min_word_len=2,
            max_word_len=15,
            custom_stopwords=custom_stopwords,
            datatype=datatype,
        )
    else:
        raise Exception("Invalid library choice!")

    tokenized = pd.Series(tokenized)
    df_text = pd.concat([text, tokenized], axis=1)
    df_text.columns = ["abstract", "token"]

    return df_text
