"""
This contains all relevant I/O operations.
"""

import json

import ipdb
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_json(fpath, use_title=False):
    """ Loading the manual json files prepared for the time being  """
    with open(fpath) as f:
        data = json.load(f)
        data_df = pd.json_normalize(data["papers"])
        corpus = data_df["abstract"]
        if use_title:
            corpus = data_df["title"] + " " + corpus

    return corpus


# TODO refactor into class
def preprocessing(
    text,
    lib="gensim",
    stemming=False,
    lemmatization=False,
    min_word_len=2,
    max_word_len=15,
):
    """
    Perform tokenization, stop word removal and optionally stemming, lemmatization.
    These steps are crucial to get meaningful clusters.

    args:
    ---
    text [pd.Series]: pandas series of paper abstracts
    lib [str]: library to use for processing. options: spacy, nltk

    returns:
    ---
    tokenized [pd.DataFrame]: Data frame with both abstracts and related tokens
    """
    import gensim.parsing.preprocessing

    def spacy_preprocess(
        text,
        stemming=False,
        lemmatization=False,
        min_word_len=2,
        max_word_len=15,
    ):

        import en_core_web_sm

        nlp = en_core_web_sm.load()
        all_stopwords = nlp.Defaults.stop_words

        tokenized = []
        print("Starting tokenization ...")
        for _, abstract in tqdm(enumerate(text)):
            abstract = gensim.parsing.preprocessing.strip_non_alphanum(abstract)
            abstract = nlp(abstract)
            tokens = []

            for doc in abstract:
                if lemmatization:
                    token = doc.lemma_
                else:
                    token = doc.text
                if stemming:
                    token = stemmer.stem(token)
                if token not in all_stopwords:
                    tokens.append(token)

            #            if lemmatization:
            #                tokens = [doc.lemma_ for doc in abstract]
            #            else:
            #                tokens = [doc.text for doc in abstract]
            #            if stemming:
            #                tokens = [stemmer.stem(token) for token in tokens]

            tokens = [word for word in tokens if not len(word) < min_word_len]
            tokens = [word for word in tokens if not len(word) > max_word_len]
            tokenized.append(tokens)
        return tokenized

    def nltk_preprocess(
        text,
        stemming=False,
        lemmatization=False,
        min_word_len=2,
        max_word_len=15,
    ):
        import nltk

        nltk.download("stopwords")
        nltk.download("punkt")
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        from nltk.tokenize import word_tokenize

        STOPWORDS = set(stopwords.words("english"))

        tokenized = []
        print("Starting tokenization ...")
        for _, abstract in tqdm(enumerate(text)):
            abstract = gensim.parsing.preprocessing.strip_non_alphanum(abstract)
            text_tokens = word_tokenize(abstract)
            if stemming:
                text_tokens = [stemmer.stem(word) for word in text_tokens]
            if lemmatization:
                lemmatizer = WordNetLemmatizer()
                text_tokens = [lemmatizer.lemmatize(word) for word in text_tokens]

            tokens = [word for word in text_tokens if word not in STOPWORDS]
            tokens = [word for word in tokens if not len(word) < min_word_len]
            tokens = [word for word in tokens if not len(word) > max_word_len]
            tokenized.append(tokens)
        return tokenized

    assert isinstance(text, pd.Series), "Please pass panda data series for text"
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
        )

    elif lib == "nltk":
        tokenized = nltk_preprocess(
            text,
            stemming=stemming,
            lemmatization=lemmatization,
            min_word_len=2,
            max_word_len=15,
        )
    else:
        raise Exception("Invalid library choice!")

    tokenized = pd.Series(tokenized)
    df_text = pd.concat([text, tokenized], axis=1)
    df_text.columns = ["abstract", "token"]

    return df_text
