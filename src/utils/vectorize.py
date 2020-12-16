#! /usr/bin/env python3

"""
This collects routines for vectorizing text data, so we can perform similarity 
and clustering computations on it.

author: Chen
"""

# TODO right now we import only when needed
# this may not PEP8 conform I believe, so resolve dependencies!

import json
import ipdb

from gensim import corpora, models
from sklearn.manifold import TSNE
import pyLDAvis.gensim

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import data as io
import visu


# TODO define vectorizer class to get all different methods and better data flow
# TODO what are good values for num_topics?
# TODO not all arguments are passable, this should be a class structure
def compute_text_vector(tokenized, num_topics=3, method="tfidf"):
    """
    Computing feature vectors for tokenized paper abstracts.
    """

    dictionary = corpora.Dictionary(tokenized)
    BoW_corpus = [dictionary.doc2bow(text) for text in tokenized]
    tfidf = models.TfidfModel(BoW_corpus)
    corpus_tfidf = tfidf[BoW_corpus]

    #    if method == "tfidf":
    lsi_tfidf = models.LsiModel(
        corpus_tfidf, id2word=dictionary, num_topics=num_topics
    )  # train model
    lsi_tfidf[corpus_tfidf[1]]  # apply model to  document

    #        return (corpus_tfidf, lsi_tfidf)

    #    elif method == "bow":
    lsi_bow = models.LsiModel(BoW_corpus, id2word=dictionary, num_topics=num_topics)
    lsi_bow[corpus_tfidf[1]]  # apply model to  document

    #        return (BoW_corpus, lsi_bow)

    #    elif method == "lda":
    # TODO this has so many arguments to pass ...
    lda_model = models.ldamodel.LdaModel(
        corpus=corpus_tfidf,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=100,
        update_every=1,
        chunksize=10,
        passes=10,
        alpha="symmetric",
        iterations=100,
        per_word_topics=True,
    )

    #        return (corpus_tfidf, lda_model)

    # TODO: furthur filter and clean data by using functions such as filter_extremes (remove all tokens that are less frequent or more frequent than a number),
    # filter_n_most_frequent(filter out the ‘remove_n’ most frequent tokens),
    # merge_with (to merge multiple dictionaries)

    ipdb.set_trace()
    # TODO look at print statements and get insight
    # TODO are these features or similarities?

    # Corpus
    print(dictionary.token2id)  # token -> tokenId.
    print(dictionary.dfs)  # token_id -> how many documents contain this token.
    print(BoW_corpus)  # list of (token_id, token_count)

    for doc in corpus_tfidf:
        print(doc)

    lsi_tfidf.print_topics()
    lsi_bow.print_topics()
    lda_model.print_topics()

    visu.plot_2d_space(BoW_corpus, lsi_bow)
    visu.plot_2d_space(corpus_tfidf, lsi_tfidf)
    visu.plot_2d_space(corpus_tfidf, lda_model)
    visu.plot_2d_space(corpus_tfidf, lda_model, use_tsne=True)


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
    lib [str]: library to use for processing. options: gensim, spacy, nltk

    returns:
    ---
    tokenized [pd.Series]: Tokenized paper abstracts
    """

    import gensim.parsing.preprocessing

    if stemming:
        from nltk.stem.snowball import SnowballStemmer

        stemmer = SnowballStemmer(language="english")

    if lib == "gensim":
        from gensim.utils import simple_preprocess

        if lemmatization:
            from gensim.utils import lemmatize
            import pattern3

        tokenized = []
        for abstract in text:
            no_numbers = gensim.parsing.preprocessing.strip_non_alphanum(abstract)
            clean = gensim.parsing.preprocessing.remove_stopwords(no_numbers)
            token = simple_preprocess(clean, min_len=min_word_len, max_len=max_word_len)
            if lemmatization:
                token = lemmatize(token)
            if stemming:
                token = stemmer.stem(token)
            tokenized.append(token)

    elif lib == "spacy":
        import spacy

        nlp = spacy.load("en_core_web_sm")
        all_stopwords = nlp.Defaults.stop_words

        tokenized = []
        for abstract in text:
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
                if not token in all_stopwords:
                    tokens.append(token)
            tokens = [word for word in tokens if not len(word) < min_word_len]
            tokens = [word for word in tokens if not len(word) > max_word_len]
            tokenized.append(tokens)

    elif lib == "nltk":
        import nltk

        nltk.download("stopwords")
        nltk.download("punkt")
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize

        STOPWORDS = set(stopwords.words("english"))

        tokenized = []
        for abstract in text:
            abstract = gensim.parsing.preprocessing.strip_non_alphanum(abstract)
            text_tokens = word_tokenize(abstract)
            if stemming:
                text_tokens = [stemmer.stem(word) for word in text_tokens]
            if lemmatization:
                from nltk.stem import WordNetLemmatizer

                lemmatizer = WordNetLemmatizer()

                text_tokens = [lemmatizer.lemmatize(word) for word in text_tokens]

            tokens = [word for word in text_tokens if not word in STOPWORDS]
            tokens = [word for word in tokens if not len(word) < min_word_len]
            tokens = [word for word in tokens if not len(word) > max_word_len]
            tokenized.append(tokens)

    return tokenized


def test(test_path):
    """ Testout processing for debugging, etc. """
    corpus = io.load_json(test_path)
    tokenized = preprocessing(corpus, lib="spacy")

    ipdb.set_trace()

    # TODO Jessica use pandas frames
    # TODO convert list of lists to new column in corpus
    compute_text_vector(tokenized, num_topics=3, method="tfidf")

    # TODO debug jessicas try
    #    vis = pyLDAvis.gensim.prepare(
    #        lda_model, corpus_tfidf, dictionary=lda_model.id2word, mds="mmds"
    #    )


if __name__ == "__main__":
    test_path = "data/manual_datasource.json"
    test(test_path)
