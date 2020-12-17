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


# TODO what are good values for num_topics?
# TODO not all arguments are passable, this should be a class structure
def compute_text_vector(tokenized, num_topics=3, method="tfidf"):
    """
    Computing feature vectors for tokenized paper abstracts.
    """

    # TODO does this work only on pandas?

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


def test(test_path):
    """ Testout processing for debugging, etc. """
    corpus = io.load_json(test_path)
    corpus_with_token = io.preprocessing(
        corpus, lib="spacy", stemming=True, lemmatization=True
    )

    ipdb.set_trace()

    # TODO pass whole corpus or only token columns?
    compute_text_vector(corpus_with_token, num_topics=3, method="tfidf")

    # TODO debug jessicas try
    #    vis = pyLDAvis.gensim.prepare(
    #        lda_model, corpus_tfidf, dictionary=lda_model.id2word, mds="mmds"
    #    )


if __name__ == "__main__":
    test_path = "data/data_jmlr_vol17.json"
    test(test_path)
