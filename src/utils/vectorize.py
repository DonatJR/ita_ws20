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


# TODO docstr
def compute_tfidf(text, dictionary, return_bow=True):
    bow_corpus = [dictionary.doc2bow(token) for token in text]
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]

    if return_bow:
        return corpus_tfidf, bow_corpus
    else
        return corpus_tfidf

# TODO docstr
def lsi(corpus, dictionary, num_topics=3):
    """ """
    # train model
    lsi_model = models.LsiModel(
        corpus, id2word=dictionary, num_topics=num_topics
    )  
    # apply model to  document
    lsi_model[corpus[1]]  
    return lsi_model



# TODO this is kinda uselessly wrapped
# TODO docstr
def lda(corpus, dictionary, num_topics=3):

    lda_model = models.ldamodel.LdaModel(
        corpus=corpus,
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
    return lda_model
