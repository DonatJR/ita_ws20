#! /usr/bin/env python3

import os
import argparse
import ipdb
from pathlib import Path

import utils.helper as helper
import utils.data as io
import utils.vectorize
from gensim import corpora

import utils.visu as visu
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

"""
Load our collected papers as a json file, preprocess, perform vectorization 
and cluster with selected algorithm. 

We will configure this via yaml configs later.
"""

# TODO think about saving results


def get_parser():
    """ Parser to configure main script via command line """
    parser = ArgumentParser(description="scientific paper clustering")
    parser.add_argument("--option1", type=str, help="Example option")
    parser.add_argument(
        "--option2",
        type=int,
        default=1,
        help="Another option with typing",
    )
    return parser


def save_results(results):
    raise NotImplementedError()


# NOTE CHEN: this is only a prototype
# NOTE CHEN: I like to configure my experiments via yaml files
# The run.py script can be used to parse the configuration files to this file
if __name__ == "__main__":

    global args, logger
    args = get_parser().parse_args()
    logger = helper.get_logger(args.save_folder)
    logger.info(args)
    logger.info("=> Starting new experiment ...")

    test_path = Path("data/data_jmlr_vol13-21.json")
    corpus = io.load_json(test_path)
    corpus = io.preprocessing(corpus, lib="spacy", stemming=True, lemmatization=True)

    # TODO JESSICA: further filter and clean data by using functions such as filter_extremes (remove all tokens that are less frequent or more frequent than a number),
    # filter_n_most_frequent(filter out the ‘remove_n’ most frequent tokens),
    # merge_with (to merge multiple dictionaries)

    ipdb.set_trace()

    dictionary = corpora.Dictionary(corpus["token"])
    corpus_tfidf, bow_corpus = compute_tfidf(text, dictionary, return_bow=True)

    lsi_tfidf = lsi(corpus_tfidf, dictionary)
    lsi_bow = lsi(bow_corpus, dictionary)
    lda_tfidf = lda(corpus_tfidf, dictionary, num_topics=3)
    lda_bow = lda(bow_corpus, dictionary, num_topics=3)

    # TODO Use sklearn clustering to cluster these

    # TODO debug jessicas try
    #    vis = pyLDAvis.gensim.prepare(
    #        lda_model, corpus_tfidf, dictionary=lda_model.id2word, mds="mmds"
    #    )

    # Visualize

#    print(dictionary.token2id)  # token -> tokenId.
#    print(dictionary.dfs)  # token_id -> how many documents contain this token.
#    print(BoW_corpus)  # list of (token_id, token_count)
#
#    for doc in corpus_tfidf:
#        print(doc)
#
#    lsi_tfidf.print_topics()
#    lsi_bow.print_topics()
#    lda_tfidf.print_topics()
#    lda_bow.print_topics()
#
#    visu.plot_2d_space(bow_corpus, lsi_bow)
#    visu.plot_2d_space(corpus_tfidf, lsi_tfidf)
#    visu.plot_2d_space(corpus_tfidf, lda_model)
#    visu.plot_2d_space(corpus_tfidf, lda_model, use_tsne=True)
