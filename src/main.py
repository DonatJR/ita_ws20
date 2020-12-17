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
# TODO think about configuration


def get_parser():
    """ Parser to configure main script via command line """
    parser = argparse.ArgumentParser(description="scientific paper clustering")
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


# TODO more algorithms
# TODO clearer options
# TODO test how to pass data
# TODO refactor
# TODO can we run this on the vectors from other libs?
def sklearn_clustering(data, n_components=None):
    """ From the online doc: https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import Normalizer
    from sklearn import metrics
    from sklearn.cluster import KMeans

    ipdb.set_trace()

    vectorizer = TfidfVectorizer(
        max_df=0.5,
        max_features=opts.n_features,
        min_df=2,
        stop_words="english",
        use_idf=opts.use_idf,
    )
    X = vectorizer.fit_transform(data)

    print("done in %fs" % (time() - t0))
    print("n_samples: %d, n_features: %d" % X.shape)
    print()

    if n_components:
        print("Performing dimensionality reduction using LSA")
        t0 = time()
        # Vectorizer results are normalized, which makes KMeans behave as
        # spherical k-means for better results. Since LSA/SVD results are
        # not normalized, we have to redo the normalization.
        svd = TruncatedSVD(n_components)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)

        X = lsa.fit_transform(X)

        print("done in %fs" % (time() - t0))

        explained_variance = svd.explained_variance_ratio_.sum()
        print(
            "Explained variance of the SVD step: {}%".format(
                int(explained_variance * 100)
            )
        )
        print()

    km = KMeans(
        n_clusters=5,
        init="k-means++",
        max_iter=100,
        n_init=1,
    )
    print("Clustering sparse data with %s" % km)
    t0 = time()
    km.fit(X)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
    print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
    print("Adjusted Rand-Index: %.3f" % metrics.adjusted_rand_score(labels, km.labels_))
    print(
        "Silhouette Coefficient: %0.3f"
        % metrics.silhouette_score(X, km.labels_, sample_size=1000)
    )

    print()

    # VISU
    print("Top terms per cluster:")
    if n_components:
        original_space_centroids = svd.inverse_transform(km.cluster_centers_)
        order_centroids = original_space_centroids.argsort()[:, ::-1]
    else:
        order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    terms = vectorizer.get_feature_names()
    for i in range(true_k):
        print("Cluster %d:" % i, end="")
        for ind in order_centroids[i, :10]:
            print(" %s" % terms[ind], end="")
        print()
    return


# NOTE CHEN: this is only a prototype
# NOTE CHEN: I like to configure my experiments via yaml files
# The run.py script can be used to parse the configuration files to this file
if __name__ == "__main__":

    #    global args, logger
    #    args = get_parser().parse_args()
    #    logger = helper.get_logger(args.save_folder)
    #    logger.info(args)
    #    logger.info("=> Starting new experiment ...")

    test_path = Path("data/data_jmlr_vol13-21.json")
    corpus = io.load_json(test_path)
    corpus = io.preprocessing(corpus, lib="spacy", stemming=True, lemmatization=True)

    # TODO JESSICA: further filter and clean data by using functions such as filter_extremes (remove all tokens that are less frequent or more frequent than a number),
    # filter_n_most_frequent(filter out the ‘remove_n’ most frequent tokens),
    # merge_with (to merge multiple dictionaries)

    dictionary = corpora.Dictionary(corpus["token"])
    corpus_tfidf, bow_corpus = utils.vectorize.compute_tfidf(
        corpus["token"], dictionary, return_bow=True
    )

    lsi_tfidf = utils.vectorize.lsi(corpus_tfidf, dictionary)
    lsi_bow = utils.vectorize.lsi(bow_corpus, dictionary)
    lda_tfidf = utils.vectorize.lda(corpus_tfidf, dictionary, num_topics=3)
    lda_bow = utils.vectorize.lda(bow_corpus, dictionary, num_topics=3)

    # TODO Use sklearn clustering to cluster these

    # TODO debug jessicas try
    #    vis = pyLDAvis.gensim.prepare(
    #        lda_model, corpus_tfidf, dictionary=lda_model.id2word, mds="mmds"
    #    )

    # Visualize

    print(dictionary.token2id)  # token -> tokenId.
    print(dictionary.dfs)  # token_id -> how many documents contain this token.
    print(bow_corpus)  # list of (token_id, token_count)

    for doc in corpus_tfidf:
        print(doc)

    lsi_tfidf.print_topics()
    lsi_bow.print_topics()
    lda_tfidf.print_topics()
    lda_bow.print_topics()

    visu.plot_2d_space(bow_corpus, lsi_bow)
    visu.plot_2d_space(corpus_tfidf, lsi_tfidf)
    visu.plot_2d_space(corpus_tfidf, lda_model)
    visu.plot_2d_space(corpus_tfidf, lda_model, use_tsne=True)
