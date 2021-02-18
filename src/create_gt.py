#! /usr/bin/env python3

import os
import argparse
import ipdb
from pathlib import Path
from time import time
import itertools
from collections import Counter

import numpy as np
import pandas as pd
from gensim import corpora
import sklearn
import utils.helper as helper
import utils.data as io
import scipy

import utils.visu as visu
import matplotlib.pyplot as plt
import seaborn as sns

"""
We want to create ground truth labels for evaluation and supervision.
This is based on the key words for the papers. 
"""


def flatten(list_of_lists):
    flattened = list(itertools.chain.from_iterable(list_of_lists))
    return flattened


def convert_list_to_str(x):
    return " ".join(x)


def naive(data, return_val="distance"):
    """
    Go over all key words and create global array.
    Query each key word for each paper. Mark every key words that is used for the
    paper as 1 and all else as 0. This will create sparse vectors for each
    paper. We can compute similarities based on a Hamming distance on these alone.
    """
    from sklearn.cluster import DBSCAN

    def get_global_key_word_vect(kword_list):
        """ Go iteratively over list levels and collapse """
        # This is for the case that we have a more recursive list of lists than 2 levels
        while isinstance(kword_list[0], list):
            kword_list = flatten(kword_list)
        # NOTE we somehow have empty strings here either from preprocessing or above operations
        set_kwords = list(set(kword_list))
        set_kwords = [x.split() for x in set_kwords]
        set_kwords = list(filter(None, set_kwords))
        set_kwords = flatten(set_kwords)
        return set_kwords

    def vectorize(data, all_keywords):
        vect = np.zeros(len(all_keywords))
        for kword in data:
            if kword in all_keywords:
                id_to_flip = all_keywords.index(kword)
                vect[id_to_flip] = 1
        return vect

    def compute_vectors(data, set_kwords):
        all_vect = []
        for kwords in data:
            bin_vect = vectorize(kwords, set_kwords)
            all_vect.append(bin_vect)
        return np.array(all_vect)

    data = data.apply(flatten)
    set_kwords = get_global_key_word_vect(data)
    features = compute_vectors(data, set_kwords)
    if return_val == "distance":
        D = sklearn.metrics.pairwise_distances(
            features, metric=scipy.spatial.distance.hamming
        )
        return D
    else:
        cluster = DBSCAN(
            # NOTE eps is very sensitive. if set too high we get easily no clusters
            eps=0.002,
            min_samples=2,
            metric=scipy.spatial.distance.hamming,
            n_jobs=4,
        ).fit(features)
        return cluster.labels_


def visualize_distances(D):
    sns.displot(D, bins=70, kde=True)
    #    plt.title("Distribution of pairwise binary key word vector hamming distances")
    #    plt.xlabel("Key word Hamming distance")
    #    plt.ylabel("#")
    plt.show()


def visualize_clusters():
    pass


def visualize_key_words(data, return_items=True):
    """ Plot histogram of key words to count """
    data = data.apply(flatten)
    data = data.apply(convert_list_to_str)
    counts = Counter(" ".join(data.values.tolist()).split(" "))
    del counts[""]
    counts = counts.items()

    fig = plt.figure()
    sns.displot(counts.values(), kde=True)
    plt.title("Distribution of counts for key words over dataset")
    plt.show()

    if return_items:
        return counts


# TODO how many different key words do we have before and after preprocessing?
def main():
    #    test_path = Path("data/data_jmlr_vol13-21.json")
    #    corpus = io.load_json(test_path, return_data="keywords")

    # NOTE this preprocessing will tokenize the key words. This way we can capture
    # similar key words to be the same thing
    # TODO Problem: if we separate key words, the meaning can change drastically
    # e.g. "dynamic programming" -> "dynamic", "programming" has a complete different semantic meaning

    #    corpus = io.preprocessing(
    #        corpus,
    #        lib="spacy",
    #        datatype="keywords",
    #        stemming=False,
    #        lemmatization=True,
    #    )

    # NOTE with preprocessing we will have 1754 key words in total
    corpus = helper.read_pickle("data/tokenized_kwords_data_jmlr_vol13-21.pkl")
    kword_counts = visualize_key_words(corpus)
    ipdb.set_trace()
    print(kword_counts.most_common(20))

    # NOTE for the first ~500 kwords do we have several counts
    # this could actually result in few clusters based on distance
    # TODO problem: if we naively cluster all keyy words contribute equally to distance
    # but some words are more descriptive than others, so a weighted contribution would be intuitive
    # this is however crazy subjective and the core problem
    D = naive(corpus, return_val="distance")
    visualize_distances(np.ravel(D))


if __name__ == "__main__":
    main()
