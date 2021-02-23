#! /usr/bin/env python3

import os
import ipdb
from pathlib import Path
import itertools
from collections import Counter
from utils.config import Config
from utils.helper import get_logger, get_parser, read_pickle
from utils.preprocessing import Preprocessing

# FIXME delte after testing
import utils.data as io

import numpy as np
import sklearn
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


def main():
    """
    Create a ground truth from clustering the key words. This turned out to be the same
    problem as before, i.e. it is hard to find meaningful words in these sets. Some
    words are more informative than others and in the end we have a lot of noise that
    makes paper appear very similar to each other.
    The naive approach with binary vectors cannot resolve this, so we ran a dbscan on vectorized
    preprocessed words.
    """
    args = get_parser().parse_args()

    config = Config.from_file(args.config)

    logger = get_logger(config.output_path)
    logger.info(args)
    logger.info("=> Starting new experiment ...")

    logger.info("Load data")
    #    test_path = Path("data/data_jmlr_vol13-21.json")
    corpus = io.load_json(config.input_path, append_title=config.use_title)

    # TODO Problem: if we separate key words, the meaning can change drastically
    # e.g. "dynamic programming" -> "dynamic", "programming" has a complete different semantic meaning
    logger.info("Perform preprocessing")
    preprocessed_corpus = Preprocessing(
        corpus["keywords"],
        config=config.preprocessing,
        logger=logger,
    ).apply_preprocessing()
    # NOTE with preprocessing we will have 1754 key words in total
    #    preprocessed_corpus = helper.read_pickle("data/tokenized_kwords_data_jmlr_vol13-21.pkl")

    ipdb.set_trace()

    kword_counts = visualize_key_words(preprocessed_corpus)
    print(kword_counts.most_common(20))

    D = naive(corpus, return_val="distance")
    visualize_distances(np.ravel(D))


if __name__ == "__main__":
    main()
