#! /usr/bin/env python3

"""
This contains all relevant visualization routines.
"""

# TODO remove unnecessary dependencies
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")
from gensim.utils import simple_preprocess
from gensim import corpora, models
from sklearn.manifold import TSNE
import pyLDAvis.gensim

import pandas as pd
import numpy as np


# TODO
def project_to_lower(vectors, k_components=2):
    """
    Given a high dimensional vector, project to lower dimensional space
    using PCA. This is useful for visualizing high dimensional features and
    their distances.
    """
    raise NotImplementedError()


# TODO make this self contained
# TODO make this more flexible to take just numpy feature vectors
# TODO refactor projection
# TODO make save option configurable
def plot_2d_space(corpus, method, use_tsne=False):
    """
    This projects higher dimensional features on a kD space using the
    k largest components in a PCA.
    """

    if isinstance(method, models.ldamodel.LdaModel):
        documents_2d_1 = [x[0][0][1] for x in method[corpus]]
        documents_2d_2 = [x[0][1][1] for x in list(method[corpus])]
    else:
        documents_2d_1 = [x[0][1] for x in method[corpus]]
        documents_2d_2 = [x[1][1] for x in list(method[corpus])]

    fig, ax = plt.subplots(figsize=(10, 10))

    # Get topic weights
    topic_weights = []
    for i, row_list in enumerate(method[corpus]):
        if isinstance(method, models.ldamodel.LdaModel):
            topic_weights.append([w for i, w in row_list[0]])
        else:
            topic_weights.append([w for i, w in row_list])

    # Array of topic weights
    arr = pd.DataFrame(topic_weights).fillna(0).values

    # Dominant topic number in each doc
    topic_num = np.argmax(arr, axis=1)

    if use_tsne:
        tsne_model = TSNE(
            n_components=2, verbose=1, random_state=0, angle=0.99, init="pca"
        )
        tsne = tsne_model.fit_transform(arr)
        documents_2d_1 = tsne[:, 0]
        documents_2d_2 = tsne[:, 1]

    ax.scatter(documents_2d_1, documents_2d_2, c=topic_num, s=80, alpha=0.8)
    #    print(corpus["title"])
    for i, data in enumerate(corpus):
        ax.annotate(i, (documents_2d_1[i] + 0.01, documents_2d_2[i]))

    plt.show()


def plot_segmentation():
    """
    Given assignments and data points, color the different cluster and plot them in a
    2D space.
    """
    raise NotImplementedError()
