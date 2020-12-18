#! /usr/bin/env python3

"""
This contains all relevant visualization routines.
"""

import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")
from mpl_toolkits.mplot3d import axes3d
from gensim.utils import simple_preprocess
from gensim import corpora, models
from sklearn.manifold import TSNE
import pyLDAvis.gensim
from wordcloud import WordCloud

import pandas as pd
import numpy as np
import ipdb


# TODO TEST 3d
# TODO pass top key words for each topic
def plot_tsne(corpus, model, dim=2, save_path=None):
    """
    This projects higher dimensional features on a kD space using the TSNE algorithm.
    t-SNE differs from PCA by preserving only small pairwise distances or
    local similarities whereas PCA is concerned with
    preserving large pairwise distances to maximize variance, see:
    https://towardsdatascience.com/an-introduction-to-t-sne-with-python-example-5a3a293108d1
    https://www.quora.com/What-advantages-does-the-t-SNE-algorithm-have-over-PCA

    arguments:
    ---
    corpus []:
    model []:
    """

    # TODO this cant be vectorized?
    # Get topic weights
    topic_weights = []
    for i, row_list in enumerate(model[corpus]):
        if isinstance(model, models.ldamodel.LdaModel):
            topic_weights.append([w for i, w in row_list[0]])
        else:
            topic_weights.append([w for i, w in row_list])

    # Array of topic weights
    arr = pd.DataFrame(topic_weights).fillna(0).values
    # Dominant topic number in each doc
    topic_num = np.argmax(arr, axis=1)

    tsne_model = TSNE(
        n_components=dim, verbose=1, random_state=0, angle=0.99, init="pca"
    )
    tsne = tsne_model.fit_transform(arr)

    if dim == 2:
        fig, ax = plt.subplots(figsize=(10, 10))
    elif dim == 3:
        fig, ax = plt.subplots(111, figsize=(10, 10), projection="3d")
    ax.scatter(*zip(*tsne), c=topic_num, s=80, alpha=0.8)
    # TODO include labels for cluster that show their topics
    # How to get top key words for num topics
    ipdb.set_trace()
    ax.legend()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_lda(corpus, model, dim=2):

    # TODO why is this only 2d?
    # TODO could we check for 3d?
    projection = []
    for i in range(dim):
        if isinstance(model, models.ldamodel.LdaModel):
            # TODO why list(model[corpus]) for i > 0?
            projection.append([x[0][i][1] for x in model[corpus]])
        else:
            projection.append([x[i][1] for x in model[corpus]])

    #    if dim == 2:
    #        if isinstance(model, models.ldamodel.LdaModel):
    #            documents_2d_1 = [x[0][0][1] for x in model[corpus]]
    #            documents_2d_2 = [x[0][1][1] for x in list(model[corpus])]
    #        else:
    #            documents_2d_1 = [x[0][1] for x in model[corpus]]
    #            documents_2d_2 = [x[1][1] for x in list(model[corpus])]

    # Get topic weights
    topic_weights = []
    for i, row_list in enumerate(model[corpus]):
        if isinstance(model, models.ldamodel.LdaModel):
            topic_weights.append([w for i, w in row_list[0]])
        else:
            topic_weights.append([w for i, w in row_list])

    # Array of topic weights
    arr = pd.DataFrame(topic_weights).fillna(0).values
    # Dominant topic number in each doc
    topic_num = np.argmax(arr, axis=1)

    if dim == 2:
        fig, ax = plt.subplots(figsize=(10, 10))
    elif dim == 3:
        fig, ax = plt.subplots(111, figsize=(10, 10), projection="3d")
    else:
        raise Exception("Higher dimensions not supported")

    ax.scatter(*zip(*tsne), c=topic_num, s=80, alpha=0.8)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# TODO this needs a projection similar to tsne
def plot_segmentation(data, label):
    """
    Given assignments and data points, color the different cluster and plot them in a
    2D space.
    """
    raise NotImplementedError()


def plot_wordcloud(text, save_path=None):
    """ Plot word cloud on joined text using frequeny count """

    if isinstance(text, pd.Series):
        text = text.str.join(" ")
        text = " ".join(text.tolist())
    elif isinstance(text, list):
        text = " ".join(text)
    else:
        raise Exception("Please pass list or pd.Series")

    # wordcloud = WordCloud(max_font_size=60, max_words=100, background_color="white").generate_from_frequencies(df_counter_words)
    wordcloud = WordCloud(
        max_font_size=60, max_words=100, background_color="white", collocations=False
    ).generate(text)

    fig = plt.figure(1, figsize=(12, 12))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
