import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import SpectralEmbedding
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from utils.config import ClusteringMethod, DimReduction


class Clustering:
    def __init__(self, corpus, clustering_config, dim_reduction_config, logger=None):
        self.__corpus = corpus
        self.__clustering_config = clustering_config
        self.__dim_reduction_config = dim_reduction_config
        self.__logger = logger

    def perform_clustering(self):
        vectorizer, tfidf_corpus = self.__get_tfidf_corpus(self.__corpus)

        corpus = self.__perform_dim_reduction(
            tfidf_corpus,
            self.__dim_reduction_config.method,
            self.__dim_reduction_config.n_components,
        )

        self.model = self.__get_model(self.__clustering_config)
        self.model.fit(corpus)
        return self.model

    def get_top_words(self):
        clusters = []
        for i in range(np.max(self.model.labels_) + 1):
            vectorizer, tfidf_corpus = self.__get_tfidf_corpus(
                self.__corpus.iloc[np.where(self.model.labels_ == i)], True
            )
            scores = zip(
                vectorizer.get_feature_names(),
                np.asarray(tfidf_corpus.sum(axis=0)).ravel(),
            )
            sorted_scores = list(zip(*sorted(scores, key=lambda x: x[1], reverse=True)))
            top_words = sorted_scores[0][:10]
            clusters.append(top_words)
        return clusters

    def __identity_tokenizer(self, text):
        return text

    def __get_tfidf_corpus(self, corpus, silent=False):
        if not silent:
            self.__log("applying TD-IDF")
        vectorizer = TfidfVectorizer(
            tokenizer=self.__identity_tokenizer, lowercase=False
        )
        return vectorizer, vectorizer.fit_transform(corpus["token"])

    def __get_model(self, config):
        if config.method == ClusteringMethod.KMEANS:
            self.__log("using KMeans-Model")
            return KMeans(n_clusters=config.n_clusters)
        elif config.method == ClusteringMethod.AGGLOMERATIVE:
            self.__log("using AgglomerativeClustering-Model")
            return AgglomerativeClustering(
                n_clusters=config.n_clusters, linkage=config.agglomerative_linkage
            )
        else:
            raise NotImplementedError("Unknown clustering method")

    def __perform_dim_reduction(self, corpus, dim_reduction_method, n_components):
        if dim_reduction_method == DimReduction.LSA:
            self.__log("performing lsa embedding")
            return self.__get_lsa_transformation(n_components, corpus)
        elif dim_reduction_method == DimReduction.SPECTRAL:
            self.__log("performing spectral embedding")
            return self.__get_spectral_transformation(n_components, corpus)
        else:
            self.__log("performing no dimensionality reduction")
            return corpus.toarray()

    def __get_lsa_transformation(self, n_components, corpus):
        svd = TruncatedSVD(n_components)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)
        return lsa.fit_transform(corpus)

    def __get_spectral_transformation(self, n_components, corpus):
        return SpectralEmbedding(n_components).fit_transform(corpus)

    def __log(self, message):
        if self.__logger is not None:
            self.__logger.info(message)
