import numpy as np
from sklearn.cluster import Birch, KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


class Clustering:
    def __init__(self, model, corpus, n_clusters):
        self.model = model
        self.n_clusters = n_clusters
        self.corpus = corpus

    def peform_clustering(self):
        # apply tfidf
        vectorizer, tfidf_corpus = self.get_tfidf_corpus(self.corpus)

        # perform dimensionality reduction
        # TODO: dim reduction

        self.model = self.model(self.n_clusters)
        self.model.fit(tfidf_corpus)
        return self.model

    def identity_tokenizer(self, text):
        return text

    def get_tfidf_corpus(self, corpus):
        vectorizer = TfidfVectorizer(tokenizer=self.identity_tokenizer, lowercase=False)
        return vectorizer, vectorizer.fit_transform(corpus["token"])

    def get_top_words(self):
        clusters = []
        for i in range(np.max(self.model.labels_) + 1):
            vectorizer, tfidf_corpus = self.get_tfidf_corpus(
                self.corpus.iloc[np.where(self.model.labels_ == i)]
            )
            scores = zip(
                vectorizer.get_feature_names(),
                np.asarray(tfidf_corpus.sum(axis=0)).ravel(),
            )
            sorted_scores = list(zip(*sorted(scores, key=lambda x: x[1], reverse=True)))
            top_words = sorted_scores[0][:10]
            clusters.append(top_words)
        return clusters
