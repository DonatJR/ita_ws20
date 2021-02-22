from enum import Enum

import yaml


class PreprocessingLib(Enum):
    NLTK = 1
    SPACY = 2

    @staticmethod
    def from_str(label):
        if label.lower() == "nltk":
            return PreprocessingLib.NLTK
        elif label.lower() == "spacy":
            return PreprocessingLib.SPACY
        else:
            raise NotImplementedError("Invalid preprocessing lib")


class DimReduction(Enum):
    LSA = 1
    SPECTRAL = 2
    NONE = 3

    @staticmethod
    def from_str(label):
        if label.lower() == "lsa":
            return DimReduction.LSA
        elif label.lower() == "spectral":
            return DimReduction.SPECTRAL
        else:
            return DimReduction.NONE


class ClusteringMethod(Enum):
    KMEANS = 1
    AGGLOMERATIVE = 2
    # ...

    @staticmethod
    def from_str(label):
        if label.lower() == "kmeans":
            return ClusteringMethod.KMEANS
        elif label.lower() == "agglomerative":
            return ClusteringMethod.AGGLOMERATIVE
        else:
            raise NotImplementedError("Invalid clustering method")


class PreprocessingConfig:
    def __init__(
        self,
        lib,
        use_stemming,
        use_lemmatization,
        min_word_len,
        max_word_len,
        custom_stopwords,
    ):
        self.lib = lib
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        self.min_word_len = min_word_len
        self.max_word_len = max_word_len
        self.custom_stopwords = custom_stopwords


class DimReductionConfig:
    def __init__(self, method, n_components):
        self.method = method
        self.n_components = n_components


class ClusteringConfig:
    def __init__(self, method, n_clusters, agglomerative_linkage):
        self.method = method
        self.n_clusters = n_clusters
        self.agglomerative_linkage = agglomerative_linkage


class Config:
    def __init__(self, config):
        self.input_path = config["input_path"]
        self.output_path = config["output_path"]
        self.use_title = config["use_title"]

        # preprocessing arguments
        lib = PreprocessingLib.from_str(config["preprocessing"]["lib"])
        use_stemming = config["preprocessing"]["stemming"]
        use_lemmatization = config["preprocessing"]["lemmatization"]
        min_word_len = config["preprocessing"]["min_word_len"]
        max_word_len = config["preprocessing"]["max_word_len"]
        custom_stopwords = config["preprocessing"]["custom_stopwords"]

        self.preprocessing = PreprocessingConfig(
            lib,
            use_stemming,
            use_lemmatization,
            min_word_len,
            max_word_len,
            custom_stopwords,
        )

        # dimensionality reduction
        dim_reduction_method = DimReduction.from_str(
            config["embedding"]["dimensionality_reduction"]
        )
        n_components = config["embedding"]["n_components"]
        self.dim_reduction = DimReductionConfig(dim_reduction_method, n_components)

        # clustering
        clustering_method = ClusteringMethod.from_str(config["clustering"]["model"])
        n_clusters = config["clustering"]["n_clusters"]

        agglomerative_linkage = config["clustering"].get(
            "agglomerative_linkage", None
        )  # only used for AgglomerativeClustering -> optional param for other methods

        self.clustering = ClusteringConfig(
            clustering_method, n_clusters, agglomerative_linkage
        )

    @staticmethod
    def from_file(config_path):
        # Read YAML file
        with open(config_path) as stream:
            config = yaml.safe_load(stream)
            return Config(config)
