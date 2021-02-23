from enum import IntEnum

import yaml


class PreprocessingLib(IntEnum):
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


class DimReduction(IntEnum):
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


class ClusteringMethod(IntEnum):
    KMEANS = 1
    AGGLOMERATIVE = 2
    AFFINITY_PROPAGATION = 3
    DBSCAN = 4
    MEAN_SHIFT = 5
    OPTICS = 6
    BIRCH = 7
    GAUSSIAN_MIXTURE = 8
    SPECTRAL = 9

    @staticmethod
    def from_str(label):
        if label.lower() == "kmeans":
            return ClusteringMethod.KMEANS
        elif label.lower() == "agglomerative":
            return ClusteringMethod.AGGLOMERATIVE
        elif label.lower() == "affinitypropagation":
            return ClusteringMethod.AFFINITY_PROPAGATION
        elif label.lower() == "dbscan":
            return ClusteringMethod.DBSCAN
        elif label.lower() == "meanshift":
            return ClusteringMethod.MEAN_SHIFT
        elif label.lower() == "optics":
            return ClusteringMethod.OPTICS
        elif label.lower() == "birch":
            return ClusteringMethod.BIRCH
        elif label.lower() == "gaussianmixture":
            return ClusteringMethod.GAUSSIAN_MIXTURE
        elif label.lower() == "spectral":
            return ClusteringMethod.SPECTRAL
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
    def __init__(
        self,
        method,
        n_clusters=None,
        agglomerative_linkage=None,
        min_samples=None,
        eps=None,
        n_jobs=None,
        n_components=None,
        covariance_type=None,
    ):
        self.method = method
        self.n_clusters = n_clusters
        self.agglomerative_linkage = agglomerative_linkage
        self.min_samples = min_samples
        self.eps = eps
        self.n_jobs = n_jobs
        self.n_components = n_components
        self.covariance_type = covariance_type


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
        n_clusters = config["clustering"].get(
            "n_clusters", None
        )  # only used for some methods -> optional param for other methods

        agglomerative_linkage = config["clustering"].get(
            "agglomerative_linkage", None
        )  # only used for some methods -> optional param for other methods

        min_samples = config["clustering"].get(
            "min_samples", None
        )  # only used for some methods -> optional param for other methods

        eps = config["clustering"].get(
            "eps", None
        )  # only used for some methods -> optional param for other methods

        n_jobs = config["clustering"].get(
            "n_jobs", -1
        )  # only used for some methods -> optional param for other methods

        n_components = config["clustering"].get(
            "n_components", None
        )  # only used for some methods -> optional param for other methods

        covariance_type = config["clustering"].get(
            "covariance_type", None
        )  # only used for some methods -> optional param for other methods

        self.clustering = ClusteringConfig(
            clustering_method,
            n_clusters,
            agglomerative_linkage,
            min_samples,
            eps,
            n_jobs,
            n_components,
            covariance_type,
        )

        self.__check_optional_params()

    def __check_optional_params(self):
        if (
            self.clustering.method == ClusteringMethod.KMEANS
            or self.clustering.method == ClusteringMethod.BIRCH
            or self.clustering.method == ClusteringMethod.SPECTRAL
        ):
            assert self.clustering.n_clusters is not None
        elif self.clustering.method == ClusteringMethod.AGGLOMERATIVE:
            assert (
                self.clustering.n_clusters is not None
                and self.clustering.agglomerative_linkage is not None
            )
        elif self.clustering.method == ClusteringMethod.DBSCAN:
            assert (
                self.clustering.min_samples is not None
                and self.clustering.eps is not None
                and self.clustering.n_jobs is not None
            )
        elif (
            self.clustering.method == ClusteringMethod.MEAN_SHIFT
            or self.clustering.method == ClusteringMethod.OPTICS
        ):
            assert self.clustering.n_jobs is not None
        elif self.clustering.method == ClusteringMethod.GAUSSIAN_MIXTURE:
            assert (
                self.clustering.n_components is not None
                and self.clustering.covariance_type is not None
            )

    @staticmethod
    def from_file(config_path):
        # Read YAML file
        with open(config_path) as stream:
            config = yaml.safe_load(stream)
            return Config(config)
