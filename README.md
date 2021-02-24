# Scientific paper clustering - ITA WS20/21

## Team members

* Daniela Fichiu
  * 3552717
  * BSc Applied Computer Science & BSc Mathematics
  * daniela.fichiu@stud.uni-heidelberg.de

* Christian Homeyer
  * 3606476
  * PhD Computer Science
  * ox182@uni-heidelberg.de

* Jessica Kaechele
  * 3588787
  * MSc Applied Computer Science
  * uo251@stud.uni-heidelberg.de

* Jonas Reinwald
  * 3600238
  * MSc Applied Computer Science
  * am248@stud.uni-Heidelberg.de

## Documents

### Proposal docs
https://github.com/DonatJR/ita_ws20/tree/main/project_proposal

### Milestone document
https://github.com/DonatJR/ita_ws20/blob/main/milestone/milestone.md

## Project

### Main project goals
Our main goal is to make it easier for users to search and explore scientific papers belonging to a specific topic or theme.
For this we want to specifically arrive at a clustering (representation) that, for one, separates the different documents into correct clusters, but is also easy to work with in downstream tasks (e.g. the mentioned inclusion in some search site).
To achieve this we basically interpret the steps mentioned in the subsection \ref{subsec:pipeline} as some coarse sub goals, which can then be worked on by different team members.
Some of these sub goals can also be further divided, for example downloading and preprocessing data from different sources or implementing distinct clustering algorithms can be done by a single team member respectively.

### Running the code
The project code can be run by executing the `main.py` script with a `--config` argument pointing to a valid yaml configuration file. If no configuration file is given, the default value `config.yaml` is used and a corresponding file has to exist in the same folder as `main.py`.

We support a variety of different options for the configuration, but encourage the usage of the provided configuration files as their configuration combinations are (well) tested.

### Configuration
The configuration supports these options:

* __input_path__: path of the (json) data file
* __output_path__: path where all results will be saved
* __use_title__: whether to include titles of papers when loading data; possible values: True | False
* __preprocessing__:
  * __stemming__: whether to use stemming in preprocessing of data; possible values: True | False
  * __lemmatization__: whether to use lemmatization in preprocessing of data; possible values: True | False
  * __lib__: library used for preprocessing; possible values: 'spacy' | 'nltk'
  * __min_word_len__: minimum length of tokens to include in preprocessing result; possible values: Integer > 0
  * __max_word_len__: maximum length of tokens to include in preprocessing result; possible values: Integer > 0 (> min_word_len)
  * __custom_stopwords__: stopword to use in addition to standard stopwords; possible values: list of str
* __clustering__:
  * __model__: type of clustering method to use; possible values: 'KMeans' | 'Agglomerative' | 'AffinityPropagation' | 'DBSCAN' | 'MeanShift' | 'OPTICS' | 'Birch' | 'GaussianMixture' | 'Spectral'
  * __n_clusters__: model parameter; possible values: Integer > 0; used for 'KMeans' | 'Agglomerative' | 'Birch' | 'Spectral' models
  * __agglomerative_linkage__: model parameter; possible values: 'ward' | 'complete' | 'average' | 'single'; used for 'Agglomerative' model
  * __min_samples__: model parameter; possible values: Integer > 0; used for 'DBSCAN' model
  * __eps__: model parameter; possible values: Float > 0; used for 'DBSCAN' model
  * __n_jobs__: model parameter; possible values: Integer > 0 (number of processors to use) or -1 (use all processors); used for 'DBSCAN'| 'MeanShift' | 'OPTICS' models
  * __n_components__: model parameter; possible values: Integer > 0; used for 'GaussianMixture' model
  * __covariance_type__: model parameter; possible values: 'full' | 'tied' | 'diag' | 'spherical'; used for 'GaussianMixture' model
* __embedding__:
  * __dimensionality_reduction__: dimensionality reduction method to use on data before attempting to cluster; possible values: None | 'LSA' | 'SPECTRAL'
  * __n_components__: parameter for dimensionality reduction (dimension of the projected subspace); possible values: Integer > 1


### Experiments
The [notebooks folder](https://github.com/DonatJR/ita_ws20/tree/main/notebooks) contains a variety of different notebooks used to experiment on data or evaluate methods. They are not part of the final project output, but contain the majority of the code used to get to the current project state and are therefore included in the repository. 
