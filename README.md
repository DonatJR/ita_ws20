# ITA WS20/21

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

TODO: explain them and the supported values!
* input_path
* output_path
* use_title
* preprocessing
  * stemming
  * lemmatization
  * lib
  * min_word_len
  * max_word_len
  * custom_stopwords
* clustering
  * model
  * n_clusters
  * agglomerative_linkage
  *  min_samples
  * eps
  * n_jobs
  * n_components
  * covariance_type
* embedding:
  * dimensionality_reduction
  * n_components


### Experiments
The [notebooks folder](https://github.com/DonatJR/ita_ws20/tree/main/notebooks) contains a variety of different notebooks used to experiment on data or evaluate methods. They are not part of the final project output, but contain the majority of the code used to get to the current project state and are therefore included in the repository. 
