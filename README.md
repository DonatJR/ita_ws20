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

### Idea
TODO: include this here?

### Run the code
The project code can be run by executing the `main.py` script with a `--config` argument pointing to a valid yaml configuration file. If no configuration file is given, the default value `config.yaml` is used and a corresponding file has to exist in the same folder as `main.py`.

We support a variety of different options for the configuration, but encourage the usage of the provided configuration files (TODO: decide which ones to include) as their configuration combinations are (well) tested.

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
* embedding:
  * dimensionality_reduction
  * n_components


### Experiments
The [notebooks folder](https://github.com/DonatJR/ita_ws20/tree/main/notebooks) contains a variety of different notebooks used to experiment on data or evaluate methods. They are not part of the final project output, but contain the majority of the code used to get to the current project state and are therefore included in the repository. 
