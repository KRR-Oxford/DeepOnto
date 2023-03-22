# Ontology Matching with BERTMap and BERTMapLt

!!! credit "Paper"

    Paper for BERTMap: [BERTMap: A BERT-based Ontology Alignment System (AAAI-2022)](https://ojs.aaai.org/index.php/AAAI/article/view/20510).

    ```
    @inproceedings{he2022bertmap,
      title={BERTMap: a BERT-based ontology alignment system},
      author={He, Yuan and Chen, Jiaoyan and Antonyrajah, Denvar and Horrocks, Ian},
      booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
      volume={36},
      number={5},
      pages={5684--5691},
      year={2022}
    }
    ```


This page gives the tutorial for $\textsf{BERTMap}$ family including the summary of the models and how to use them.

<p align="center">
    <img alt="deeponto" src="../images/bertmap.svg">
    <p align="center">Figure 1. Pipeline illustration of BERTMap.</p>
</p>

<br>
The ontology matching (OM) pipeline of $\textsf{BERTMap}$ consists of following steps:

1. Load the source and target ontologies and build **annotation indices** from them based on selected annotation properties.
2. Construct the **text semantics corpora** including **intra-ontology** (from input ontologies), **cross-ontology** (optional, from input mappings), and **auxiliary** (optional, from auxiliary ontologies) sub-corpora. 
3. Split samples in the form of `(src_annotation, tgt_annotation, synonym_label)` into training and validation sets.
4. Fine-tune a **BERT synonym classifier** on the samples and obtain the best checkpoint on the validation split.
5. Predict mappings for each class $c$ of the source ontology $\mathcal{O}$ by:
    
    - Selecting plausible candidates $c'$s in the target ontology $\mathcal{O'}$ based on **idf scores** w.r.t. the **sub-word inverted index** built from the target ontology annotation index.
    For $c$ and a candidate $c'$, first check if they can be string-matched (i.e., share a common annotation, or equivalently the maximum edit similarity score is $1.0$); if not,
    consider all combinations (cartesian product) of their respective class annotations, compute a synonym score for each combination, and take the **average of synonym scores as the mapping score**.
    - $N$ best scored mappings (no filtering) will be preserved as raw predictions which should have relatively higher recall and lower precision.

6. Extend the raw predictions using an **iterative** algorithm based on the **locality principle**. To be specific, if $c$ and $c'$ are matched with a **relatively high mapping score** ($\geq \kappa$), then search for plausible mappings between the *parents* (resp. *children*) of $c$ and the *parents* (resp. *children*) of $c'$. This process is iterative because there would be new
highly scored mappings at each round. Terminate mapping extension when there is no
new mapping with score $\geq \kappa$ found or it exceeds the maximum number of iterations. Note that $\kappa$ is set to $0.9$ by default, as in the original paper.

7. Truncate the extended mappings by preserving only those with scores $\geq \lambda$. In the original paper, $\lambda$ is supposed to be tuned on validation mappings – which are often not available. Also, $\lambda$ is not a sensitive hyperparameter in practice. Therefore, we manually set $\lambda$ to $0.9995$ as a default value which usually yields a higher F1 score. Note that both $\kappa$ and $\lambda$ are made available in the configuration file.

8. Repair the rest of the mappings with the repair module built in LogMap (BERTMap does not focus on mapping repair). In short, a minimum set of inconsistent mappings will be removed (further improve precision).

Steps 5-8 are referred to as the **global matching** process which computes OM mappings from two input ontologies. $\textsf{BERTMapLt}$ is the light-weight version without BERT training and mapping refinement. The mapping filtering threshold for $\textsf{BERTMapLt}$ is $1.0$ (i.e., string-matched). 

In addition to the traditional OM procedure, the scoring modules of $\textsf{BERTMap}$ and $\textsf{BERTMapLt}$ can be used to evaluate any class pair given their selected annotations. This is useful in ranking-based evaluation. 

!!! warning

    The $\textsf{BERTMap}$ family rely on sufficient class annotations for constructing training corpora
    of the BERT synonym classifier, especially under the **unsupervised** setting where there are no input mappings and/or external resources. It is very important to specify correct [**annotation properties**](#annotation-properties) in the configuration file.


## Usage

To use $\textsf{BERTMap}$, a configuration file and two input ontologies to be matched should be imported.

```python
from deeponto.onto import Ontology
from deeponto.align.bertmap import BERTMapPipeline

config_file = "path_to_config.yaml"
src_onto_file = "path_to_the_source_ontology.owl"  
tgt_onto_file = "path_to_the_target_ontology.owl" 

config = BERTMapPipeline.load_bertmap_config(config_file)
src_onto = Ontology(src_onto_file)
tgt_onto = Ontology(tgt_onto_file)

BERTMapPipeline(src_onto, tgt_onto, config)
```

The default configuration file can be loaded as:

```python
from deeponto.align.bertmap import BERTMapPipeline, DEFAULT_CONFIG_FILE

config = BERTMapPipeline.load_bertmap_config(DEFAULT_CONFIG_FILE)
```

The loaded configuration is a `CfgNode` object supporting attribute access of dictionary keys. 
To customise the configuration, users can either copy the `DEFAULT_CONFIG_FILE`, save it locally using `BERTMapPipeline.save_bertmap_config` method, and modify it accordingly; or change it in the run time.

```python
from deeponto.align.bertmap import BERTMapPipeline, DEFAULT_CONFIG_FILE

config = BERTMapPipeline.load_bertmap_config(DEFAULT_CONFIG_FILE)

# save the configuration file
BERTMapPipeline.save_bertmap_config(config, "path_to_saved_config.yaml")

# modify it in the run time
# for example, add more annotation properties for synonyms
config.annotation_property_iris.append("http://...") 
```

If using $\textsf{BERTMap}$ for scoring class pairs instead of global matching, disable automatic global matching and load class pairs to be scored.

```python
from deeponto.onto import Ontology
from deeponto.align.bertmap import BERTMapPipeline

config_file = "path_to_config.yaml"
src_onto_file = "path_to_the_source_ontology.owl"  
tgt_onto_file = "path_to_the_target_ontology.owl" 

config = BERTMapPipeline.load_bertmap_config(config_file)
config.global_match.enabled = False
src_onto = Ontology(src_onto_file)
tgt_onto = Ontology(tgt_onto_file)

bertmap = BERTMapPipeline(src_onto, tgt_onto, config)

class_pairs_to_be_scored = [...]  # (src_class_iri, tgt_class_iri)
for src_class_iri, tgt_class_iri in class_pairs_to_be_scored:
    # retrieve class annotations
    src_class_annotations = bertmap.src_annotation_index[src_class_iri]
    tgt_class_annotations = bertmap.tgt_annotation_index[tgt_class_iri]
    # the bertmap score
    bertmap_score = bertmap.mapping_predictor.bert_mapping_score(
        src_class_annotations, tgt_class_annotations
    )
    # the bertmaplt score
    bertmaplt_score = bertmap.mapping_predictor.edit_similarity_mapping_score(
        src_class_annotations, tgt_class_annotations
    )
    ...
```

!!! tip

    The implemented $\textsf{BERTMap}$ by default searches **for each source ontology class** a set of possible matched target ontology classes.
    Because of this, it is recommended to set the source ontology as the one with a **smaller number of classes** for efficiency.

    Note that in the original paper, the model is expected to match for both directions `src2tgt` and `tgt2src`, and also consider the combination
    of both results. However, this does not usually bring better performance and significantly consumes more time. Therefore, this feature is discarded
    and the users can choose which direction to match.


## Configuration

The default configuration file looks like:

```yaml
model: bertmap  # bertmap or bertmaplt

output_path: null  # if not provided, the current path "." is used

annotation_property_iris:
  - http://www.w3.org/2000/01/rdf-schema#label  # rdfs:label
  - http://www.geneontology.org/formats/oboInOwl#hasSynonym
  - http://www.geneontology.org/formats/oboInOwl#hasExactSynonym
  - http://www.w3.org/2004/02/skos/core#exactMatch
  - http://www.ebi.ac.uk/efo/alternative_term
  - http://www.orpha.net/ORDO/Orphanet_#symbol
  - http://purl.org/sig/ont/fma/synonym
  - http://www.w3.org/2004/02/skos/core#prefLabel
  - http://www.w3.org/2004/02/skos/core#altLabel
  - http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#P108
  - http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#P90
  
# additional corpora 
known_mappings: null  # cross-ontology corpus
auxiliary_ontos: [] # auxiliary corpus

# bert config
bert:  
  pretrained_path: emilyalsentzer/Bio_ClinicalBERT  
  max_length_for_input: 128 
  num_epochs_for_training: 3.0
  batch_size_for_training: 32
  batch_size_for_prediction: 128
  resume_training: null

# global matching config
global_matching:
  enabled: true
  num_raw_candidates: 200 
  num_best_predictions: 10 
  mapping_extension_threshold: 0.9   
  mapping_filtered_threshold: 0.9995 
```

### BERTMap or BERTMapLt

`config.model`
:   By changing this parameter to `bertmap` or `bertmaplt`, users can switch between 
$\textsf{BERTMap}$ and $\textsf{BERTMapLt}$. Note that $\textsf{BERTMapLt}$ does not use any
training and mapping refinement parameters.

### Annotation Properties :warning:

`config.annotation_property_iris`
:   The IRIs stored in this parameter refer to annotation properties with literal values
that define the **synonyms** of an ontology class. Many ontology matching systems rely on synonyms for good performance, including the $\textsf{BERTMap}$ family. The default `config.annotation_property_iris` are in line with the **Bio-ML** dataset, which will be constantly updated. Users can append or delete IRIs for specific input ontologies.

Note that it is safe to specify all possible annotation properties regardless of input ontologies because the ones that are not used will be ignored.

### Additional Training Data

The **text semantics corpora** by default (unsupervised setting) will consist of two **intra-ontology sub-corpora** built from two input ontologies (based on the specified annotation properties). To add more training data, users can opt to feed input mappings (**cross-ontology sub-corpus**) and/or a list of auxiliary ontologies (**auxiliary sub-corpora**). 

`config.known_mappings`
:   Specify the path to input mapping file here; the input mapping file should be a `.tsv` or `.csv` file with three columns with headings: `["SrcEntity", "TgtEntity", "Score"]`. Each row corresponds to a triple $(c, c', s(c, c'))$ where $c$ is a source ontology class, $c'$ is a target ontology class, and $s(c, c')$ is the matching score. Note that in the BERTMap context, input mapppings are assumed to be gold standard (reference) mappings with scores equal to $1.0$. Regardless of scores specified in the mapping file, the scores of the input mapppings will be adjusted to $1.0$ automatically.

`config.auxiliary_ontos`
:   Specify a list of paths to auxiliary ontology files here. For each auxiliary ontology, a corresponding intra-ontology corpus will be created and thus produce more synonym and non-synonym samples.

### BERT Settings

`config.bert.pretrained_path`
:   $\textsf{BERTMap}$ uses the pre-trained **Bio-Clincal BERT** as specified in this parameter because it was originally applied on biomedical ontologies. For general purpose ontology matching, users can use pre-trained variants such as `bert-base-uncased`.

`config.bert.batch_size_for_training`
: Batch size for BERT fine-tuning.

`config.bert.batch_size_for_prediction`
: Batch size for BERT validation and mapping prediction.

Adjust these two parameters if users found an inappropriate GPU memory fit. 

`config.bert.resume_training`
:   Set to `true` if the BERT training process is somehow interrupted and users wish to continue training.

### Global Matching Settings

`config.global_matching.enabled`
:   As mentioned in [usage](#usage), users can disable automatic global matching by setting this parameter to `false` if they wish to use the mapping scoring module only. 

`config.global_matching.num_raw_candidates`
: Set the number of raw candidates selected in the mapping prediction phase. 

`config.global_matching.num_best_predictions`
:   Set the number of best scored mappings preserved in the mapping prediction phase. The default value `10` is often more than enough.

`config.global_matching.mapping_extension_threshold`
:   Set the score threshold of mappings used in the iterative mapping extension process. Higher value shortens the time but reduces the recall. 

`config.global_matching.mapping_filtered_threshold`
:   The score threshold of mappings preserved for final mapping refinement. 


## Output Format

Running $\textsf{BERTMap}$ will create a directory named `bertmap` or `bertmaplt` in the specified output path.
The file structure of this directory is as follows:

```
bertmap
├── data
│   ├── fine-tune.data.json
│   └── text-semantics.corpora.json
├── bert
│   ├── tensorboard
│   ├── checkpoint-{some_number}
│   └── checkpoint-{some_number}
├── match
│   ├── logmap-repair
│   ├── raw_mappings.json
│   ├── repaired_mappings.tsv 
│   ├── raw_mappings.tsv
│   ├── extended_mappings.tsv
│   └── filtered_mappings.tsv
├── bertmap.log
└── config.yaml
```

It is worth mentioning that the `match` sub-directory contains all the global matching files:

`raw_mappings.tsv`
: The raw mapping predictions before mapping refinement. The `.json` one is used internally to prevent accidental interruption. Note that `bertmaplt` only produces raw mapping predictions (no mapping refinement).

`extended_mappings.tsv`
:   The output mappings after applying mapping extension. 

`filtered_mappings.tsv`
: The output mappings after mapping extension and threshold filtering. 

`logmap-repair`
: A folder containing intermediate files needed for applying LogMap's debugger.

`repaired_mappings.tsv`
: The **final** output mappings after mapping repair.
