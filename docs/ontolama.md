# OntoLAMA: Dataset Overview and Usage Guide

!!! credit "paper"

    Paper for OntoLAMA:
    [Language Model Analysis for Ontology Subsumption Inference (Findings of ACL 2023)](https://aclanthology.org/2023.findings-acl.213).

    ```
    @inproceedings{he-etal-2023-language,
        title = "Language Model Analysis for Ontology Subsumption Inference",
        author = "He, Yuan  and
        Chen, Jiaoyan  and
        Jimenez-Ruiz, Ernesto  and
        Dong, Hang  and
        Horrocks, Ian",
        booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
        month = jul,
        year = "2023",
        address = "Toronto, Canada",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2023.findings-acl.213",
        pages = "3439--3453"
    }
    ```

This page provides an overview of the $\textsf{OntoLAMA}$ datasets, how to use them, and the related probing approach introduced in the research paper.

## Overview

$\textsf{OntoLAMA}$ is a set of language model (LM) probing datasets for ontology subsumption inference. The work follows the "LMs-as-KBs" literature but focuses on conceptualised knowledge extracted from formalised KBs such as the OWL ontologies. Specifically, the subsumption inference (SI) task is introduced and formulated in the Natural Language Inference (NLI) style, where the sub-concept and the super-concept involved in a subsumption axiom are verbalised and fitted into a template to form the premise and hypothesis, respectively. The sampled axioms are verified through ontology reasoning. The SI task is further divided into Atomic SI and Complex SI where the former involves only atomic named concepts and the latter involves both atomic and complex concepts. Real-world ontologies of different scales and domains are used for constructing OntoLAMA and in total there are **four** Atomic SI datasets and **two** Complex SI datasets.

## Useful Links

- Datasets available at Zenodo: *<https://doi.org/10.5281/zenodo.6480540>* (CC BY 4.0 International).
- Also available at Huggingface: *<https://huggingface.co/datasets/krr-oxford/OntoLAMA>*.
- The source code for dataset construction and LM probing is available at: *<https://krr-oxford.github.io/DeepOnto/deeponto/probe/ontolama/>*.


## Statistics

<small>

| Source     | #NamedConcepts | #EquivAxioms | #Dataset (Train/Dev/Test)                                              |
|------------|----------------|--------------|------------------------------------------------------------------------|
| Schema.org | 894            | -            | Atomic SI: 808/404/2,830                                               |
| DOID       | 11,157         | -            | Atomic SI: 90,500/11,312/11,314                                        |
| FoodOn     | 30,995         | 2,383        | Atomic SI: 768,486/96,060/96,062 <br /> Complex SI: 3,754/1,850/13,080 |
| GO         | 43,303         | 11,456       | Atomic SI: 772,870/96,608/96,610 <br /> Complex SI: 72,318/9,040/9,040 |
| MNLI       | -              | -            | biMNLI: 235,622/26,180/12,906                                          |

</small>

## Usage

Users have two options for accessing the OntoLAMA datasets. They can either download the datasets directly from [Zenodo](https://doi.org/10.5281/zenodo.6480540) or use the [Huggingface Datasets platform](https://huggingface.co/datasets/krr-oxford/OntoLAMA). 

If using Huggingface, users should first install the `dataset` package:

```bash
pip install datasets
```

Then, a dataset can be accessed by:

```python
from datasets import load_dataset
# dataset = load_dataset("krr-oxford/OntoLAMA", dataset_name)
# for example, loading the Complex SI dataset of Go
dataset = load_dataset("krr-oxford/OntoLAMA", "go-complex-SI") 
```

Options of `dataset_name` include:

- `"bimnli"` (from MNLI)
- `"schemaorg-atomic-SI"` (from Schema.org)
- `"doid-atomic-SI"` (from DOID)
- `"foodon-atomic-SI"`, `"foodon-complex-SI"` (from FoodOn)
- `"go-atomic-SI"`, `"go-complex-SI"` (from Go)

After loading the dataset, a particular data split can be accessed by:

```python
dataset[split_name]  # split_name = "train", "validation", or "test"
```

Please refer to the [Huggingface page](https://huggingface.co/datasets/krr-oxford/OntoLAMA) for examples of data points and explanations of data fields.

If downloading from Zenodo, users can simply target on specific `.jsonl` files.


## Prompt-based Probing

$\textsf{OntoLAMA}$ adopts the prompt-based probing approach to examine an LM's knowledge. Specifically, it wraps the verbalised sub-concept and super-concept into a template with a masked position; the LM is expected to predict the masked token and determine whether there exists a subsumption relationship between the two concepts.

> The verbalisation algorithm has been implemented as a separate ontology processing module, see 
[verbalise ontology concepts](https://krr-oxford.github.io/DeepOnto/verbaliser/).

To conduct probing, users can write the following code into a script, e.g., `probing.py`:

```python
from openprompt.config import get_config
from deeponto.probe.ontolama import run_inference

run_inference(config, args)
```

Then, run the script with the following command:

```bash
python probing.py --config_yaml config.yaml
```

See an example of `config.yaml` at [`DeepOnto/scripts/ontolama/config.yaml`](https://github.com/KRR-Oxford/DeepOnto/blob/main/scripts/ontolama/config.yaml)

The template file for the SI task (two templates) is located in [`DeepOnto/scripts/ontolama/si_templates.txt`](https://github.com/KRR-Oxford/DeepOnto/blob/main/scripts/ontolama/si_templates.txt).

The template file for the biMNLI task (two templates) is located in [`DeepOnto/scripts/ontolama/nli_templates.txt`](https://github.com/KRR-Oxford/DeepOnto/blob/main/scripts/ontolama/nli_templates.txt).

The label word file for both SI and biMNLI tasks is located in [`DeepOnto/scripts/ontolama/label_words.jsonl`](https://github.com/KRR-Oxford/DeepOnto/blob/main/scripts/ontolama/label_words.jsonl).

