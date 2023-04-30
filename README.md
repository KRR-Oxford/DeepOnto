<!---
Copyright 2021 Yuan He (KRR-Oxford). All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

<p align="center">
  <a href="https://krr-oxford.github.io/DeepOnto/">
    <img alt="deeponto" src="https://raw.githubusercontent.com/KRR-Oxford/DeepOnto/main/docs/images/icon.svg">
  </a>
</p>

<p align="center">
    <a href="https://github.com/KRR-Oxford/DeepOnto/blob/main/LICENSE">
        <img alt="license" src="https://img.shields.io/github/license/KRR-Oxford/DeepOnto">
    </a>
    <a href="https://krr-oxford.github.io/DeepOnto/">
        <img alt="docs" src="https://img.shields.io/badge/website-online-informational">
    </a>
    <a href="https://pypi.org/project/deeponto/">
        <img alt="pypi" src="https://img.shields.io/pypi/v/deeponto">
    </a>
</p>

<p align="center">
  A package for ontology engineering with deep learning. 
</p>


**News** :newspaper:

- [X] Deploy the `deeponto.subs.bertsubs` and `deeponto.onto.pruning` modules (**v0.7.0**).
- [X] Deploy the `deeponto.probe.ontolama` and `deeponto.onto.verbalisation` modules (**v0.6.0**). 
- [X] Rebuild the whole package based on the OWLAPI; remove owlready2 from the essential dependencies (from **v0.5.x**). 

The complete changelog is available at: [repository](https://github.com/KRR-Oxford/DeepOnto/blob/main/docs/changelog.md) or [website](https://krr-oxford.github.io/DeepOnto/changelog/).

## About

$\textsf{DeepOnto}$ aims to provide tools for implementing deep learning models, constructing resources, and conducting evaluation
for various ontology engineering purposes.

- **Documentation**: *<https://krr-oxford.github.io/DeepOnto/>*.
- **Github Repository**: *<https://github.com/KRR-Oxford/DeepOnto>*. 
- **PyPI**: *<https://pypi.org/project/deeponto/>*. 


## Installation

### OWLAPI

$\textsf{DeepOnto}$ relies on [OWLAPI](http://owlapi.sourceforge.net/) version 4 (written in Java) for ontologies. 

We use what has been implemented in [mOWL](https://mowl.readthedocs.io/en/latest/index.html) that uses [JPype](https://jpype.readthedocs.io/en/latest/) to bridge Python and Java Virtual Machine (JVM). 


!!! Warning
  
    According to [mOWL](https://mowl.readthedocs.io/en/latest/index.html), the current integration with OWLAPI can **work on Linux or Mac OS** but **not Windows**.

### Pytorch

$\textsf{DeepOnto}$ relies on [Pytorch](https://pytorch.org/) for deep learning framework.

Configure Pytorch installation with CUDA support using, for example:

```bash
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

Basic usage of Ontology does not rely on GPUs, but for efficient deep learning model training, please make sure
`torch.cuda.is_available()` returns `True`.

### Install from PyPI

Other dependencies are specified in `setup.cfg` and `requirements.txt` which are supposed to be installed along with `deeponto`.

```bash
# requiring Python>=3.8
pip install deeponto
```

### Use Git Repository

One can git clone the repository without installing through PyPI and install the dependencies manually by:

```bash
pip install -r requirements.txt
```


## Main Features

<!-- <p align="center">
  <img alt="deeponto" src="https://raw.githubusercontent.com/KRR-Oxford/DeepOnto/main/docs/images/deeponto.svg" height="500" style="width: 80%;">
</p> -->

### Ontology Processing

The base class of $\textsf{DeepOnto}$ is [`Ontology`][deeponto.onto.Ontology], which encapsulates and extends the features of the OWLAPI library for processing ontologies. See quick usage at [load an ontology](https://krr-oxford.github.io/DeepOnto/ontology/). Several essential modules that revolve around [`Ontology`][deeponto.onto.Ontology] are then built to enhance the core ontology API, including:

- **Ontology Reasoning**: Each instance of $\textsf{DeepOnto}$ has an[`OntologyReasoner`][deeponto.onto.OntologyReasoner] as its attribute, which handles all the reasoning activities about the ontology such as checking consistency and entailment.

- **Ontology Pruning**: This module aims to incorporate pruning algorithms for extracting a sub-ontology from an input ontology.

- **Ontology Verbalisation**: The recursive ontology verbaliser originally proposed in [4] is implemented here as an essential module for briding ontologies and texts. See [verbalising ontology concepts](https://krr-oxford.github.io/DeepOnto/verbaliser).

- We have a plan of releasing more modules to support ontology normalisation, ontology-to-graph transformation, and more.

<!-- - **Ontology Normalisation** (not ready):   -->

### Tools and Resources

Individual tools and resources are implemented based on the core ontology API. Currently, $\textsf{DeepOnto}$ supports the following:

- **BERTMap** [1] is a BERT-based *ontology matching* (OM) system originally developed in [repo](https://github.com/KRR-Oxford/BERTMap) but is now maintained in $\textsf{DeepOnto}$. See [ontology Matching with BERTMap & BERTMapLt](https://krr-oxford.github.io/DeepOnto/bertmap/).

- **Bio-ML** [2] is an OM resource that has been used in the [Bio-ML track of the OAEI](https://www.cs.ox.ac.uk/isg/projects/ConCur/oaei/). See [bio-ML specifications](https://krr-oxford.github.io/DeepOnto/bio-ml/).

- **BERTSubs** [3] is a system for ontology subsumption prediction. We are working on transforming its original [experimental code](https://gitlab.com/chen00217/bert_subsumption) to this project.

## License

!!! license "License"

    Copyright 2021-2023 Yuan He.
    Copyright 2023 Yuan He, Jiaoyan Chen.
    All rights reserved.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at *<http://www.apache.org/licenses/LICENSE-2.0>*

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

## Publications

- [1] *Yuan He‚ Jiaoyan Chen‚ Denvar Antonyrajah and Ian Horrocks.* **BERTMap: A BERT−Based Ontology Alignment System**. In Proceedings of 36th AAAI Conference on Artificial Intelligence 2022 (AAAI-2022). /[arxiv](https://arxiv.org/abs/2112.02682)/ /[aaai](https://ojs.aaai.org/index.php/AAAI/article/view/20510)/  <a name="bertmap_paper"></a>
- [2] *Yuan He‚ Jiaoyan Chen‚ Hang Dong, Ernesto Jiménez-Ruiz, Ali Hadian and Ian Horrocks.* **Machine Learning-Friendly Biomedical Datasets for Equivalence and Subsumption Ontology Matching**. The 21st International Semantic Web Conference (ISWC-2022, **Best Resource Paper Candidate**). /[arxiv](https://arxiv.org/abs/2205.03447)/ /[iswc](https://link.springer.com/chapter/10.1007/978-3-031-19433-7_33)/  <a name="bioml_paper"></a>
- [3] *Jiaoyan Chen, Yuan He, Yuxia Geng, Ernesto Jiménez-Ruiz, Hang Dong and Ian Horrocks.* **Contextual Semantic Embeddings for Ontology Subsumption Prediction**. World Wide Web Journal (accepted, BERTSubs paper). /[arxiv](https://arxiv.org/abs/2202.09791)/  <a name="bertsubs_paper"></a>
- [4] *Yuan He‚ Jiaoyan Chen, Ernesto Jiménez-Ruiz, Hang Dong and Ian Horrocks.* **Language Model Analysis for Ontology Subsumption Inference**. 2023 (Under review). /[arxiv](https://arxiv.org/abs/2302.06761)/  <a name="ontolama_paper"></a>


----------------------------------------------------------------

Please report any bugs or queries by raising a GitHub issue or sending emails to the maintainers (Yuan He or Jiaoyan Chen) through:

> first_name.last_name@cs.ox.ac.uk
