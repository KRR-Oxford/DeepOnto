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
</p>

<p align="center">
  A package for ontology engineering with deep learning. 
</p>

## About <!-- {docsify-ignore} -->

DeepOnto aims to provide implemented deep learning models and an evaluation platform for various ontology engineering purposes. 

- **Documentation**: *https://krr-oxford.github.io/DeepOnto/#/*.
- **Github Repository**: *https://github.com/KRR-Oxford/DeepOnto*. 
- **PyPI**: *https://pypi.org/project/deeponto/*. (experimental)

To use DeepOnto scripts, it is sufficient to git the project and run the scripts directly; to build on a new project extending DeepOnto, please install DeepOnto from PyPI by:

```bash
pip install deeponto
```

## Essential Dependencies

DeepOnto is mainly extended from the following packages:

- [OwlReady2](https://owlready2.readthedocs.io/) for basic ontology processing.
- [OWLAPI](http://owlapi.sourceforge.net/) (in Java) for advanced ontology processing. The Python-Java interaction relies on what has been implemented in [mOWL](https://mowl.readthedocs.io/en/latest/index.html) which uses [JPype](https://jpype.readthedocs.io/en/latest/).
- [Transformers](https://github.com/huggingface/transformers) for pre-trained language models.

To use DeepOnto, please manually configure Pytorch installation using:

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

Then, install other dependencies in [`requirement.txt`](https://raw.githubusercontent.com/KRR-Oxford/DeepOnto/main/requirements.txt):

```bash
pip install -r requirements.txt
```

## Main Features

### Ontology Matching 

DeepOnto has implemented a family of BERT-based Ontology Matching (OM) models including:
- **BERTMap** and **BERTMapLt** (EditSim) for equivalence OM;
- **BERTSubs** (not ready) for subsumption OM.

?> Click [here](https://krr-oxford.github.io/DeepOnto/#/bertmap) for BERT-based OM.

It also incorporates the OM resource **Bio-ML**:
-  Download link: *https://doi.org/10.5281/zenodo.6510086* (CC BY 4.0 International);
-  Instructions: *https://krr-oxford.github.io/DeepOnto/#/om_resources*;
-  OAEI track: *https://www.cs.ox.ac.uk/isg/projects/ConCur/oaei/*.

?> Click [here](https://krr-oxford.github.io/DeepOnto/#/om_resources) for OM resources.

### Using OWLAPI Reasoner in Python

Thanks for [JPype](https://jpype.readthedocs.io/en/latest/), we can use the OWLAPI reasoner in Python for better integration with the deep learning modules. See the page [here](https://krr-oxford.github.io/DeepOnto/#/reasoning) for more information.

## License

Copyright 2021 Yuan He (KRR-Oxford). All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at *http://www.apache.org/licenses/LICENSE-2.0*

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## Publications

- *Yuan He‚ Jiaoyan Chen‚ Denvar Antonyrajah and Ian Horrocks.* **BERTMap: A BERT−Based Ontology Alignment System**. In Proceedings of 36th AAAI Conference on Artificial Intelligence 2022 (AAAI-2022). /[arxiv](https://arxiv.org/abs/2112.02682)/ /[aaai](https://ojs.aaai.org/index.php/AAAI/article/view/20510)/
- *Yuan He‚ Jiaoyan Chen‚ Hang Dong, Ernesto Jiménez-Ruiz, Ali Hadian and Ian Horrocks.* **Machine Learning-Friendly Biomedical Datasets for Equivalence and Subsumption Ontology Matching**. The 21st International Semantic Web Conference (ISWC-2022, **Best Resource Paper Candidate**). /[arxiv](https://arxiv.org/abs/2205.03447)/ /[iswc](https://link.springer.com/chapter/10.1007/978-3-031-19433-7_33)/
- *Jiaoyan Chen, Yuan He, Yuxia Geng, Ernesto Jiménez-Ruiz, Hang Dong and Ian Horrocks.* **Contextual Semantic Embeddings for Ontology Subsumption Prediction**. 2022 (Under review). /[arxiv](https://arxiv.org/abs/2202.09791)/
