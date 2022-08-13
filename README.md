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
- **PyPI**: *https://pypi.org/project/deeponto/*. (not fully ready)

## Essential Dependencies

DeepOnto is mainly extended from the following packages:

- [OwlReady2](https://owlready2.readthedocs.io/) for basic ontology processing.
- [Transformers](https://github.com/huggingface/transformers) for pre-trained language models.

To use DeepOnto, please manually configure Pytorch installation using:

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

Then, install other dependencies in [`requirement.txt`](https://raw.githubusercontent.com/KRR-Oxford/DeepOnto/main/requirements.txt):

```bash
pip install -r requirements.txt
```

## Implemented Models


<!-- #### **OntoAlign** -->

<!-- <div align="center"> -->

| Name                 | Task                            | Type              |
| :-------------       | :---------------                | :--------------   |
| BERTMap              | Ontology Matching (Equivalence) | Learning-based    |
| StringMatch          | Ontology Matching (Equivalence) | Rule-based        |
| EditSimiarity        | Ontology Matching (Equivalence) | Rule-based        |
| BERTSubs (not ready) | Ontology Matching (Subsumption) | Learning-based    |

<!-- </div> -->

?> Click [here](using_deeponto.md) for script usage.

## Ontology Matchng Resources

**Bio-ML** for equivalence and subsumption ontology matching:
-  Download link: *https://doi.org/10.5281/zenodo.6510086* (CC BY 4.0 International);
-  Instructions: *https://krr-oxford.github.io/DeepOnto/#/om_resources*;
-  OAEI track: *https://www.cs.ox.ac.uk/isg/projects/ConCur/oaei/*.

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

- *Yuan He‚ Jiaoyan Chen‚ Denvar Antonyrajah and Ian Horrocks.* **BERTMap: A BERT−Based Ontology Alignment System**. In Proceedings of 36th AAAI Conference on Artificial Intelligence 2022 (AAAI-2022). [arxiv](https://arxiv.org/abs/2112.02682) [aaai](https://ojs.aaai.org/index.php/AAAI/article/view/20510)
- *Yuan He‚ Jiaoyan Chen‚ Hang Dong, Ernesto Jiménez-Ruiz, Ali Hadian and Ian Horrocks.* **Machine Learning-Friendly Biomedical Datasets for Equivalence and Subsumption Ontology Matching**. The 21st International Semantic Web Conference (ISWC-2022, to appear). [arxiv](https://arxiv.org/abs/2205.03447)
- *Jiaoyan Chen, Yuan He, Yuxia Geng, Ernesto Jiménez-Ruiz, Hang Dong and Ian Horrocks.* **Contextual Semantic Embeddings for Ontology Subsumption Prediction**. 2022 (Under review). [arxiv](https://arxiv.org/abs/2112.10006)
