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

## Essential Dependencies

DeepOnto is mainly extended from the following packages:

- [OwlReady2](https://owlready2.readthedocs.io/) for basic ontology processing.
- [Transformers](https://github.com/huggingface/transformers) for pre-trained language models.

## Implemented Models

| Name            | Task                            | Type              |
| :-------------  | :---------------                | :--------------   |
| BERTMap         | Ontology Matching (Equivalence) | Learning-based    |
| StringMatch     | Ontology Matching (Equivalence) | Rule-based        |
| EditSimiarity   | Ontology Matching (Equivalence) | Rule-based        |

> See https://krr-oxford.github.io/DeepOnto/#/intro for quick usage.

## License

Copyright 2021 Yuan He (KRR-Oxford). All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

## Publications

- *Yuan He‚ Jiaoyan Chen‚ Denvar Antonyrajah and Ian Horrocks.* **BERTMap: A BERT−based Ontology Alignment System**. In Proceedings of 36th AAAI Conference on Artificial Intelligence 2022 (AAAI-22). To Appear.
