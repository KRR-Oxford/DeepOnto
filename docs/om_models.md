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

# Ontology Matching Models

## BERTMap

`BERTMap` is a BERT-based ontology alignment system. It defines a thesarus from class labels of input ontologies to construct text semantics corpora for fine-tunig a BERT classifier. Such classifier is used for computing cross-ontology mappings that indicate domain equivalence. Structure-based mapping extension and logic-based mapping repair are used for refining the scored mappings in the `global_match` mode.

!> Because of the limitations of the `owlready2` library (only unique IRIs are allowed), if loading ontologies of the same IRI but different versions, e.g., as auxiliary sources for complementary corpus construction, the users need to manually change the IRIs of such ontologies for successful loading.

### Configurations
> See an example config file in `config/bertmap.json`.

- **lab_props**(*list*): specify the which annotation properties to be used for extracting class labels, default is `["http://www.w3.org/2000/01/rdf-schema#label"]` (*rdfs:label*). 

?> This parameter is very important for BERTMap as it tries to establish a synonym classifier trained on sufficient class labels; auxiliary ontologies are recommended for augmenting training data if the input ontologies are deficient in class labels.

- **tokenizer**(*dict*):
  - **type**(*str*): the type of the tokenizer is either `pretrained` (sub-word-based, learnable) or `rule_based` (word-level).
  - **path**(*str*): the path to pre-defined tokenizer is either a [huggingface model](https://huggingface.co/models) path (for `pretrained`) or a *spacy* library path (for `rule_based`), e.g., `emilyalsentzer/Bio_ClinicalBERT` from BioClinicalBERT, `en_core_web_sm` from [spacy library](https://spacy.io/).
- **search**(*dict*):
  - **n_best**(*int*): the number of top ranked prediction mappings preserved for each source class.
  - **cand_pool_size**(*int*): the number of top ranked (in terms of idf scores) target classes considered in candidate selection for each source class.
  - **apply_string_match**(*bool*): apply string matching to filter "easy" mappings or not, default is `true`.
  - **match_src2tgt**(*bool*): apply global matching for each class in the source ontology.
  - **match_tgt2src**(*bool*: apply global matching for each class in the target ontology.
  - **extension_threshold**(*float*): threshold for mapping extension.
- **corpora**(*dict*):
  - **apply_transitivity**(*bool*): apply transitivity closure to extracted synonyms or not.
  - **neg_ratio**(*int*): the number of non-synonym pairs sampled for each synonym pair.
  - **train_mappings_path**(*str*): path to the tsv file that stores the training mappings, default is `None`.
  - **val_mappings_path**(*str*): path to the tsv file that stores the validation mappings, default is `None`.
  - **test_mappings_path**(*str*): path to the tsv file that stores the testing mappings, default is `None`.
  - **null_mappings_path**(*str*): path to the tsv file that stores the null mappings (ignored in evaluation), default is `None`.
  - **aux_onto_paths**(*List[str]*): a list of paths to auxiliary ontologies for data augmentation.
- **bert**(*dict*):
  - **pretrained_path**(*str*): path to the pre-trained BERT model from the *huggingface* library.
  - **max_length**(*int*): the maximum length of the tokens in BERT's inputs.
  - **num_epochs**(*int*): the number of epochs for fine-tuning.
  - **batch_size_for_training**(*int*): the batch size for fine-tuning a BERT model.
  - **batch_size_for_prediction**(*int*): the batch szie for mapping prediction using the fine-tuned BERT model.
  - **early_stop_patience**(*Optional[int]*): the number of patiences for early stopping, default is `None`.
  - **device_num**(*int*): the id of GPU device for mapping prediction; for fine-tuning, it by default uses multiple GPUs.

### Outputs
BERTMap's outputs are stored in either `$exp_dir/bertmap/global_match` or `$exp_dir/bertmap/pair_score` depending on which mode is selected.  

For `pair_score`, depending on the alignment flag (`src2tgt` or `tgt2src`) of the input unscored mappings (in `.tsv`), the output scored mappings will be stored in `exp_dir/bertmap/pair_score/src2tgt` or `exp_dir/bertmap/pair_score/tgt2src`.

For `global_match`: 
- If both `match_src2tgt` and `match_tgt2src` are set to be `true` in config, then the mappings predicted from both directions will be generated in `exp_dir/bertmap/global_match/src2tgt` and `exp_dir/bertmap/global_match/tgt2src`, respectively. 
- The best mapping type (`src2tgt`, `tgt2src`, or `combined`) is determined by the performance on validation mappings (`src2tgt` by default if without validation). 
- Mapping extension and repair is applied on mappings that obtain best validation results; however, when `combined` wins, the refinement procedure is applied to both. 
- Final outputs will be stored in `exp_dir/bertmap/global_match/final_output.maps.{pkl,json,tsv}` after the mapping refinement.
- Evaluation conducted on the intermediate prediction mappings in `src2tgt` or `tgt2src` needs to add in the mapping score threshold argument (often set between $0.999$ and $0.9995$); the threshold is not required when evaluating the final output mappings as it has been determined by the model.

## EditSimiarity (BERTMap-Lite)

`EditSimiarity` is a simple rule-based ontology alignment system that computes the normalized edit similarities (1 - normalized edit distance) between class labels and use the maximum of them as the mappping score. It performs surprisingly well for ontology pairs that have a similar naming scheme. `StringMatch` is a special case of `EditSimiarity` that considers only mapping scores of 1.0 and it was aggregated into BERTMap model for accelerating prediction. Since it can be seen as a part of BERTMap, we also refer to it as `BERTMap-Lite`.

### Configurations
> See an example config file in `config/edit_sim.json`.

- **lab_props**(*list*): specify the which annotation properties to be used for extracting class labels, default is `["http://www.w3.org/2000/01/rdf-schema#label"]` (*rdfs:label*). 
- **tokenizer**(*dict*):
  - **type**(*str*): the type of the tokenizer is either `pretrained` (sub-word-based, learnable) or `rule_based` (word-level).
  - **path**(*str*): the path to pre-defined tokenizer is either a [huggingface model](https://huggingface.co/models) path (for `pretrained`) or a *spacy* library path (for `rule_based`), e.g., `emilyalsentzer/Bio_ClinicalBERT` from BioClinicalBERT, `en_core_web_sm` from [spacy library](https://spacy.io/).
- **search**(*dict*):
  - **n_best**(*int*): the number of top ranked prediction mappings preserved for each source class.
  - **cand_pool_size**(*int*): the number of top ranked (in terms of idf scores) target classes considered in candidate selection for each source class.
  - **match_src2tgt**(*bool*): apply global matching for each class in the source ontology.
  - **match_tgt2src**(*bool*: apply global matching for each class in the target ontology.


### Outputs

The outputs of EditSimilarity/StringMatch model are similar to those of BERTMap but without any automatic validation results and further mapping refinement. Evaluation should be conducted on `src2tgt` or `tgt2src` prediction mappings with a chosen mapping threshold. Note that by setting `threshold = 1.0`, the EditSimiarity model is exactly the same as the StringMatch model.