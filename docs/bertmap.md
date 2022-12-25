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

# BERTMap

`BERTMap` is a BERT-based ontology alignment system. It defines a thesarus from class labels of input ontologies to construct text semantics corpora for fine-tunig a BERT classifier. Such classifier is used for computing cross-ontology mappings that indicate domain equivalence. Structure-based mapping extension and logic-based mapping repair are used for refining the scored mappings in the `global_match` mode.

!> Because of the limitations of the `owlready2` library (only unique IRIs are allowed), if loading ontologies of the same IRI but different versions, e.g., as auxiliary sources for complementary corpus construction, the users need to manually change the IRIs of such ontologies for successful loading.

`BERTMapLt` (or `EditSimilarity`) is the rule-base component of `BERTMap` which computes the normalized edit similarities (1 - normalized edit distance) between class labels and use the maximum of them as the mappping score. It performs surprisingly well for ontology pairs that have a similar naming scheme. `StringMatch` is a special case of `EditSimiarity` that considers only mapping scores of 1.0 and it was aggregated into BERTMap model for accelerating prediction. 

## Use BERTMap or BERTMap-Lite

There are two modes for OM with BERTMap or BERTMap-Lite: `global_match` and `pair_score`. `global_match` aims to compute mappings given two input ontologies and `pair_score` is for scoring provided class pairs. For both modes, the source and target input ontologies are required; for `pair_score`, an input file containing the unscored class pairs and a flag that indicates the direction of the class pairs (`src2tgt` or `tgt2src`) are further required (see [below](using_deeponto?id=pair_scoring)).

?> `Precision`, `Recall`, and `F-score` are frequently used in evaluating `global_match`; whereas `Accuracy` can be used in evaluating any class pair input, ranking-based metrics like `Hits@K` and `MRR` are used in evaluating `pair_score` when the input class pairs are grouped as *1 positive + N negatives*. See our [resource paper](https://arxiv.org/abs/2205.03447) for detailed guidance of ontology matching evaluation.

### Global Matching

In this mode, the OM model is expected to search for all plausible cross-ontology class pairs that are semantically related (through, e.g., equivalence, subsumption). To search and compute mappings globally, the OM model needs to address *(1)* how semantically close two classes are; *(2)* how to search efficiently (naive traversal takes quadratic time). Futher refinement such as extension and repair are also optional for postprocessing.

Parameters for `bertmap.py` in `global_match` mode:

- **saved_path**(*str*): the path to the main output directory.
- **src_onto**(*str*): the path to the source ontology file.
- **tgt_onto**(*str*): the path to the target ontology file.
- **config_path**(*str*): the path to the configuration file, default minimal configurations for each OM model is available at `deeponto_repo/cofig`.

Example usage of `bertmap.py` for global matching:

**Step 1**: Run the script with above arguments specified.

```bash
# matching DOID and ORDO with minimal configurations
python bertmap.py \
--saved_path "./experiment/doid2ordo" \  
--src_onto "./data/doid.owl" \
--tgt_onto "./data/ordo.owl" \
--config_path "./config/bertmap.json"
```

**Step 2**: Choose `global_match` and an implemented OM model.

```bash
######################################################################
###                   Choose a Supported OM Mode                   ###
######################################################################

[0]: global_match
[1]: pair_score
Enter a number: 0

######################################################################
###                 Choose an Implemented OM Model                 ###
######################################################################

[0]: bertmap
[1]: bertmap-lite (edit-sim)
Enter a number: 0
```

Then the script will do the followings:
- Load and parse the input ontologies;
- Train the scoring function with constructed data if the selected model is learning-based;
- Globally compute the mappings and apply any refinement steps if any.


### Pair Scoring

In this mode, the OM model is expected to compute the matching scores for input class pairs. Compared to Global Matching, extra arguments for unscored mappings (in `.tsv`) and its flag (`src2tgt` or `tgt2src`) are needed. The unscored mappings come from two types of `.tsv` files:

- With two columns: `"SrcEntity"` and `"TgtEntity"`, which are pairs of class IRIs from source and target ontologies, respectively.
- With three columns: `"SrcEntity"`, `"TgtEntity"`, and `"TgtCandidates"`, which are pairs of source-target class IRIs and candidates generated from target ontologies. Please refer to [*local ranking*](om_resources?id=evaluation_framework) evaluation for details.

Parameters for `bertmap.py` in `pair_score` mode:

- **saved_path**(*str*): the path to the main output directory.
- **src_onto**(*str*): the path to the source ontology file.
- **tgt_onto**(*str*): the path to the target ontology file.
- **config_path**(*str*): the path to the configuration file, default minimal configurations for each OM model is available at `deeponto_repo/cofig`.
- **--to_be_scored_mappings_path**(*str*): the path to the to-be-scored mappings in `.tsv` as described above.
- **--to_be_scored_mappings_flag**(*str*): the flag that indicates the direction of the to-be-scored class pairs (`src2tgt` or `tgt2src`).

Example usage of `bertmap.py` for pair scoring:

**Step 1**: Run the script with above arguments specified:

```bash
# scoring class pairs DOID and ORDO with minimal configurations
python bertmap.py \
--saved_path "./experiment/doid2ordo" \  
--src_onto "./data/doid.owl" \
--tgt_onto "./data/ordo.owl" \
--config_path "./config/bertmap.json"
--to_be_scored_mappings_path "./data/doid2ordo_unscored.tsv" \
--to_be_scored_mappings_flag "src2tgt"
```

**Step 2**: Choose `pair_score` and an implemented OM model.

```bash
######################################################################
###                   Choose a Supported OM Mode                   ###
######################################################################

[0]: global_match
[1]: pair_score
Enter a number: 1

######################################################################
###                 Choose an Implemented OM Model                 ###
######################################################################

[0]: bertmap
[1]: bertmap-lite (edit-sim)
Enter a number: 0
```

Then the script will do the followings:
- Load and parse the input ontologies;
- Train the scoring function with constructed data if the selected model is learning-based;
- Compute the scores for the input mappings and save the results in `./${exp_dir}/${model_name}/pair_score/${flag}`. 


## Configurations and Output Formats

### BERTMap Configurations
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

### BERTMap Outputs
BERTMap's outputs are stored in either `$exp_dir/bertmap/global_match` or `$exp_dir/bertmap/pair_score` depending on which mode is selected.  

For `pair_score`, depending on the alignment flag (`src2tgt` or `tgt2src`) of the input unscored mappings (in `.tsv`), the output scored mappings will be stored in `exp_dir/bertmap/pair_score/src2tgt` or `exp_dir/bertmap/pair_score/tgt2src`.

For `global_match`: 
- If both `match_src2tgt` and `match_tgt2src` are set to be `true` in config, then the mappings predicted from both directions will be generated in `exp_dir/bertmap/global_match/src2tgt` and `exp_dir/bertmap/global_match/tgt2src`, respectively. 
- The best mapping type (`src2tgt`, `tgt2src`, or `combined`) is determined by the performance on validation mappings (`src2tgt` by default if without validation). 
- Mapping extension and repair is applied on mappings that obtain best validation results; however, when `combined` wins, the refinement procedure is applied to both. 
- Final outputs will be stored in `exp_dir/bertmap/global_match/final_output.maps.{pkl,json,tsv}` after the mapping refinement.
- Evaluation conducted on the intermediate prediction mappings in `src2tgt` or `tgt2src` needs to add in the mapping score threshold argument (often set between $0.999$ and $0.9995$); the threshold is not required when evaluating the final output mappings as it has been determined by the model.


### BERTMap-Lite Configurations
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


### BERTMap-Lite Outputs

The outputs of this model are similar to those of BERTMap but without any automatic validation results and further mapping refinement. Evaluation should be conducted on `src2tgt` or `tgt2src` prediction mappings with a chosen mapping threshold. Note that by setting `threshold = 1.0`, the EditSimiarity model is exactly the same as the StringMatch model.

----------------------------------------------------------------

## OM Evaluation

We provide an OM evaluation script for global matching (to compute `Precision`, `Recall`, and `F-score` on output mappings against reference mappings) and local ranking (to compute `Hits@K` and `MRR` on reference mappings against selected negative candidates).

Parameters for `om_eval.py` in `global_match` mode:

- **saved_path**(*str*): the path to the main output directory.
- **pred_path**(*str*): the path to the prediction mappings saved using [`OntoMappings`](data_structures?id=onto_mapping).
- **ref_path**(*str*): the path to the reference mappings saved in `.tsv`.
- **null_ref_path**(*str*): the path to the reference mappings (in `.tsv`) that should be ignored in calculating Precision, Recall, and F-score, e.g., training and validation mappings when evaluating on the testing mappings.
- **--threshold**(*float*): mappings with scores $\leq$ the threshold $\lambda \in [0, 1]$ are considered in the evaluation.
- **--show_more_f_scores**(*str*): whether or not to display more variants of F-score besides F1.


Example usage of `om_eval.py` for global matching evaluation:

**Step 1**: Run the script with above arguments specified.


```bash
python om_eval.py \
--saved_path "./om_results" \  
--pred_path "./om_exp/bertmap/global_match" \ 
--ref_path "./data/test.tsv" \
--null_ref_path "./data/train+val.tsv" \
--threshold 0.9995 \  
--show_more_f_scores False
```

**Step 2**: Choose `global_match` evaluation mode.
```
######################################################################
###                    Choose a Evaluation Mode                    ###
######################################################################

[0]: global_match
[1]: local_rank
Enter a number: 0

######################################################################
###          Eval using P, R, F-score with threshold: 0.0          ###
######################################################################

{
    "P": 0.966,
    "R": 0.606,
    "f_score": 0.745
}
```

Parameters for `om_eval.py` in `local_rank` mode:

- **saved_path**(*str*): the path to the main output directory.
- **pred_path**(*str*): the path to the prediction mappings saved using [`OntoMappings`](data_structures?id=onto_mapping).
- **ref_anchored_path**(*str*): the path to the file of reference mappings and their corresponding candidate mappings saved in `.tsv`.
- **--hits_at**(*Tuple[int]*): the numbers of hits considered in computing Hits@K, default is `[1, 5, 10, 30, 100]`.

Example usage of `om_eval.py` for local ranking evaluation:


**Step 1**: Run the script with above arguments specified.


```bash
python om_eval.py \
--saved_path "./om_results" \  
--pred_path "./om_exp/bertmap/pair_score/src2tgt" \ 
--ref_anchored_path "./data/test.cands.tsv" \
--hits_at 1 5 10 30 100
```


**Step 2**: Choose `local_rank` evaluation mode.
```
######################################################################
###                    Choose a Evaluation Mode                    ###
######################################################################

[0]: global_match
[1]: local_rank
Enter a number: 1

######################################################################
###                     Eval using Hits@K, MRR                     ###
######################################################################

523987/523987 of scored mappings are filled to corresponding anchors.
{
    "MRR": 0.919,
    "Hits@1": 0.876,
    "Hits@5": 0.976,
    "Hits@10": 0.99,
    "Hits@30": 0.997,
    "Hits@100": 1.0
}
```


