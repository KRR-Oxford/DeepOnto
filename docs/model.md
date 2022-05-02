## BERTMap

`BERTMap` is a BERT-based ontology alignment system. It defines a thesarus from class labels of input ontologies to construct text semantics corpora for fine-tunig a BERT classifier. Such classifier is used for computing cross-ontology mappings that indicate domain equivalence. Structure-based mapping extension and logic-based mapping repair are used for refining the scored mappings in the `global_match` mode.

### Configurations
> See an example `config` file in `./config/bertmap.json`.

- `lab_props [list]`: specify the which annotation properties to be used for extracting class labels, default is `[label]`.
- `tokenizer [dict]`:
  - `type [str]`: the type of the tokenizer is either `pretrained` (sub-word-based, learnable) or `rule_based` (word-level).
  - `path [str]`: the path to pre-defined tokenizer is either a *huggingface* model path (for pretrained) or a *spacy* library path (for rule_based), e.g., `emilyalsentzer/Bio_ClinicalBERT` from BioClinicalBERT, `en_core_web_sm` from *spacy* library.
- `search [dict]`:
  - `n_best [int]`: the number of top ranked prediction mappings preserved for each source class.
  - `cand_pool_size [int]`: the number of top ranked (in terms of idf scores) target classes considered in candidate selection for each source class.
  - `apply_string_match [bool]`: apply string matching to filter "easy" mappings or not, default is `true`.
  - `match_src2tgt [bool]`: apply global matching for each class in the source ontology.
  - `match_tgt2src [bool]`: apply global matching for each class in the target ontology.
  - `extension_threshold [float]`: threshold for mapping extension.
- `corpora [dict]`:
  - `apply_transitivity [bool]`: apply transitivity closure to extracted synonyms or not.
  - `neg_ratio [int]`: the number of non-synonym pairs sampled for each synonym pair.
  - `train_mappings_path [str]`: path to the tsv file that stores the training mappings, default is `None`.
  - `val_mappings_path [str]`: path to the tsv file that stores the validation mappings, default is `None`.
  - `test_mappings_path [str]`: path to the tsv file that stores the testing mappings, default is `None`.
  - `null_mappings_path [str]`: path to the tsv file that stores the null mappings (ignored in evaluation), default is `None`.
  - `aux_onto_paths [List[str]]`: a list of paths to auxiliary ontologies for data augmentation.
- `bert [dict]`:
  - `pretrained_path [str]`: path to the pre-trained BERT model from the *huggingface* library.
  - `max_length [int]`: the maximum length of the tokens in BERT's inputs.
  - `num_epochs [int]`: the number of epochs for fine-tuning.
  - `batch_size_for_training`: the batch size for fine-tuning a BERT model.
  - `batch_size_for_prediction`: the batch szie for mapping prediction using the fine-tuned BERT model.
  - `early_stop_patience`: the number of patiences for early stopping, default is `None`.
  - `device_num`: the id of GPU device for mapping prediction; for fine-tuning, it by default uses multiple GPUs.

## EditSimiarity & StringMatch

`EditSimiarity` is a simple rule-based ontology alignment system that computes the normalized edit similarities (1 - normalized edit distance) between class labels and use the maximum of them as the mappping score. It performs surprisingly well for ontology pairs that have a similar naming scheme. `StringMatch` is a special case of `EditSimiarity` that considers only mapping scores of 1.0.

### Configurations
> See an example `config` file in `./config/edit_sim.json`.
> 
- `lab_props [list]`: specify the which annotation properties to be used for extracting class labels, default is `[label]`.
- `tokenizer [dict]`:
  - `type [str]`: the type of the tokenizer is either `pretrained` (sub-word-based, learnable) or `rule_based` (word-level).
  - `path [str]`: the path to pre-defined tokenizer is either a *huggingface* model path (for pretrained) or a *spacy* library path (for rule_based), e.g., `emilyalsentzer/Bio_ClinicalBERT` from BioClinicalBERT, `en_core_web_sm` from *spacy* library.
- `search [dict]`:
  - `n_best [int]`: the number of top ranked prediction mappings preserved for each source class.
  - `cand_pool_size [int]`: the number of top ranked (in terms of idf scores) target classes considered in candidate selection for each source class.
  - `apply_string_match [bool]`: apply string matching to filter "easy" mappings or not, default is `true`.
  - `match_src2tgt [bool]`: apply global matching for each class in the source ontology.
  - `match_tgt2src [bool]`: apply global matching for each class in the target ontology.