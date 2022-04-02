## BERTMap

`BERTMap` is a BERT-based ontology alignment system. It defines a thesarus from class labels of input ontologies to construct text semantics corpora for fine-tunig a BERT classifier. Such classifier is used for computing cross-ontology mappings that indicate domain equivalence. Structure-based mapping extension and logic-based mapping repair are used for refining the scored mappings in the `global_match` mode.

### Configurations
> See an example `config` file in `./config/bertmap.json`.

- `lab_props [list]`: specify the which annotation properties to be used for extracting class labels, default is `[label]`.
- `tokenizer [dict]`:
  - `type [str]`: the type of the tokenizer is either `pretrained` (sub-word-based, learnable) or `rule_based` (word-level).
  - `path [str]`: the path to pre-defined tokenizer is either a *huggingface* model path (for pretrained) or a *spacy* library path (for rule_based), e.g., `emilyalsentzer/Bio_ClinicalBERT` from BioClinicalBERT, `en_core_web_sm` from *spacy* library.

## EditSimiarity

`EditSimiarity` is a simple rule-based ontology alignment system that computes the normalized edit similarities (1 - normalized edit distance) between class labels and use the maximum of them as the mappping score. It performs surprisingly well for ontology pairs that have a similar naming scheme.

### Configurations
> See an example `config` file in `./config/edit_sim.json`.
> 
- `lab_props [list]`: specify the which annotation properties to be used for extracting class labels, default is `[label]`.
- `tokenizer [dict]`:
  - `type [str]`: the type of the tokenizer is either `pretrained` (sub-word-based, learnable) or `rule_based` (word-level).
  - `path [str]`: the path to pre-defined tokenizer is either a *huggingface* model path (for pretrained) or a *spacy* library path (for rule_based), e.g., `emilyalsentzer/Bio_ClinicalBERT` from BioClinicalBERT, `en_core_web_sm` from *spacy* library.