This page describes important data structures implemented in DeepOnto for processing ontology data.

## Ontology

> Source file: `deeponto/onto/ontology.py`

The `Ontology` class relies on `owlready2` for loading OWL object, and then applies extra text-level processing on labels of selected *annotation properties* and optionally construct an *inverted index* from these class labels. 

Arguments:
- `onto_path [str]`: the path of an ontology file (preferrably in `.owl` format).
- `lab_props [List[str]]`: a list of selected annotation properties for text processing, default is `["label"].
- `tokenizer [Optional[Tokenizer]]`: an instance of the `Tokenizer` class used for tokenizing the class labels, default is `None`.

Example usage:

**Create, save, and reload an ontology**:

```python
from deeponto.onto import Ontology
from deeponto.onto.text import Tokenizer

# use the sub-word tokenizer pre-trained with BERT
tokz = Tokenizer.from_pretrained("bert-base-uncased") 

onto = Ontology.from_new(
  onto_path="./onto_files/pizza.owl", 
  lab_props=["label", "hasExactSynonym", "prefLabel"],
  tokenizer=tokz 
)

# a .pkl file will be saved with processed information
# and the original ontology file will be copied to ./saved because
# the owlready2 ontology is not serializable 
onto.save_instance("./saved")

# reload the ontology without repeated processing
onto = Ontology.from_saved("./saved")

# see onto.__dict__ for useful information
```

**Use the inverted index for candidate selection**

```python
from deeponto.onto import Ontology
from deeponto.onto.text import Tokenizer

# use the sub-word tokenizer pre-trained with BERT
tokz = Tokenizer.from_pretrained("bert-base-uncased") 

onto = Ontology.from_new(
  onto_path="./onto_files/pizza.owl", 
  lab_props=["label", "hasExactSynonym", "prefLabel"],
  tokenizer=tokz 
)

# onto.class2idx: {class_name: class_index}
# onto.idx2class: {class_index: class_name}

# get the preprocessed labels for a class of index 10 in this ontology
# for Ontology Matching, the class should be retrieved from another ontology
labs = onto.idx2labs[10]

# tokens from the class labels
toks = tokz.tokenize_all(labs)

# retrieve ontology classes with idf scores
candidates = onto.idf_select(toks)

```




