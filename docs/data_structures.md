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

```





