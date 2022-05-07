This page describes important data structures implemented in DeepOnto for processing ontology data.

## Ontology

> Source file: `deeponto/onto/ontology.py`

The `Ontology` class relies on `owlready2` for loading OWL object, and then applies extra text-level processing on labels of selected *annotation properties* and optionally construct an *inverted index* from these class labels. 

Arguments:
- `onto_path` (str): the path of an ontology file (preferrably in `.owl` format).
- `lab_props` (List[str], default to `["label"]`): a list of selected annotation properties for text processing such as `["label", "preferred_name", "synonym"]`.
- `tokenizer` (Tokenizer, *optional*, default to `None`) 

#### Loading a new ontology (preferrably in `.owl` format)

```python
from deeponto.onto import Ontology

# without tokenizer and thus without inverted index
onto = Ontology.from_new(onto_path="./onto_files/pizza.owl", lab_props=)



```



