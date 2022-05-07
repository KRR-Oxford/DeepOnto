This page describes important data structures implemented in DeepOnto for processing ontology data.

## Ontology

> Source file: `deeponto/onto/ontology.py`

The `Ontology` class relies on `owlready2` for loading OWL object, and then applies extra text-level processing on labels of selected *annotation properties* and optionally construct an *inverted index* from these class labels. 

Arguments for `Ontology`:
- `onto_path [str]`: the path of an ontology file (preferrably in `.owl` format).
- `lab_props [List[str]]`: a list of selected annotation properties for text processing, default is `["label"].
- `tokenizer [Optional[Tokenizer]]`: an instance of the `Tokenizer` class used for tokenizing the class labels, default is `None`.

Example usage:

Create, save, and reload an ontology:

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

Use the inverted index for candidate selection:

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
# onto.idx2labs: {class_index: class_labels}
# the class names are abbreviated class IRIs (see deeponto/onto/iris.py)

# get the preprocessed labels for a class of index 10 in this ontology
# for Ontology Matching, the class should be retrieved from another ontology
labs = onto.idx2labs[10]

# tokens from the class labels
toks = tokz.tokenize_all(labs)

# retrieve ontology classes with idf scores
candidates = onto.idf_select(toks)

```

## Mappings

> Source file: `deeponto/onto/mapping.py`

Several python classes for entity (in string) mappings are defined here as main data structrues used for OM system outputs and evaluation.

<!-- Arguments for `EntityMapping`:
- `src_ent_name [str]`: name (abbreviated IRI) for the source entity -->


Example usage:

An entity IRI can be abbreviated provided that its ontology prefix has been defined in Deeponto (see `deeponto/onto/iris.py`).

```python
from deeponto.onto.text.text_utils import abbr_iri, unfold_iri

ent_iri = "http://purl.org/sig/ont/fma/fma7376"

abbr_iri(ent_iri)
# > "fma:fma7376"

unfold(abbr_iri(ent_iri))
# > "http://purl.org/sig/ont/fma/fma7376"
```

Create an equivalence mapping for `snomed:55940004` and `fma:fma54970`:

```python
from deeponto.onto.mapping import EntityMapping
from deeponto.onto.text.text_utils import abbr_iri, unfold_iri

src_ent_iri = "http://snomed.info/id/55940004"
tgt_ent_iri = "http://purl.org/sig/ont/fma/fma54970"

src2tgt_map = EntityMapping(
  src_ent_name=abbr_iri(src_ent_iri), 
  tgt_ent_name=abbr_iri(tgt_ent_iri), 
  rel="=", 
  score=1.0
)
```

Create an `OntoMappings` (a collection of ranked entity mappings) object and feed mappings:

```python
from deeponto.onto.mapping import OntoMappings

onto_maps = OntoMappings(flag="src2tgt", n_best=10, rel="=")

# add a single mapping
onto_maps.add(src2tgt_map) # EntityMapping

# add many mappings
onto_maps.add_many(*list_of_src2tgt_maps) # List[EntityMapping]

# ranked mappings in [dict] with source entities as keys
onto_maps.ranked

# return top 1 scored mappings with scores >=0.9 for each source entity
onto_maps.topK(threshold=0.9, K=1)

# return all mappings as tuples
onto_maps.to_tuples()

# save the mappings
# a .pkl file is saved for the object and a .json file is saved for readability
onto_maps.save_instance("./src2tgt_maps")

# reload the mappings
onto_maps = OntoMappings.from_saved("./src2tgt_maps")

```

`AnchoredOntoMappings` is the data structure for the collection of entity mappings w.r.t. each anchor mapping. For example, to generate negative candidates for each reference mapping for *local ranking evaluation* (see link-to-be-updated), we need to use the reference class pair as a double-key and add candidate mapping correspondingly.

```python
from deeponto.onto.mapping import AnchoredOntoMappings

anchored_onto_maps = AnchoredOntoMappings(flag="src2tgt", n_best=10, rel="=")

# add a candidate mapping for a anchor (reference) mapping
# NOTE: source entities in anchor_map and cand_map are the same
anchored_onto_maps.add(anchor_map, cand_map)  

# generate unscored candidate mappings for local ranking
unscored_maps = anchored_onto_maps.unscored_cand_maps() 

# feed the scored mappings back
# scored_maps <- some scoring method applied to unscored_maps
anchored_onto_maps.fill_scored_maps(scored_maps)

# saving and reloading are the same as OntoMappings

```

