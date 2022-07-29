?> This page describes important data structures implemented in DeepOnto for processing ontology data.

## Ontology

>  [:link:](https://github.com/KRR-Oxford/DeepOnto/blob/main/src/deeponto/onto/ontology.py)
*CLASS* &ensp; deeponto.onto.Ontology(<em>owl_path: str</em>)


The `Ontology` class relies on [OwlReady2](https://owlready2.readthedocs.io/) for loading OWL object, and then applies extra text processing on labels of selected *annotation properties* and optionally construct an *inverted index* from these class labels. The following method is used for creating a new ontology object.

> *METHOD* &ensp; Ontology.from_new(<em>onto_path: str, lab_props: List[str], tokenizer: Optional[Tokenizer]</em>)

Parameters for constructing a new `Ontology` object:
- **onto_path**(<em>str</em>): the path of an ontology file (preferrably in **.owl** format).
- **lab_props**(<em>List[str]</em>): a list of selected annotation properties (of class labels, aliases, synonyms, etc.) for text processing, default is `["http://www.w3.org/2000/01/rdf-schema#label"]`.
- **tokenizer**(<em>Optional[Tokenizer]</em>): an instance of the `Tokenizer` class used for tokenizing the class labels, default is `None` if not to construct an inverted index.


Example usage for creating, save, and reload an ontology:
```python
from deeponto.onto import Ontology
from deeponto.onto.text import Tokenizer

# use the sub-word tokenizer pre-trained with BERT
tokz = Tokenizer.from_pretrained("bert-base-uncased") 

onto = Ontology.from_new(
  onto_path="./onto_files/pizza.owl", 
  lab_props=["http://www.w3.org/2000/01/rdf-schema#label"],  # rdfs:label
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

*[Usage]* Use the inverted index for candidate selection:

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

# get the preprocessed labels for a class in this ontology
labs = onto.iri2labs["some-entity-iri"]

# tokens from the class labels
toks = tokz.tokenize_all(labs)

# retrieve ontology classes with idf scores
candidates = onto.idf_select(toks)
```

## Mapping

Several python classes for entity (in string) mappings are defined here as main data structrues used for OM system outputs and evaluation.

### EntityMapping

>  [:link:](https://github.com/KRR-Oxford/DeepOnto/blob/main/src/deeponto/onto/mapping.py)
*CLASS* &ensp; deeponto.onto.mapping.EntityMapping(<em>src_ent_iri: str, tgt_ent_iri: str, rel: str, score: float</em>)

The basic data structure for representing a mapping between entities.

Parameters for constructing a new `EntityMapping` object:
- **src_ent_iri**(<em>str</em>): the source entity IRI in string.
- **tgt_ent_iri**(<em>str</em>): the target entity IRI in string.
- **rel**(<em>str</em>): the semantic relation between the source and target entities.
- **score**(<em>float</em>): the score of the mapping ranging from $0.0$ to $1.0$.

Example usage for creating an equivalence entity mapping for `snomed:55940004` and `fma:54970`:

```python
from deeponto.onto.mapping import EntityMapping

src_ent_iri = "http://snomed.info/id/55940004"
tgt_ent_iri = "http://purl.org/sig/ont/fma/fma54970"

src2tgt_map = EntityMapping(
  src_ent_iri=src_ent_iri, 
  tgt_ent_iri=tgt_ent_iri, 
  rel="=",  # semantic relation symbol
  score=1.0 # score ranges from [0.0, 1.0]; 
            # 1.0 is usually set for a reference mapping
)
```

### AnchorMapping

>  [:link:](https://github.com/KRR-Oxford/DeepOnto/blob/main/src/deeponto/onto/mapping.py)
*CLASS* &ensp; deeponto.onto.mapping.AnchorMapping(<em>src_ent_iri: str, tgt_ent_iri: str, rel: str, score: float</em>)

It extends from the basic `EntityMapping` and can incorporate other entity mappings as its candidates.

Parameters for constructing a new `AnchorMapping` object:
- **src_ent_iri**(<em>str</em>): the source entity IRI in string.
- **tgt_ent_iri**(<em>str</em>): the target entity IRI in string.
- **rel**(<em>str</em>): the semantic relation between the source and target entities.
- **score**(<em>float</em>): the score of the mapping ranging from $0.0$ to $1.0$.

Example usage for constructing a new `AnchorMapping` and adding a `EntityMapping` as a candidate.

```python
from deeponto.onto.mapping import AnchorMapping

src_ent_iri = "http://snomed.info/id/55940004"
tgt_ent_iri = "http://purl.org/sig/ont/fma/fma54970"

anchor_map = AnchorMapping(
  src_ent_iri=src_ent_iri, 
  tgt_ent_iri=tgt_ent_iri, 
  rel="=", 
  score=1.0 # score ranges from [0.0, 1.0]; 
            # 1.0 is usually set for a reference mapping
)

cand_map = EntityMapping(...)
anchor_map.add_candidate(cand_map)
```

### OntoMappings

>  [:link:](https://github.com/KRR-Oxford/DeepOnto/blob/main/src/deeponto/onto/mapping.py)
*CLASS* &ensp; deeponto.onto.mapping.OntoMappings(<em>
        flag: str, 
        n_best: Optional[int],
        rel: str,
        dup_strategy: str,
        *ent_mappings: EntityMapping</em>)

The dict-based data structure which stores entity mappings of the same direction (`src2tgt` or `tgt2src`); the `map_dict` attribute stores entity mappings in form of a nested dictionary: 

```python
# {src_ent: {tgt_ent: score}}
{
  "http://snomed.info/id/55940004http://snomed.info/id/55940004": 
  {
    "http://purl.org/sig/ont/fma/fma54970": 1.0, 
    "http://purl.org/sig/ont/fma/fma34913": 0.563
  }
}
```

Parameters for constructing a new `OntoMappings` object:
- **flag**(<em>str</em>): the direction of the mapping (`src2tgt`, `tgt2src`).
- **n_best**(<em>int</em>): the number of top ranked mappings stored for each head entity, default is `None` which means there is no limit.
- **rel**(<em>str</em>): the semantic relation between the source and target entities.
- **dup_strategy**(<em>str</em>): the strategy used to dealing with new mappings that have duplicated source and target entities (`average`, `kept_old`, `kept_new`).
- **\*ent_mappings**(<em>EntityMapping</em>): sequence of `EntityMapping` to be added.

Entity mappings that belong to the same `src_ent` are sorted by scores in descending order, if `n_best` is set, only the top `n_best` mappings w.r.t. each `src_ent` will be kept.

Example usage for creating an `OntoMappings` object and feed mappings:

```python
from deeponto.onto.mapping import OntoMappings

onto_maps = OntoMappings(flag="src2tgt", n_best=10, rel="=")

# add a single mapping
onto_maps.add(src2tgt_map) # EntityMapping

# add many mappings
onto_maps.add_many(*list_of_src2tgt_maps) # List[EntityMapping]

# ranked mappings in [dict] with source entities as keys
onto_maps.map_dict

# return all mappings as tuples
onto_maps.to_tuples()

# return the top k ranked mappings for a source entity 
# that exceeds certain mapping score threshold
onto_maps.topks_for_ent(src_ent_iri, K=3, threshold: float = 0.5)

# save the mappings
# a .pkl file is saved for the object and a .json/.tsv file is saved for readability
onto_maps.save_instance("./src2tgt_maps")

# reload the mappings
onto_maps = OntoMappings.from_saved("./src2tgt_maps")

```

### AnchoredOntoMappings

>  [:link:](https://github.com/KRR-Oxford/DeepOnto/blob/main/src/deeponto/onto/mapping.py)
*CLASS* &ensp; deeponto.onto.mapping.AnchoredOntoMappings(<em>
        flag: str, 
        n_best: Optional[int],
        rel: str,
        dup_strategy: str,
        *anchor_mappings: AnchorMapping</em>)

It is a similar data structure as `OntoMappings` except that its keys are entity pairs (anchors) instead of a single source entity. It is essentially a collection of candidate mappings w.r.t. each 
anchor mapping (in `AnchorMapping`). For example, to generate negative candidates for each reference mapping (as an anchor) for [local ranking evaluation](using_deeponto?id=om-evaluation), we need to use the reference class pair as a double-key and add candidate mappings as values.

Parameters for constructing a new `AnchoredOntoMappings` object:
- **flag**(<em>str</em>): the direction of the mapping (`src2tgt`, `tgt2src`).
- **n_best**(<em>int</em>): the number of top ranked mappings stored for each anchor mapping, default is `None` which means there is no limit.
- **rel**(<em>str</em>): the semantic relation between the source and target entities.
- **dup_strategy**(<em>str</em>): the strategy used to dealing with new mappings that have duplicated source and target entities (`average`, `kept_old`, `kept_new`).
- **\*anchor_mappings**(<em>EntityMapping</em>): sequence of `AnchorMapping` to be added.

Example usage for creating an `AnchoredOntoMappings` object and feed mappings:
```python
from deeponto.onto.mapping import AnchoredOntoMappings

anchored_onto_maps = AnchoredOntoMappings(flag="src2tgt", n_best=10, rel="=")

# add an (reference) anchor mapping that has an unscored candidate mapping
anchor_map = AnchorMapping(ref_src_ent_iri, ref_tgt_ent_iri, "=", 1.0)
anchor_map.add_candidate(EntityMapping(ref_src_ent_iri, cand_tgt_ent_iri, "=", 0.0))
anchored_onto_maps.add(anchor_map)  

# flatten AnchoredOntoMappings to OntoMappings with all zero scores for
# conducting pair scoring prediction without duplicates
unscored_maps = anchored_onto_maps.unscored_cand_maps() 

# feed the scored OntoMappings back
# scored_maps <- some scoring method applied to unscored_maps
anchored_onto_maps.fill_scored_maps(scored_maps)

# saving and reloading are the same as OntoMappings
```
