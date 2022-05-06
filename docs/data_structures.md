This page describes important data structures implemented in DeepOnto for processing ontology data.

## Ontology

> Source file: `deeponto/onto/ontology.py`

The `Ontology` class is relies on `owlready2` for loading OWL object, and then applies extra text-level processing on labels of selected annotation properties and optionally construct an inverted index from these class labels. 
