# Using Ontology

$\textsf{DeepOnto}$ extends from the OWLAPI and implements many useful methods for ontology processing and reasoning, integrated in the base class
[`Ontology`][deeponto.onto.Ontology].

This page gives typical examples of how to use [`Ontology`][deeponto.onto.Ontology]. There are other more specific usages, please refer to the documentation by clicking [`Ontology`][deeponto.onto.Ontology].

## Loading Ontology

[`Ontology`][deeponto.onto.Ontology] can be easily loaded from a local ontology file by its path:

```python
from deeponto.onto import Ontology
onto = Ontology("path_to_ontology.owl")
```
## Acessing Ontology Entities

The most fundamental feature of [`Ontology`][deeponto.onto.Ontology] is to access entities in the ontology such as **classes** (or *concepts*) and **properties** (*object*, *data*, and *annotation* properties). To get an entity by its IRI, use the following:

```python
from deeponto.onto import Ontology
# e.g., load the disease ontology
doid = Ontology("doid.owl")
# class or property IRI as input
doid.get_owl_object_from_iri("http://purl.obolibrary.org/obo/DOID_9969")
```

To obtain the literal values (as `Set[str]`) of an annotation property (such as *rdfs:label*) for an entity:

```python
# note that annotations with no language tags are deemed as in English ("en")
>>> doid.get_owl_object_annotations(
... doid.get_owl_object_from_iri("http://purl.obolibrary.org/obo/DOID_9969"),
... annotation_property_iri='http://www.w3.org/2000/01/rdf-schema#label',
... annotation_language_tag=None,
... apply_lowercasing=False
... )
{'carotenemia'}
```

To get the special entities related to $\top$ and $\bot$:

```python
>>> doid.OWLThing
<java object 'uk.ac.manchester.cs.owl.owlapi.OWLClassImpl'>
>>> doid.OWLBottomDataProperty
<java object 'uk.ac.manchester.cs.owl.owlapi.OWLDataPropertyImpl'>
```

## Ontology Reasoning

[`Ontology`][deeponto.onto.Ontology] has an important attribute `.reasoner` which refers to the [HermitT](http://www.hermit-reasoner.com/) reasoning tool.

To get the **super-classes** (or **super-properties**) of an entity, use the following:

```python
>>> doid_class = doid.get_owl_object_from_iri("http://purl.obolibrary.org/obo/DOID_9969")
>>> doid.reasoner.super_entities_of(doid_class, direct=False) 
['http://purl.obolibrary.org/obo/DOID_0014667',
 'http://purl.obolibrary.org/obo/DOID_0060158',
 'http://purl.obolibrary.org/obo/DOID_4']
```

The outputs are IRIs of the corresponding super-entities. `direct` is a boolean value indicating whether the returned entities
are **parents** (`direct` is `True`) or **ancestors** (`direct` is `False`).

To get the **sub-entities** (**children** or **descendants**), simply replace the method name with `.sub_entities_of`.

To retrieve the entailed **instances** of a class:

```python
doid.reasoner.instances_of(doid_class)
```

The implemented reasoner also supports several **logical checks** for subsumption, disjointness, and so on. For example:

```python
doid.reasoner.check_subsumption(doid_potential_sub_entity, doid_potential_super_entity)
```

## Ontology Pruning

The pruning function aims to remove unwanted ontology classes while preserving the relevant hierarchy. Specifically, for each class $c$ to be removed, subsumption axioms will be created between the parents of $c$ and the children of $c'$. Then, an `OWLEntityRemover` will be used to apply the pruning.

!!! credit "paper"

    The ontology pruning algorithm is introduced in the paper: 
    [*Machine Learning-Friendly Biomedical Datasets for Equivalence and Subsumption Ontology Matching (ISWC 2022)*](https://link.springer.com/chapter/10.1007/978-3-031-19433-7_33).

```python
from deeponto.onto import Ontology

doid = Ontology("doid.owl")
to_be_removed_class_iris = [
    "http://purl.obolibrary.org/obo/DOID_0060158",
    "http://purl.obolibrary.org/obo/DOID_9969"
]
doid.apply_pruning(to_be_removed_class_iris)
doid.save_onto("doid.pruned.owl")
```

## Ontology Verbalisation

To be added next...
