# Basic Usage of Ontology

$\textsf{DeepOnto}$ extends from the OWLAPI and implements many useful methods for ontology processing and reasoning, integrated in the base class
[`Ontology`][deeponto.onto.Ontology].

This page gives typical examples of how to use [`Ontology`][deeponto.onto.Ontology]. There are other more specific usages, please refer to the documentation by clicking [`Ontology`][deeponto.onto.Ontology].

## Loading Ontology

[`Ontology`][deeponto.onto.Ontology] can be easily loaded from a local ontology file by its path:

```python
from deeponto.onto import Ontology
```

Importing `Ontology` will require JVM memory allocation (defaults to `8g`):

```python
Please enter the maximum memory located to JVM: [8g]: 16g

16g maximum memory allocated to JVM.
JVM started successfully.
```

Loading an ontology from a local file:

```
onto = Ontology("path_to_ontology.owl")
```
## Acessing Ontology Entities

The most fundamental feature of [`Ontology`][deeponto.onto.Ontology] is to access entities in the ontology such as **classes** (or *concepts*) and **properties** (*object*, *data*, and *annotation* properties). To get an entity by its IRI, do the following:

```python
from deeponto.onto import Ontology
# e.g., load the disease ontology
doid = Ontology("doid.owl")
# class or property IRI as input
doid.get_owl_object_from_iri("http://purl.obolibrary.org/obo/DOID_9969")
```

To get the asserted parents or children of a given class or property, do the following:

```python
doid.get_asserted_parents(doid.get_owl_object_from_iri("http://purl.obolibrary.org/obo/DOID_9969"))
doid.get_asserted_children(doid.get_owl_object_from_iri("http://purl.obolibrary.org/obo/DOID_9969"))
```

To obtain the literal values (as `#!python Set[str]`) of an annotation property (such as $\texttt{rdfs:label}$) for an entity:

```python
# note that annotations with no language tags are deemed as in English ("en")
doid.get_owl_object_annotations(
    doid.get_owl_object_from_iri("http://purl.obolibrary.org/obo/DOID_9969"),
    annotation_property_iri='http://www.w3.org/2000/01/rdf-schema#label',
    annotation_language_tag=None,
    apply_lowercasing=False,
    normalise_identifiers=False
)
```

`#!console Output:`
:   &#32;
    ```python
    {'carotenemia'}
    ```

To get the **special entities** related to top ($\top$) and bottom ($\bot$), for example, to get $\texttt{owl:Thing}$:

```python
doid.OWLThing
```

## Ontology Reasoning

[`Ontology`][deeponto.onto.Ontology] has an important attribute `.reasoner` which refers to the [HermitT](http://www.hermit-reasoner.com/) reasoning tool.

### Inferring Super- and Sub-Entities

To get the **super-entities** (a super-class, or a super-propety) of an entity, do the following:

```python
doid_class = doid.get_owl_object_from_iri("http://purl.obolibrary.org/obo/DOID_9969")
doid.reasoner.get_inferred_super_entities(doid_class, direct=False) 
```

`#!console Output:`
:   &#32;
    ```python
    ['http://purl.obolibrary.org/obo/DOID_0014667',
    'http://purl.obolibrary.org/obo/DOID_0060158',
    'http://purl.obolibrary.org/obo/DOID_4']
    ```

The outputs are IRIs of the corresponding super-entities. `direct` is a boolean value indicating whether the returned entities are **parents** (`#!python direct=True`) or **ancestors** (`#!python direct=False`).

To get the **sub-entities**, simply replace the method name with `#!python sub_entities_of`.

### Inferring Class Instances

To retrieve the entailed **instances** of a class:

```python
doid.reasoner.instances_of(doid_class)
```

### Checking Entailment

The implemented reasoner also supports several **entailment checks** for subsumption, disjointness, and so on. For example:

```python
doid.reasoner.check_subsumption(doid_potential_sub_entity, doid_potential_super_entity)
```

----------------------------------------------------------------
## Feature Requests

Should you have any feature requests (such as those commonly used in the OWLAPI), please raise a ticket in the $\textsf{DeepOnto}$ GitHub repository.