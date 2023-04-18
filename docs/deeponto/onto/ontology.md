# Ontology

Python classes in this page are strongly dependent on the OWLAPI library. 
The base class [`Ontology`][deeponto.onto.Ontology] extends several features
including convenient access to specially defined entities (e.g., `owl:Thing` and `owl:Nothing`),
indexing of entities in the signature with their IRIs as keys, and some other customised functions
for specific ontology engineering purposes. [`Ontology`][deeponto.onto.Ontology] also has an 
[`OntologyReasoner`][deeponto.onto.OntologyReasoner] attribute which provides reasoning facilities
such as classifying entities, checking entailment, and so on. Users who are familiar with the OWLAPI
should feel relatively easy to extend the Python classes here.


::: deeponto.onto.ontology
    heading_level: 2
    options:
        members: ["Ontology"]
