# Ontology Taxonomy

Extracting the taxonomy from an ontology often comes in handy for graph-based machine learning techniques. Here we provide a basic [`Taxonomy`][deeponto.onto.taxonomy.Taxonomy] class built upon `networkx.DiGraph` where nodes represent entities and edges represent subsumptions. We then provide the [`OntologyTaxonomy`][deeponto.onto.taxonomy.OntologyTaxonomy] class that extends the basic [`Taxonomy`][deeponto.onto.taxonomy.Taxonomy]. It utilises the simple [structural reasoner](https://owlcs.github.io/owlapi/apidocs_4/org/semanticweb/owlapi/reasoner/structural/StructuralReasoner.html) to enrich the ontology subsumptions beyond asserted ones, and build the taxonomy over the expanded subsumptions. Each node represents a named class and has a label (`rdfs:label`) attribute. The root node `owl:Thing` is also specified for functions like counting the node depths, etc. Moreover, we provide the [`WordnetTaxonomy`][deeponto.onto.taxonomy.WordnetTaxonomy] class that wraps the WordNet knowledge graph for easier access.

!!! note

        It is also possible to use [`OntologyProjector`][deeponto.onto.projection.OntologyProjector] to extract triples from the ontology as edges of the taxonomy. We will consider this feature in the future.

::: deeponto.onto.taxonomy
    heading_level: 2
