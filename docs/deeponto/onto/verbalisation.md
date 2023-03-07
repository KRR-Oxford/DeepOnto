Verbalising an ontology into natural language texts is a challenging task. $\textsf{DeepOnto}$ provides some basic building blocks for achieving this goal. The implemented [`OntologyVerbaliser`][deeponto.onto.verbalisation.OntologyVerbaliser] is essentially a recursive concept verbaliser that first splits a complex concept $C$ into a sub-formula tree, verbalising the leaf nodes (atomic concepts or object properties) by their names, then merging the verbalised child nodes according to the logical pattern at their parent node. 

Please cite the following paper if you consider using our verbaliser.

!!! credit "Paper"

    [Language Model Analysis for Ontology Subsumption Inference](https://arxiv.org/abs/2302.06761).


::: deeponto.onto.verbalisation
    heading_level: 2
