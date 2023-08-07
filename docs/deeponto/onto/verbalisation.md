# Ontology Verbalisation

Verbalising an ontology into natural language texts is a challenging task. $\textsf{DeepOnto}$ provides some basic building blocks for achieving this goal. The implemented [`OntologyVerbaliser`][deeponto.onto.verbalisation.OntologyVerbaliser] is essentially a recursive concept verbaliser that first splits a complex concept $C$ into a sub-formula tree, verbalising the leaf nodes (atomic concepts or object properties) by their names, then merging the verbalised child nodes according to the logical pattern at their parent node. 

Please cite the following paper if you consider using our verbaliser.

!!! credit "Paper"

    The recursive concept verbaliser is proposed in the paper:
    [Language Model Analysis for Ontology Subsumption Inference (Findings of ACL 2023)](https://aclanthology.org/2023.findings-acl.213).

    ```
    @inproceedings{he-etal-2023-language,
        title = "Language Model Analysis for Ontology Subsumption Inference",
        author = "He, Yuan  and
        Chen, Jiaoyan  and
        Jimenez-Ruiz, Ernesto  and
        Dong, Hang  and
        Horrocks, Ian",
        booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
        month = jul,
        year = "2023",
        address = "Toronto, Canada",
        publisher = "Association for Computational Linguistics",
        url = "https://aclanthology.org/2023.findings-acl.213",
        doi = "10.18653/v1/2023.findings-acl.213",
        pages = "3439--3453"
    }
    ```


::: deeponto.onto.verbalisation
    heading_level: 2
