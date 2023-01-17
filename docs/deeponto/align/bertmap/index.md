## Overview

!!! credit "Paper"

    BERTMap is proposed in the paper:
    *[BERTMap: A BERT-based Ontology Alignment System (AAAI-2022)](https://ojs.aaai.org/index.php/AAAI/article/view/20510)*.

    
The pipeline of BERTMap consists of following steps:

1. Load the source and target ontologies and build **annotation indices** from them based on selected annotation properties.
2. Construct the **text semantics corpora** including **intra-ontology**, **cross-ontology** (optional), and **auxiliary** (optional) sub-corpora.
3. Split samples in the form of `(src_annotation, tgt_annotation, synonym_label)` into training and validation splits.
4. Fine-tune a **BERT synonym classifier** on the samples and obtain the best checkpoint on the validation split.
5. Predict mappings for each class $c$ of the source ontology $\mathcal{O}$ by:
    
    - Selecting plausible candidates $c'$s in the target ontology $\mathcal{O'}$ based on **idf scores** w.r.t. the **sub-word inverted index** built from `src_annotation_index`.
    - For $c$ and a candidate $c'$, consider all combinations (cartesian product) of their respective class annotations, compute a synonym score
    for each combination, and take the **average of synonym scores as the mapping score**.
    - Candidates $c'$s involved in the top scored mappings will be preserved.

::: deeponto.align.bertmap.pipeline
    handler: python
    heading_level: 2


::: deeponto.align.bertmap.text_semantics
    handler: python
    heading_level: 2


::: deeponto.align.bertmap.bert_classifier
    handler: python
    heading_level: 2

