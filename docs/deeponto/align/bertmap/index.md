## Overview

!!! credit "Paper"

    $\textsf{BERTMap}$ is proposed in the paper:
    *[BERTMap: A BERT-based Ontology Alignment System (AAAI-2022)](https://ojs.aaai.org/index.php/AAAI/article/view/20510)*.

    
The ontology matching (OM) pipeline of $\textsf{BERTMap}$ consists of following steps:

1. Load the source and target ontologies and build **annotation indices** from them based on selected annotation properties.
2. Construct the **text semantics corpora** including **intra-ontology**, **cross-ontology** (optional), and **auxiliary** (optional) sub-corpora.
3. Split samples in the form of `(src_annotation, tgt_annotation, synonym_label)` into training and validation splits.
4. Fine-tune a **BERT synonym classifier** on the samples and obtain the best checkpoint on the validation split.
5. Predict mappings for each class $c$ of the source ontology $\mathcal{O}$ by:
    
    - Selecting plausible candidates $c'$s in the target ontology $\mathcal{O'}$ based on **idf scores** w.r.t. the **sub-word inverted index** built from `tgt_annotation_index`.
    For $c$ and a candidate $c'$. first check if they can be string-matched (i.e., share a common annotation); if not,
    consider all combinations (cartesian product) of their respective class annotations, compute a synonym score for each combination, and take the **average of synonym scores as the mapping score**.
    - $N$ best scored mappings (no filtering) will be preserved as raw predictions which should have relatively higher recall and lower precision.

6. Extend the raw predictions using an **iterative** algorithm based on the **locality principle**. To be specific, if $c$ and $c'$ are matched with a **relatively high mapping score** ($\geq \kappa$), then search for plausible mappings between the *parents* (resp. *children*) of $c$ and the *parents* (resp. *children*) of $c'$. This process is iterative because there would be new
highly scored mappings at each round. Terminate mapping extension when there is no
new mapping with score $\geq \kappa$ found or exceed the maximum number of iterations. Note that $\kappa$ is set to $0.9$ by default same as in the original paper.

7. Truncate the extended mappings by preserving only those with scores $\geq \lambda$. In the original paper, $\lambda$ is supposed to be tuned on validation mappings – which are often not available. Also, $\lambda$ is not a sensitive hyperparameter in practice. Therefore, we manually set $\lambda$ to $0.9995$ as a default value which usually yields a higher F1 score. Note that both $\kappa$ and $\lambda$ are made available in the configuration file.

8. Repair the rest of the mappings with the repair module built in LogMap (BERTMap does not focus on mapping repair). In short, a minimum set of inconsistent mappings will be removed (further improve precision).

$\textsf{BERTMap}$ with only the string match module and the candidate selection process is referred to as $\textsf{BERTMapLt}$, the light version without BERT training and mapping refinement.

In addition to the traditional OM procedure, the scoring module of $\textsf{BERTMap}$ and $\textsf{BERTMapLt}$ can be used to evaluate any class pair given their annotations. This is useful in ranking-based evaluation. 

::: deeponto.align.bertmap.pipeline
    handler: python
    heading_level: 2


::: deeponto.align.bertmap.text_semantics
    handler: python
    heading_level: 2


::: deeponto.align.bertmap.bert_classifier
    handler: python
    heading_level: 2

::: deeponto.align.bertmap.mapping_prediction
    handler: python
    heading_level: 2

::: deeponto.align.bertmap.mapping_refinement
    handler: python
    heading_level: 2
