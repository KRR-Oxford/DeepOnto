<!-- !!! credit "Paper"

    $\textsf{BERTMap}$ is proposed in the paper:
    [BERTMap: A BERT-based Ontology Alignment System (AAAI-2022)](https://ojs.aaai.org/index.php/AAAI/article/view/20510).

    ```
    @inproceedings{he2022bertmap,
        title={BERTMap: a BERT-based ontology alignment system},
        author={He, Yuan and Chen, Jiaoyan and Antonyrajah, Denvar and Horrocks, Ian},
        booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
        volume={36},
        number={5},
        pages={5684--5691},
        year={2022}
    }
    ```


$\textsf{BERTMap}$ is a BERT-based ontology matching (OM) system consisting of following components:

- **Text semantics corpora** construction from input ontologies, and optionally from input mappings and other auxiliary ontologies.
- **BERT synonym classifier** training on synonym and non-synonym samples in text semantics corpora.
- **Sub-word Inverted Index** construction from the tokenised class annotations for candidate selection in mapping prediction.
- **Mapping Predictor** which integrates a simple edit distance-based string matching module and the fine-tuned BERT synonym classifier for mapping scoring. For each source ontology class, narrow down
target class candidates using the sub-word inverted index, apply string matching for "easy" mappings
and then apply BERT matching.
- **Mapping Refiner** which consists of the mapping extension and mapping repair modules. Mapping extension is an iterative process based on the *locality principle*. Mapping repair utilises the LogMap's debugger. 

$\textsf{BERTMapLt}$ is a light-weight version of $\textsf{BERTMap}$ *without* the BERT module and mapping refiner.

See the tutorial for $\textsf{BERTMap}$ [here](../../../bertmap). -->
 

::: deeponto.subs.bertsubs.pipeline_inter
    heading_level: 2

