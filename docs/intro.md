## Ontology Alignment (or Matching)

There are two modes for Ontology Alignment (or Matching; OM): `global_match` and `pair_score`. `global_match` aims to compute mappings given two input ontologies and `pair_score` is for scoring provided class pairs. 

> Precision, Recall, and F1 are frequently used in evaluating `global_match` whereas ranking-based metrics like Hits@K and MRR are used in evaluating `pair_score`.


### Global Matching

In this mode, the OM model is expected to search for all plausible class pairs $$(c \in C, c' \in C')$$ where $$C$$ and $$C'$$ are sets of classes of the source ontology $$O$$ and the target ontology $$O'$$, respectively. To search and compute mappings globally, the OM model needs to address (1) how semantically close two classes are; (2) how to search efficiently (naive traversal takes $$O(n^2)$$). Futher refinement such as extension and repair are also popular for postprocessing.

Example usage of `onto_match.py`:

**Step 1**: Run the script with specified output directory, source and target ontologies, and configuration file (if not provided, default minimal configurations are used).

```
python onto_match.py \
--saved_path "./om_experiments" \  
--src_onto "./data/fma.owl" \
--tgt_onto "./data/snomed.owl" \
--config_path "./config/bertmap.json"
```

### Pair Scoring
