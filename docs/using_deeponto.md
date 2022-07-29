<!---
Copyright 2021 Yuan He (KRR-Oxford). All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

?>DeepOnto provides scripts for different purposes.
- `onto_match.py`: using implemented Ontology Alignment (a.k.a. Ontology Matching; OM) models.
- `om_eval.py`: evaluate the performance of OM models.
- `onto_prune.py`: pruning an ontology while preserving the hierarchy of remaining classes.
- `om_subs.py`: generating inter-ontology subsumption mappings from inter-ontology equivalence mappings.
- `om_cands.py`: generating (hard) negative candidate mappings for given positive reference mappings.

Please see the following sections for example usage.

!> Contents regarding `pair_score` mode of OM and `om_cands.py` are not fully ready.

## Ontology Matching

There are two modes for OM: `global_match` and `pair_score`. `global_match` aims to compute mappings given two input ontologies and `pair_score` is for scoring provided class pairs. For both modes, the source and target input ontologies are required; for `pair_score`, an input file containing the unscored class pairs and a flag that indicates the direction of the class pairs (`src2tgt` or `tgt2src`) are further required.

?> `Precision`, `Recall`, and `F-score` are frequently used in evaluating `global_match`; whereas `Accuracy` can be used in evaluating any class pair input, ranking-based metrics like `Hits@K` and `MRR` are used in evaluating `pair_score` when the input class pairs are grouped as *1 positive + N negatives*. See our [resource paper](https://arxiv.org/abs/2205.03447) for detailed guidance of ontology matching evaluation.  

?> A configuration file in `.json` needs to be provided for each OM model, see detailed configurations [here](om_models).


### Global Matching

In this mode, the OM model is expected to search for all plausible cross-ontology class pairs that are semantically related (through, e.g., equivalence, subsumption). To search and compute mappings globally, the OM model needs to address *(1)* how semantically close two classes are; *(2)* how to search efficiently (naive traversal takes quadratic time). Futher refinement such as extension and repair are also optional for postprocessing.

Parameters for `onto_match.py` in `global_match` mode:

- **saved_path**(*str*): the path to the main output directory.
- **src_onto**(*str*): the path to the source ontology file.
- **tgt_onto**(*str*): the path to the target ontology file.
- **config_path**(*str*): the path to the configuration file, default minimal configurations for each OM model is available at `./cofig`.

Example usage of `onto_match.py` for global matching:

**Step 1**: Run the script with above arguments specified.

```bash
# matching DOID and ORDO with minimal configurations
python onto_match.py \
--saved_path "./experiment/doid2ordo" \  
--src_onto "./data/doid.owl" \
--tgt_onto "./data/ordo.owl" \
--config_path "./config/bertmap.json"
```

**Step 2**: Choose `global_match` and an implemented OM model.

```bash
######################################################################
###                   Choose a Supported OM Mode                   ###
######################################################################

[0]: global_match
[1]: pair_score
Enter a number: 0

######################################################################
###                 Choose an Implemented OM Model                 ###
######################################################################

[0]: bertmap
[1]: string_match
[2]: edit_sim
Enter a number: 0
```

Then the script will do the followings:
- Load and parse the input ontologies;
- Train the scoring function with constructed data if the selected model is learning-based;
- Globally compute the mappings and apply any refinement steps if any.

?> Depending on the specific configurations of the selected model, the outputs could be different. Please refer to the documentation page of [OM models](om_models.md) for details.


### Pair Scoring

In this mode, the OM model is expected to compute the matching scores for input class pairs. Compared to Global Matching, extra arguments for unscored mappings and its flag (`src2tgt` or `tgt2src`) are needed. The easiest ...

The unscored mappings are implemented using `OntoMappings` data structre and saved in a folder containing two files: `.pkl` for the serialized object and `.json` for human readable format. Such mappings can be generated using the script: `./data_scripts/om_rank_cands.py`, which generates negative candidates for each reference mapping for *local ranking* evaluation. Users can also transform a `.tsv` file with three columns: "SrcEntity", "TgtEntity", and "Score" to an `OntoMappings` (see [Datastructures](https://krr-oxford.github.io/DeepOnto/#/data_structures?id=mapping)) object using the following code:

```python
# Fix import error if not download deeponto from PyPI
import os
import sys

main_dir = os.getcwd().split("DeepOnto")[0] + "DeepOnto/src"
sys.path.append(main_dir)

# load OntoMappings data structure
from deeponto.onto.mapping import OntoMappings

# note that for unscored mappings please set all values in "Score" column as 0.0
onto_maps = OntoMappings.read_tsv_mappings("path_to_tsv_file", flag="src2tgt")
onto_maps.save_instance("./unscored_maps")
```

Example usage of `onto_match.py` for pair scoring:

**Step 1**: Run the script with specified output directory, source and target ontologies, configuration file (if not provided, default minimal configurations are used), path to the unscored mappings (saved using `OntoMappings` data structure), and the flag of the unscored mappings (`src2tgt` (resp. `tgt2src`) if the input mappings are organized as `(src_class, tgt_class)` pairs (resp. `(tgt_class, src_class)` pairs)).

```bash
python onto_match.py \
--saved_path ./onto_match_experiment \  
--src_onto_path ./data/src_onto.owl \
--tgt_onto_path ./data/tgt_onto.owl \
--config_path ./config/bertmap.json \
--to_be_scored_maps_path ./unscored_maps \
--to_be_scored_flag src2tgt
```

**Step 2**: Choose `global_match` and any implemented OM model.

```bash
######################################################################
###                   Choose a Supported OM Mode                   ###
######################################################################

[0]: global_match
[1]: pair_score
Enter a number: 1

######################################################################
###                 Choose an Implemented OM Model                 ###
######################################################################

[0]: bertmap
[1]: string_match
[2]: edit_sim
Enter a number: 0
```

Then the script will do the followings:
- Load and parse the input ontologies;
- Train the scoring function with constructed data if the selected model is learning-based;
- Compute the scores for the input mappings and save the results in `./${exp_dir}/${model_name}/pair_score/${flag}` using `OntoMappings.save_instance()` (see explanation of `OntoMappings` [here](https://krr-oxford.github.io/DeepOnto/#/data_structures?id=mapping)).


### OM Evaluation

We provide an OM evaluation script for global matching (to compute `Precision`, `Recall`, and `F-score` on full alignment) and local ranking (to compute `Hits@K` and `MRR` on selected candidates).


Example usage of `om_eval.py` for global matching evaluation:

**Step 1**: For global matching evaluation, run the script with specified output directory, the path to prediction mappings (saved as `OntoMappings` consisting of a `.pkl` file for loading and a `.json` or `.tsv` file for reading), the path the reference mappings (saved as `.tsv` with columns "SrcEntity", "TgtEntity", and "Score"), the path to the null reference mappings (`.tsv`) which are reference mappings to be ignored in evaluation (e.g., training and validation mappings when evaluating on the testing mappings), the mapping threshold (leave it blank if the final outputs are determined), the choice of displaying more F-scores (e.g., $F_2$ and $F_{0.5}$).


```bash
python om_eval.py \
--saved_path ./om_results \  
--pred_path ./om_models/results/pred_maps \ 
--ref_path ./data/test_maps.tsv \
--null_ref_path ./data/train+val_maps.tsv \
--threshold 0.0 \  
--show_more_f_scores False
```

**Step 2**: Choose `global_match` evaluation mode.
```
######################################################################
###                    Choose a Evaluation Mode                    ###
######################################################################

[0]: global_match
[1]: local_rank
Enter a number: 0

######################################################################
###          Eval using P, R, F-score with threshold: 0.0          ###
######################################################################

{
    "P": 0.966,
    "R": 0.606,
    "f_score": 0.745
}
```

Example usage of `om_eval.py` for local ranking evaluation:


**Step 1**: For local ranking evaluation, run the script with specified output directory, the path to prediction mappings (saved as `OntoMappings` consisting of a `.pkl` file for loading and a `.json` file for reading), the path the anchored reference mappings (saved as `AnchoredOntoMappings` consisting of a `.pkl` file for loading and a `.json` file for reading), and the number of hits concerned. 

The prediction mappings consist of all scored entity pairs from an OM model implemented in DeepOnto whereas the anchored reference mappings consist of unscored candidate mappings associated with each reference pair (can be generated by `om_rank_cands.py`). Essentially, the prediction mappings are filled back to the anchored reference mappings before calculating the ranking-based metrics.

> See `OntoMappings` and `AnchoredOntoMappings` [here](https://krr-oxford.github.io/DeepOnto/#/data_structures?id=mapping) for details.


```bash
python om_eval.py \
--saved_path ./om_results \  
--pred_path ./om_models/results/scored_maps \ 
--ref_anchor_path ./data/scored_anchored_maps \
--hits_at 1 5 10 30 100
```


**Step 2**: Choose `local_rank` evaluation mode.
```
######################################################################
###                    Choose a Evaluation Mode                    ###
######################################################################

[0]: global_match
[1]: local_rank
Enter a number: 1

######################################################################
###                     Eval using Hits@K, MRR                     ###
######################################################################

523987/523987 of scored mappings are filled to corresponding anchors.
{
    "MRR": 0.919,
    "Hits@1": 0.876,
    "Hits@5": 0.976,
    "Hits@10": 0.99,
    "Hits@30": 0.997,
    "Hits@100": 1.0
}
```

--------------------------------
## Ontology Data Processing

This section introduces some scripts for processing ontology data and creating OM resources. The pruning, subsumption mapping generation, negative candidate mapping generation algorithms were proposed in our [resource paper](https://arxiv.org/abs/2205.03447).

### Ontology Pruning

The script `onto_prune.py` is used to prune an ontology by preserving classes according to a list of class IRIs while keeping the hierarchy. Specifically, if a class's IRI is not present in the input IRI list, then any axioms it involves will be deleted, and its parents will be asserted to be the parents of its children, so as to preserve the hierarchy. 

?> This can be seen as the most basic hierarchy preservation; in future, we will exploit logic modules to preserve more hierarchical information.

Parameters of `onto_prune.py`:

- **saved_path**(*str*): the path to the output directory.
- **src_onto_path**(*str*): the path to the source ontology that is about to be pruned.
- **preserved_iris_path**(*str*): the path to the file of class IRIs (one per line) that will be preserved.

Example usage of `onto_prune.py` for ontology pruning:

**Step 1**: Run the script with the above arguments specified.

```python
python onto_prune.py \
--saved_path ./pruned_onto \  
--src_onto_path ./data/src_onto.owl
--preserved_iris_path ./preserved_class_iris.txt

```

### Subsumption Mapping Generation

To construct inter-ontology subsumption mappings, we could utilize the **inter-ontology equivalence mappings** as the anchors. Specifically, we fix the source class of the equivalence mapping, and search for the ancestors (or descendants) of the target class, combining them with the source class to form the subsumption mappings. 

Parameters of `om_subs.py`: 

- **saved_path**(*str*): the path to the output directory.
- **src_onto**(*str*): the path to the source ontology file.
- **tgt_onto**(*str*): the path to the target ontology file.
- **equiv_maps_path**(*str*): the path to the equivalence mappings in `.tsv` with columns of `"SrcEntity"`, `"TgtEntity"`, and `"Score"`.
- **max_subs_ratio**(*int*): the maximum number of subsumption mappings generated for each input equivalence mapping, default is `1`.
- **is_delete_equiv_tgt**(*bool*): whether or not to delete the target class of the equivalence mapping used for generating any subsumption mappings. 
- **max_hop**(*int*): the maximum number of hops for searching the subsumption candidates, default is `1`, referring to the *most specific* subsumers.

Example usage of `om_subs.py` script for inter-ontology subsumption mapping construction:

**Step 1**: Run the script with arguments above specified.

```bash
python om_subs.py \
--saved_path ./subs_maps/ \  
--src_onto_path ./data/src_onto.owl \
--tgt_onto_path ./data/tgt_onto.owl \
--equiv_maps_path ./data/src2tgt_equiv_maps.tsv \ 
--max_subs_ratio 1 \ 
--is_delete_equiv_tgt True \
--max_hop 1
```

If `is_delete_equiv_tgt` is set to be `True`, then it means we are corrupting the equivalence mappings (that are *used* for generating any subsumption mappings) by deleting their target side classes to prevent an OM system from inferring subsumptions directly from the equivalence mappings. In this case, the deleted target classes may intervene the generation of some subsumption candidates because they may occasionally appear in a generated subsumption mapping. Two algorithms (`online` or `static`) were proposed (see details in our [resource paper](https://arxiv.org/abs/2205.03447)) to deal with the class deletion. 

If `is_delete_equiv_tgt` is set to be `False`, choosing either algorithm will generate the same set of subsumption candidates because no classes will be marked for deletion.

**Step 2**: Choose which algorithm to be used and which subsumption relation to be considered. 

```bash
######################################################################
###                    Choose a Generation Type                    ###
######################################################################

[0]: static
[1]: online
Enter a number: 0

######################################################################
###                 Choose a Subsumption Relation                  ###
######################################################################

[0]: '<' (subClassOf)
[1]: '>' (superClassOf)
Enter a number: 0
```

There are two ways (`static` or `online`) of doing subsumption mapping construction with class deletion considered. By choosing `static`, target classes present in the equivalence mappings will be marked for deletion first unless they have no ancestors (resp. descendants) for `"<"` (resp. for `">"`) that can be used for constructing subsumption mappings; then it starts generating subsumption mappings and exclude those with target classes marked deleted. By choosing `online`, the deletion and construction are operated instantly for each iteration over the traversal of equivalence mappings; target classes of the equivalence mappings will not be omitted from deletion if they have been included in the subsumption mappings generated in the previous iterations.

In principle, `online` will generate more subsumption mappings than `static` because less classes are marked for deletion. However, `online` is less stable than `static` because it can be affected by the order of the equivalence mappings during traversal.

**Step 3**: If `is_delete_equiv_tgt` is set to be `True`, the script will generate target class IRIs that should be preserved. Then, the user need to manually run the `onto_prune.py` script described [above](using_deeponto?id=ontology-pruning) to delete the corresponding classes.


### Negative Candidate Mapping Generation

To come up with a meaningful evaluation over the input class pairs in `pair_score` mode, instead of using all the possible negative candidates, selecting representative (hard) ones is essential for evaluation efficiency and a decent approximation of overall performance. Ranking over the selected candidates is referred to as **local ranking** and can be evaluated using [`om_eval.py`](using_deeponto?id=om-evaluation).

Specifically, for each reference mapping $(c, c')$, we can fix the source side and sample negative candidates from the target side ontology, which can then be combined with the $c$ to form negative candidate mappings. As such, each reference mapping can be seen as an `AnchorMapping` that has $N$ `EntityMapping` as candidates (see explanation of mapping data structures [here](data_structure?id=mapping)).

Parameters of `om_cands.py` for generating target negative candidates