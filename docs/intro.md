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


## Ontology Matching

There are two modes for Ontology Alignment (a.k.a. Ontology Matching; OM): `global_match` and `pair_score`. `global_match` aims to compute mappings given two input ontologies and `pair_score` is for scoring provided class pairs. 

?> Precision, Recall, and F1 are frequently used in evaluating `global_match` whereas ranking-based metrics like Hits@K and MRR are used in evaluating `pair_score`. See our incoming publication (to-be-updated) for detailed guidance of ontology matching evaluation.  


### Global Matching

In this mode, the OM model is expected to search for all plausible cross-ontology class pairs that are semantically related (through, e.g., equivalence, subsumption). To search and compute mappings globally, the OM model needs to address *(1)* how semantically close two classes are; *(2)* how to search efficiently (naive traversal takes quadratic time). Futher refinement such as extension and repair are also popular for postprocessing.

Example usage of `onto_match.py` for global matching:

**Step 1**: Run the script with specified output directory, source and target ontologies, and configuration file (if not provided, default minimal configurations are used).

```bash
python onto_match.py \
--saved_path "./onto_match_experiment" \  
--src_onto "./data/src_onto.owl" \
--tgt_onto "./data/tgt_onto.owl" \
--config_path "./config/bertmap.json"
```

**Step 2**: Choose `global_match` and any implemented OM model.

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


### Pair Scoring

In this mode, the OM model is expected to compute the matching scores for input class pairs. Compared to Global Matching, an extra argument for unscored mappings is needed. The unscored mappings are implemented using `OntoMappings` data structre and saved in a folder containing two files: `.pkl` for the serialized object and `.json` for human readable format. Such mappings can be generated using the script: `./data_scripts/om_rank_cands.py`, which generates negative candidates for each reference mapping for *local ranking* evaluation. Users can also transform a `.tsv` file with three columns: "SrcEntity", "TgtEntity", and "Score" to an `OntoMappings` (see [Datastructures](https://krr-oxford.github.io/DeepOnto/#/data_structures?id=mapping)) object using the following code:

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
- Compute the scores for the input mappings and save the results in `./${exp_dir}/${model_name}/pair_score/${flag}` using `OntoMappings.save_instance()`. 

--------------------------------
## Ontology Pruning

The script `onto_prune.py` is used to prune an ontology by preserving classes according to a list of class IRIs while keeping the hierarchy. Specifically, if a class's IRI is not present in the input IRI list, then any axioms it involves will be deleted, and its parents will be asserted to be the parents of its children, so as to preserve the hierarchy. 

> This can be seen as the most basic hierarchy preservation; in future, we will exploit logic modules to preserve more hierarchical information.

Example usage of `onto_prune.py` for ontology pruning:

**Step 1**: Run the script with the specified output directory, the path to the source ontology to be pruned, and the file of preserved class IRIs where each line states a class IRI.

```python
python onto_prune.py \
--saved_path ./pruned_onto \  
--src_onto_path ./data/src_onto.owl
--preserved_iris_path ./preserved_class_iris.txt

```
--------------------------------
## Subsumption Mappings

To construct inter-ontology subsumption mappings, we could utilize the inter-ontology equivalence mappings as the anchors. Specifically, we fix the source class of the equivalence mapping, and search for the ancestors (or descendants) of the target class, combining them with the source class to form the subsumption mappings. 

Example usage of `om_subs.py` script for inter-ontology subsumption mapping construction:

**Step 1**: Run the script with the specified output directory, the paths to the source and target ontologies, the path to the equivalence mappings between source and target ontologies (a `.tsv` file with columns "SrcEntity", "TgtEntity", and "Score"), the subsumption relation (`"<"` or `">"` for IS-A and the inverse of IS-A), the maximum number of subsumption mappings generated for each equivalence mappping, the decision of deleting the target side classes of the *used* equivalence mapppings, the maximum number of hops for hierarchy search.

```bash
python om_subs.py \
--saved_path ./subs \  
--src_onto_path ./data/src_onto.owl \
--tgt_onto_path ./data/tgt_onto.owl \
--equiv_maps_path ./data/equiv_maps.tsv \ 
--subs_relation "<" \ 
--max_subs_ratio 1 \ 
--is_delete_equiv_tgt True \
--max_hop 1
```

If `is_delete_equiv_tgt` is set to be `True`, then it means we are corrupting the equivalence mappings (that are *used* for generating any subsumption mappings) by deleting their target side classes to prevent an OM system from inferring subsumptions directly from the equivalence mappings. 

There are two ways (`static` or `online`) of doing subsumption mapping construction with class deletion considered. By choosing `static`, target classes present in the equivalence mappings will be marked for deletion first unless they have no ancestors (resp. descendants) for `"<"` (resp. for `">"`) that can be used for constructing subsumption mappings; then it starts generating subsumption mappings and exclude those with target classes marked deleted. By choosing `online`, the deletion and construction are operated instantly for each iteration over the traversal of equivalence mappings; target classes of the equivalence mappings will not be omitted from deletion if they have been included in the subsumption mappings generated in the previous iterations.

In principle, `online` will generate more subsumption mappings than `static` because less classes are marked for deletion. However, `online` is less stable than `static` because it can be affected by the order of the equivalence mappings during traversal.

> If `is_delete_equiv_tgt` is set to be `False`, choosing `static` or `online` will make no difference because no classes will be marked for deletion.

**Step 2**: Choose which algorithm to be used.

```bash
######################################################################
###                    Choose a Generation Type                    ###
######################################################################

[0]: static
[1]: online
Enter a number: 0
```
**Step 3**: If `is_delete_equiv_tgt` is set to be `True`, use the `onto_prune.py` script described [above](https://krr-oxford.github.io/DeepOnto/#/intro?id=ontology-pruning) to delete the corresponding classes.


