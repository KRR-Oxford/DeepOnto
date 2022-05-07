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

> Precision, Recall, and F1 are frequently used in evaluating `global_match` whereas ranking-based metrics like Hits@K and MRR are used in evaluating `pair_score`. See our incoming publication (to-be-updated) for detailed guidance of ontology matching evaluation.  


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

In this mode, the OM model is expected to compute the matching scores for input class pairs. Compared to Global Matching, an extra argument for unscored mappings is needed. The unscored mappings are implemented using `OntoMappings` data structre and saved in a folder containing two files: `.pkl` for the serialized object and `.json` for human readable format. Such mappings can be generated using the script: `./data_scripts/om_rank_cands.py`, which generates negative candidates for each reference mapping for *local ranking* evaluation. Users can also transform a `.tsv` file with three columns: "SrcEntity", "TgtEntity", and "Score" to an `OntoMappings` object using the following code:

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
--src_onto ./data/src_onto.owl \
--tgt_onto ./data/tgt_onto.owl \
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
