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

Besides model development and implementation, DeepOnto also aims to contribute new Ontology Matching (OM) resources and comprehensive evaluation workaround. In this page, we provide download links to our datasets and instructions for data usage.

## Bio-ML

The `Bio-ML` dataset provides five ontology pairs for both equivalence and subsumption ontology matching.

> Latest Update: Version 3, 28 July 2022

> Download link: https://doi.org/10.5281/zenodo.6917501 (CC BY 4.0 International).

> See detailed resource construction and evaluation methods in our paper: https://arxiv.org/abs/2205.03447.

> OAEI track: https://www.cs.ox.ac.uk/isg/projects/ConCur/oaei/ 

Statistics for the equivalence matching set-ups. In the **Category** column, "Disease" indicates that the Mondo data are mainly about disease concepts, while "Body", "Pharm", and "Neoplas" denote semantic types of "Body Part, Organ, or Organ Components", "Pharmacologic Substance", and "Neoplastic Process" in UMLS, respectively.

| Source | Task        | Category | #Classes      | #RefMaps (equiv) | #Annot.  | AvgDepths |
|--------|-------------|----------|---------------|------------------|----------|-----------|
| Mondo  | OMIM-ORDO   | Disease  | 9,642-8838    | 3,721            | 34K-34K  | 1.44-1.63 |
| Mondo  | NCIT-DOID   | Disease  | 6,835-8,848   | 4,684            | 80K-38K  | 2.04-6.85 |
| UMLS   | SNOMED-FMA  | Body     | 24,182-64,726 | 7,256            | 39K-711K | 1.86-9.32 |
| UMLS   | SNOMED-NCIT | Pharm    | 16,045-15,250 | 5,803            | 19K-220K | 1.09-3.26 |
| UMLS   | SNOMED-NCIT | Neoplas  | 11,271-13,956 | 3,804            | 23K-182K | 1.15-1.68 |

Statistics for the subsumption matching set-ups. Note that each subsumption matching task is constructed from an equivalence matching task subject to target side class deletion.


| Source | Task        | Category | #Classes      | #RefMaps (subs)  |
|--------|-------------|----------|---------------|------------------|
| Mondo  | OMIM-ORDO   | Disease  | 9,642-8,735   | 103              | 
| Mondo  | NCIT-DOID   | Disease  | 6,835-5,113   | 3,339            | 
| UMLS   | SNOMED-FMA  | Body     | 24,182-59,567 | 5,506            | 
| UMLS   | SNOMED-NCIT | Pharm    | 16,045-12,462 | 4,225            | 
| UMLS   | SNOMED-NCIT | Neoplas  | 11,271-13,790 | 213              | 

The downloaded datasets include `Mondo.zip` and `UMLS.zip` for resources constructed from Mondo and UMLS, respectively.
Each `.zip` file has three folders: `raw_data`, `equiv_match`, and `subs_match`, corresponding to the raw source ontologies, data for equivalence matching, and data for subsumption matching, respectively. Detailed structure is presented in the following figure. 

<br/>
<p align="center">
  <a href="https://doi.org/10.5281/zenodo.6917501">
    <img alt="deeponto" src="https://raw.githubusercontent.com/KRR-Oxford/DeepOnto/main/docs/images/largebiomeddata.svg" height="600" style="width: 100%;">
  </a>
</p>

There are two evaluation schemes (**local ranking** and **global matching**) and two data split settings (**unsupervised** and **(semi-)supervised**).

- For local ranking, an OM model is required to rank candidates stored in `src2tgt.rank`.
  -  `src2tgt` here means the **anchors/keys** are the source ontology classes, and the **candidates/values** are generated from the target ontology. There are three options for loading the anchored candidate mapping:
     -  *Option 1*: Load the whole data folder using [`AnchoredOntoMappings`](data_structures?id=anchoredontomappings) implemented in DeepOnto: 
      ```python
      from deeponto.onto.mapping import AnchoredOntoMappings
      AnchoredOntoMappings.from_saved("src2tgt.rank")
      ```
     -  *Option 2*: Load the `.json` file in the `src2tgt.rank`, a nested dictionary in form of `{"(ref_src, ref_tgt)": tgt_cand: 0.0}` where `(ref_src, ref_tgt)` is a reference class pair (anchor), `tgt_cand` is a target candidate w.r.t. this anchor, `0.0` is the default score for the candidate mapping `(ref_src, tgt_cand)`.
      ```python
      from deeponto import SavedObj
      SavedObj.load_json("src2tgt.rank/src2tgt.anchored.maps.json")
      ```  
      - *Option 3*: Load the `.tsv` file in the `src2tgt.rank`, where the columns are `["SrcEntity", "TgtEntity", "TgtCandidates"]` standing for the source class iri of a reference mapping, the target class iri of this reference mapping, and the corresponding target candidate class iris in a sequence, which can be decoded using `ast.literal_eval(tgt_cands_seq)`.

  - [`AnchoredOntoMappings`](data_structures?id=anchoredontomappings) is essentially a dictionary with each reference mapping (in the form of class tuple) as a key (anchor) and its corresponding candidates (100 negative + 1 positive classes from the target ontology). 
  - `Hits@K` and `MRR` are used as evaluation metrics. Note that the candidates are separately generated w.r.t the testing mapping in each data split setting.

- For global matching, an OM model is required to output full mappings and compare them with the reference mappings using `Precision`, `Recall`, and `F1`. 
  - For each OM pair, a `refs/full.tsv` file is provided for full reference mapping; the columns of this `.tsv` file are `["SrcEntity", "TgtEntity", "Score"]` standing for the source reference class iri, target class iri, and the score (set to $1.0$ for reference mappings).
    -  `val.tsv` and `test.tsv` are provided in `refs/unsupervised` for validation (10%) and testing (90%), respectively.
    -  `train.tsv`, `val.tsv`, `train+val.tsv` and `test.tsv` are provided in `refs/semi_supervised` for training (20%), validation (10%), merged training and validation file for evaluation, and testing (70%), respectively.
  - When computing these scores, mappings that are not in the testing set should be ignored by substraction from both system output and reference mappings. 
    - i.e., when evaluating on `refs/unsupervised/test.tsv`, `refs/unsupervised/val.tsv` should be ignored; when evaluating on `refs/semi_supervised/test.tsv`, `refs/semi_supervised/train+val.tsv` should be ignored. 
    - This feature is supported in [`om_eval.py`](using_deeponto?id=om-evaluation) script of DeepOnto where the arguement `null_ref_path` is for inputting the reference mappings that should be ignored.

- Since the subsumption mappings are inherently incomplete, we suggest apply only local ranking for evaluating subsumption matching.

