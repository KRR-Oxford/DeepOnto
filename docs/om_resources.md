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

# Ontology Matching Resources

!!! credit "paper"

    The paper for the $\textsf{Bio-ML}$ ontology matching dataset:
    [*Machine Learning-Friendly Biomedical Datasets for Equivalence and Subsumption Ontology Matching (ISWC 2022)*](https://link.springer.com/chapter/10.1007/978-3-031-19433-7_33).

This page provides detailed [instructions](#bio-ml) for using $\textsf{Bio-ML}$.

It also gives the tutorial for how to apply the [OM dataset construction approaches](#om-dataset-construction) as proposed in the $\textsf{Bio-ML}$,  which can be applied to other ontologies.


## Bio-ML

The **Bio-ML** dataset provides **five** ontology pairs for both equivalence and subsumption ontology matching. These OM pairs are constructed using the approaches described in the [OM construction section](#om-dataset-construction).

- **Download link**: *<https://doi.org/10.5281/zenodo.6510086>* (CC BY 4.0 International).
- **Resource Paper**: *<https://arxiv.org/abs/2205.03447>*.
- **OAEI Track**: *<https://www.cs.ox.ac.uk/isg/projects/ConCur/oaei/>*. 

### Data Statistics

<!-- tabs:start -->

#### **Equivalence Matching**

Statistics for the equivalence matching set-ups. In the **Category** column, *"Disease"* indicates that the Mondo data are mainly about disease concepts, while *"Body"*, *"Pharm"*, and *"Neoplas"* denote semantic types of *"Body Part, Organ, or Organ Components"*, *"Pharmacologic Substance*"*, and *"Neoplastic Process"* in UMLS, respectively.

<center>
<small>

| Source | Task        | Category | #Classes      | #RefMaps (equiv) | #Annot.  | AvgDepths |
|--------|:-----------:|:--------:|:-------------:|:----------------:|:--------:|:---------:|
| Mondo  | OMIM-ORDO   | Disease  | 9,642-8838    | 3,721            | 34K-34K  | 1.44-1.63 |
| Mondo  | NCIT-DOID   | Disease  | 6,835-8,848   | 4,684            | 80K-38K  | 2.04-6.85 |
| UMLS   | SNOMED-FMA  | Body     | 24,182-64,726 | 7,256            | 39K-711K | 1.86-9.32 |
| UMLS   | SNOMED-NCIT | Pharm    | 16,045-15,250 | 5,803            | 19K-220K | 1.09-3.26 |
| UMLS   | SNOMED-NCIT | Neoplas  | 11,271-13,956 | 3,804            | 23K-182K | 1.15-1.68 |

</small>
</center>


#### **Subsumption Matching**

Statistics for the subsumption matching set-ups. Note that each subsumption matching task is constructed from an equivalence matching task subject to target side class deletion.

<center>
<small>

| Source | Task        | Category | #Classes      | #RefMaps (subs)  |
|--------|:-----------:|:--------:|:-------------:|:----------------:|
| Mondo  | OMIM-ORDO   | Disease  | 9,642-8,735   | 103              | 
| Mondo  | NCIT-DOID   | Disease  | 6,835-5,113   | 3,339            | 
| UMLS   | SNOMED-FMA  | Body     | 24,182-59,567 | 5,506            | 
| UMLS   | SNOMED-NCIT | Pharm    | 16,045-12,462 | 4,225            | 
| UMLS   | SNOMED-NCIT | Neoplas  | 11,271-13,790 | 213              | 

</small>
</center>

<!-- tabs:end -->


### File Structure

The downloaded datasets include `Mondo.zip` and `UMLS.zip` for resources constructed from Mondo and UMLS, respectively.
Each `.zip` file has three folders: `raw_data`, `equiv_match`, and `subs_match`, corresponding to the raw source ontologies, data for equivalence matching, and data for subsumption matching, respectively. Detailed structure is presented in the following figure. 

<br/>
<p align="center">
  <img alt="deeponto" src="../images/largebiomeddata.svg" height="600" style="width: 100%;">
</p>

### Evaluation Framework

There are two evaluation schemes (**local ranking** and **global matching**) and two data split settings (**unsupervised** and **semi-supervised**).


#### Local Ranking

For **local ranking**, an OM model is required to rank candidate mappings (in `test.cands.tsv`) generated from test mappings and evalute using $Hits@K$ and $MRR$. 

Load a `test.cands.tsv` file using:

```python
from deeponto.utils import FileUtils
df = FileUtils.read_table("test.cands.tsv")  # with headings: "SrcEntity", "TgtEntity", "TgtCandidates"
df.head(3)
>>> 	SrcEntity	TgtEntity	TgtCandidates
0	http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus...	http://purl.obolibrary.org/obo/DOID_0050806	('http://purl.obolibrary.org/obo/DOID_0110279'...
1	http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus...	http://purl.obolibrary.org/obo/DOID_775	('http://purl.obolibrary.org/obo/DOID_1670', '...
2	http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus...	http://purl.obolibrary.org/obo/DOID_4917	('http://purl.obolibrary.org/obo/DOID_3704', '...
```

The `"SrcEntity"` and `"TgtEntity"` columns refer to the source class IRI and the target class IRI involved in a reference mapping.
The `"TgtCandidates"` column stores a sequence of target candidate class IRIs (including the correct one) used for ranking, which can be accessed as:

```python
eval(df.loc[0]["TgtCandidates"])
>>> ('http://purl.obolibrary.org/obo/DOID_0110279',
 'http://purl.obolibrary.org/obo/DOID_3185',
 'http://purl.obolibrary.org/obo/DOID_7008',
 ...)
```

An OM model is expected to re-rank the candidates in `"TgtCandidates"` according to the given reference source class in `"SrcEntity"`. The higher rank of `"TgtEntity"` is among the `"TgtCandidates"`, the higher the ranking score will be.


The candidate mappings were separately generated w.r.t. the tesing data (`test.tsv`) in each data split.

  - *Unsupervised*: `test.cands.tsv` in `refs/unsupervised` refers to candidate mappings generated from `refs/unsupervised/test.tsv` and `refs/unsupervised/val.tsv` is ensured to be excluded from candidates.
  - *Semi-supervised*: `test.cands.tsv` in `refs/semi_supervised` referes to candidate mappings generated from `refs/semi_supervised/test.tsv` and `refs/semi_supervised/
  
#### Global Matching

For **global matching**, an OM model is required to output full mappings and compare them with the reference mappings using $Precision$, $Recall$, and $F1$.

For each OM pair, a `refs/full.tsv` file is provided for full reference mapping; the columns of this `.tsv` file are `["SrcEntity", "TgtEntity", "Score"]` standing for the source reference class iri, target class iri, and the score (set to $1.0$ for reference mappings). 

Load a mapping file using:

```python
from deeponto.align.mapping import ReferenceMapping
refs = ReferenceMapping.read_table_mappings("refs/full.tsv")
```

 The full reference mappings in `full.tsv` are divided into different splits for training (semi-supervised), validation, and testing purposes.

  -  *Unsupervised*: `val.tsv` and `test.tsv` are provided in `refs/unsupervised` for validation (10%) and testing (90%), respectively.
  -  *Semi-supervised*: `train.tsv`, `val.tsv`, `train+val.tsv` and `test.tsv` are provided in `refs/semi_supervised` for training (20%), validation (10%), merged training and validation file for evaluation, and testing (70%), respectively.
  

!!! tip

    When computing the scores (P, R, F1), mappings that are not in the testing set should be **ignored by substraction from both system output and reference mappings**. 
      - i.e., when evaluating on `refs/unsupervised/test.tsv`, `refs/unsupervised/val.tsv` should be ignored; when evaluating on `refs/semi_supervised/test.tsv`, `refs/semi_supervised/train+val.tsv` should be ignored. 
      - This feature is supported in the code [here][deeponto.align.evaluation.AlignmentEvaluator.f1] where the arguement `null_reference_mappings` is for inputting the reference mappings that should be ignored.

    Since the subsumption mappings are inherently incomplete, we suggest apply **only local ranking for evaluating subsumption matching**.




## OM Dataset Construction

### Ontology Pruning

In order to obtain scalable OM pairs, the **ontology pruning** algorithm proposed in the $\textsf{Bio-ML}$ paper can be used to truncate a large-scale ontology according to certain criteria such as the **semantic type**. Once obtaining the list of class IRIs to be truncated, apply the ontology pruning following the code [here](/ontology/#ontology-pruning).

### Subsumption Mapping Construction

It is often that OM datasets include equivalence matching but **not** subsumption matching. However, it is possible to create a subsumption matching task from each equivalence matching task. Given a list of **equivalence reference mappings** (see [how to load from a local file][deeponto.align.mapping.ReferenceMapping.read_table_mappings]) in the form of $\{(c, c') | c \equiv c' \}$, the reference subsumption mappings can be created by searching for the **subsumers** of $c'$, then yielding $\{(c, c'') | c \equiv c', c' \sqsubseteq c'' \}$. We have implemented a [subsumption mapping generator][deeponto.align.mapping.SubsFromEquivMappingGenerator] for this purpose:

```python
from deeponto.onto import Ontology
from deeponto.align.mapping import SubsFromEquivMappingGenerator, ReferenceMapping

ncit = Ontology("ncit.owl")  # load the NCIt ontology
doid = Ontology("doid.owl")  # load the disease ontology
ncit2doid_equiv_mappings = ReferenceMapping.read_table_mappings("ncit2doid_equiv_mappings.tsv")  # with headings ["SrcEntity", "TgtEntity", "Score"]

subs_generator = SubsFromEquivMappingGenerator(
  ncit, doid, ncit2doid_equiv_mappings, 
  subs_generation_ratio=1, delete_used_equiv_tgt_class=True
)
>>> 3299/4686 are used for creating at least one subsumption mapping.
3305 subsumption mappings are created in the end.

subs_generator.subs_from_equivs
>>> [('http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C9311',
  'http://purl.obolibrary.org/obo/DOID_120',
  1.0),
 ('http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C8410',
  'http://purl.obolibrary.org/obo/DOID_1612',
  1.0), ...]
```

The `subs_generation_ratio` parameter determines at most how many subsumption mappings can be generated from an equivalence mapping. The `delete_used_equiv_tgt_class` determines whether or not to sabotage equivalence mappings used for creating at least one subsumption mappings. If setting to `True`, then the target side of an (**used**) equivalence mapping will be marked as deleted from the target ontology. Then, apply ontology pruning to the list of to-be-deleted target ontology classes which can be accessed as `subs_generator.used_equiv_tgt_class_iris`.

### Negative Candidate Mapping Generation

In order to examine an OM model's ability to distinguish the correct mapping among a set of challenging negative candidates, we can apply the [negative canddiate mapping generation][deeponto.align.mapping.NegativeCandidateMappingGenerator] algorithm as proposed in the paper, which utilises the [`idf_sample`][deeponto.align.mapping.NegativeCandidateMappingGenerator.idf_sample] to generate candidates ambiguious at the textual level (similar naming), and [`neighbour_sample`][deeponto.align.mapping.NegativeCandidateMappingGenerator.neighbour_sample] to generate candidates ambiguious at the structural level (e.g., siblings). The algorithm makes sure that **none** of the reference mappings will be added as a negative candidate, and for the subsumption case, the algorithm also takes care of **excluding the ancestors** because they are correct subsumptions.

```python
from deeponto.onto import Ontology
from deeponto.align.mapping import NegativeCandidateMappingGenerator, ReferenceMapping

ncit = Ontology("ncit.owl")  # load the NCIt ontology
doid = Ontology("doid.owl")  # load the disease ontology
ncit2doid_equiv_mappings = ReferenceMapping.read_table_mappings("ncit2doid_equiv_mappings.tsv")  # with headings ["SrcEntity", "TgtEntity", "Score"]

cand_generator = NegativeCandidateMappingGenerator(
  ncit, doid, ncit2doid_equiv_mappings, 
  annotation_property_iris = [...],  # used for idf sample
  tokenizer=Tokenizer.from_pretrained(...),  # used for idf sample
  max_hops=5, # used for neighbour sample
  for_subsumptions=False,  # set to False because the input mappings in this example are equivalence mappings
)
```

Sampling using the *idf scores* is originally proposed in the BERTMap paper. The parameter `annotation_property_iris` specifies the list of annotation properties used for extracting the **names** or **aliases** of an ontology class. The parameter `tokenizer` refers to a pre-trained sub-word level tokenizer used to build the inverted annotation index. They have been well-explained in the [BERTMap tutorial](/bertmap).
