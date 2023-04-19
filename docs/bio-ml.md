# Bio-ML Sepecifications

!!! credit "paper"

    Paper for Bio-ML:
    [Machine Learning-Friendly Biomedical Datasets for Equivalence and Subsumption Ontology Matching (ISWC 2022)](https://link.springer.com/chapter/10.1007/978-3-031-19433-7_33). It was nominated as the **best resource paper candidate** at ISWC 2022.

    ```
    @inproceedings{he2022machine,
      title={Machine Learning-Friendly Biomedical Datasets for Equivalence and Subsumption Ontology Matching},
      author={He, Yuan and Chen, Jiaoyan and Dong, Hang and Jim{\'e}nez-Ruiz, Ernesto and Hadian, Ali and Horrocks, Ian},
      booktitle={The Semantic Web--ISWC 2022: 21st International Semantic Web Conference, Virtual Event, October 23--27, 2022, Proceedings},
      pages={575--591},
      year={2022},
      organization={Springer}
    }
    ```


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
| Mondo  | OMIM-ORDO   | Disease  | 9,642-8,838    | 3,721           | 34K-34K  | 1.44-1.63 |
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

Each OM pair in $\textsf{Bio-ML}$ consists of an equivalence matching track and a subsumption matching track. Each track considers two perspectives for evaluation, **global matching** and **local ranking**, each of which has two data split settings, **unsupervised** and **semi-supervised**.

#### Local Ranking

Local ranking aims to examine an OM model's ability to distinguish a correct mapping out of several challenging negatives. The model should assign a high score (thus a high ranking) to the correct mapping. The overall results are evaluated using $Hits@K$ and $MRR$.

In $\textsf{Bio-ML}$, candidate mappings are generated for each reference mapping in the test set. As shown in the [file structure](#file-structure), each task setting contains a `test.cands.tsv` file with each entry a reference mapping and its corresponding target class candidates -- which can be combined with the source reference class to form the candidate mappings. 

> Download a <a href="../assets/example_candidate_mappings.tsv" download>small fragment</a>.

A `test.cands.tsv` can be loaded with the following code:

```python
from deeponto.utils import FileUtils
df = FileUtils.read_table("test.cands.tsv")
```

The `"SrcEntity"` and `"TgtEntity"` columns refer to the source class IRI and the target class IRI involved in a reference mapping. The `"TgtCandidates"` column stores a sequence of target candidate class IRIs (**including the correct one**) used for ranking, which can be accessed by the built-in `eval` function as:

```python
# get the first sample's candidates
eval(df.loc[0]["TgtCandidates"])
```

`#!console Output:`
:   &#32;
    ```python
    ('http://purl.obolibrary.org/obo/DOID_0110279',
     'http://purl.obolibrary.org/obo/DOID_3185',
     'http://purl.obolibrary.org/obo/DOID_7008',
     ...)
    ```

An OM model is expected to compute a score (or a relative ranking) for each candidate class in `"TgtCandidates"` -- to decide how likely it can be matched with the source reference class. In the ideal case, the reference target class (as in `"TgtEntity"`) should be ranked first.

The candidate mappings were separately generated w.r.t. the tesing data (`test.tsv`) in each data split.

`unsupervised`
:   `test.cands.tsv` in `refs/unsupervised` refers to candidate mappings generated from `refs/unsupervised/  test.tsv` and `refs/unsupervised/val.tsv` is ensured to be excluded from candidates.

`semi_supervised`
:  `test.cands.tsv` in `refs/semi_supervised` referes to candidate mappings generated from `refs/semi_supervised/test.tsv` and `refs/semi_supervised/train+val.tsv` is ensured to be excluded from candidate mappings generated from candidates.
  
#### Global Matching

Global matching aims to examine the overall OM performance by comparing the output mappings with the reference mappings using $Precision$, $Recall$, and $F1$.

For each OM pair, a `refs/full.tsv` file is provided for the full set of reference mappings; the columns of this `.tsv` file have the headings `"SrcEntity"`, `"TgtEntity"`, `"Score"` standing for the source reference class, target reference class, and the score (automatically set to $1.0$ for reference mappings). 

> Download a <a href="../assets/example_reference_mappings.tsv" download>small fragment</a>.


A reference mapping file such as `refs/full.tsv` can be loaded using:

```python
from deeponto.align.mapping import ReferenceMapping
refs = ReferenceMapping.read_table_mappings("refs/full.tsv")
```

Reference mappings in `full.tsv` are further divided into different splits for training (semi-supervised), validation, and testing.

`unsupervised`
:  `val.tsv` and `test.tsv` are provided in `refs/unsupervised` for validation ($10\%$) and testing ($90\%$), respectively.

`semi-supervised`
:  `train.tsv`, `val.tsv`, `train+val.tsv` and `test.tsv` are provided in `refs/semi_supervised` for training ($20\%$), validation ($10\%$), merged training and validation file for evaluation, and testing ($70\%$), respectively.
  

!!! tip

    When computing the scores (P, R, F1), mappings that are not in the testing set should be **ignored by substraction from both system output and reference mappings**. 
      - i.e., when evaluating on `unsupervised/test.tsv`, `unsupervised/val.tsv` should be ignored; when evaluating on `semi_supervised/test.tsv`, `semi_supervised/train+val.tsv` should be ignored. 
      - This feature is supported in the code [here][deeponto.align.evaluation.AlignmentEvaluator.f1] where the arguement `null_reference_mappings` is for inputting the reference mappings that should be ignored.

    Since the subsumption mappings are inherently incomplete, we suggest apply **only local ranking for evaluating subsumption matching**.




## OM Dataset Construction



### Ontology Pruning

In order to obtain scalable OM pairs, the **ontology pruning** algorithm proposed in the $\textsf{Bio-ML}$ paper can be used to truncate a large-scale ontology according to certain criteria such as the **semantic type**. The pruning function aims to remove unwanted ontology classes while preserving the relevant hierarchy. Specifically, for each class $c$ to be removed, subsumption axioms will be created between the parents of $c$ and the children of $c'$. Following this is the removal of all axioms related to the unwanted classes.

Once obtaining the list of class IRIs to be removed, apply the ontology pruning following the code below:

```python
from deeponto.onto import Ontology, OntologyPruner

doid = Ontology("doid.owl")
pruner = OntologyPruner(doid)
to_be_removed_class_iris = [
    "http://purl.obolibrary.org/obo/DOID_0060158",
    "http://purl.obolibrary.org/obo/DOID_9969"
]
pruner.min_hierarchy_prune(to_be_removed_class_iris)
pruner.save_onto("doid.pruned.owl")  # save the pruned ontology locally
```


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
```

`#!console Output:`
:   &#32;
    ```python
    3299/4686 are used for creating at least one subsumption mapping.
    3305 subsumption mappings are created in the end.
    ```

Get the generated subsumption mappings by:

```python
subs_generator.subs_from_equivs
```

`#!console Output:`
:   &#32;
    ```python
    [('http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C9311',
      'http://purl.obolibrary.org/obo/DOID_120',
      1.0),
     ('http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C8410',
      'http://purl.obolibrary.org/obo/DOID_1612',
      1.0), ...]
    ```

The `subs_generation_ratio` parameter determines at most how many subsumption mappings can be generated from an equivalence mapping. The `delete_used_equiv_tgt_class` determines whether or not to sabotage equivalence mappings used for creating at least one subsumption mappings. If it is set to `#!python True`, then the target side of an (**used**) equivalence mapping will be marked as deleted from the target ontology. Then, apply ontology pruning to the list of to-be-deleted target ontology classes which can be accessed as `subs_generator.used_equiv_tgt_class_iris`.

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

Sampling using the *idf scores* is originally proposed in the BERTMap paper. The parameter `annotation_property_iris` specifies the list of annotation properties used for extracting the **names** or **aliases** of an ontology class. The parameter `tokenizer` refers to a pre-trained sub-word level tokenizer used to build the inverted annotation index. They have been well-explained in the [BERTMap tutorial](../bertmap).
