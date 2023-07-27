# Bio-ML: A Comprehensive Documentation

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

## About

$\textsf{Bio-ML}$ incorporates **five** ontology pairs for both equivalence and subsumption ontology matching, two of them are based on **Mondo** and three of them are based on **UMLS**. These datasets are constructed with the following steps:

- **Ontology Preprocessing**: Checking ontology integrity, removing deprecated or redundant classes.
- **Ontology Pruning**: Obtaining a sub-ontology subject to a list of preserved class IRIs. For Mondo ontologies, classes are preserved based on the **reference mappings**; For UMLS ontologies, classes are preserved based on the **semantic types** (see [Ontology Pruning](#ontology-pruning)).
- **Subsumption Mapping Construction**: Reference subsumption mappings are constructed from the reference equivalence mappings, subject to target class deletion, i.e., if an equivalence mapping is used for constructing a subsumption mapping, its target ontology class will be removed to enforce direct subsumption matching (see [Subsumption Mapping Construction](#subsumption-mapping-construction)). 
- **Candidate Mapping Generation**: To evaluate an OM system using ranking-based metrics, we generate a list of negative candidate mappings for each reference mapping using different heuristics (see [Candidate Mapping Generation](#candidate-mapping-generation)).
- **Locality Module Enrichment** (NEW :star2:): In the OAEI 2023 version, we enrich the pruned ontologies with classes that serve as context (and marked as **not used in alignment**) for the existing classes, using the **locality module** technique ([code](https://github.com/ernestojimenezruiz/logmap-matcher/blob/master/src/test/java/uk/ac/ox/krr/logmap2/test/oaei/CreateModulesForBioMLTrack.java)). OM systems can utilise these additional classes as auxiliary information while omitting them in the alignment process; the final evaluation will omit these additional classes as well.
- **Bio-LLM: A Special Sub-Track for Large Language Models** (NEW :star2:): In the OAEI 2023 version, we introduce a special sub-track for Large Language Model (LLM)-based OM systems by extracting small but challenging subsets from NCIT-DOID and SNOMED-FMA (Body) datasets. See [OAEI Bio-LLM 2023](#oaei-bio-llm-2023) for detail.

## Links

- **Dataset Download** (CC BY 4.0 International):
    - **OAEI 2022**: <https://zenodo.org/record/6946466> (see [OAEI Bio-ML 2022](#oaei-bio-ml-2022))
    - **OAEI 2023** (NEW :star2:): to be released (see [OAEI Bio-ML 2023](#oaei-bio-ml-2023))

- **Detailed Documentation**: *<https://krr-oxford.github.io/DeepOnto/bio-ml/>* (this page)
- **Resource Paper**: *<https://arxiv.org/abs/2205.03447>* (arXiv version)
- **Official OAEI Page**: *<https://www.cs.ox.ac.uk/isg/projects/ConCur/oaei/index.html>*


## Ontology Pruning

In order to obtain scalable OM pairs, the **ontology pruning** algorithm proposed in the $\textsf{Bio-ML}$ paper can be used to truncate a large-scale ontology according to certain criteria such as being involved in a **reference mapping** or being associated with certain **semantic type**. The pruning function aims to remove unwanted ontology classes while preserving the relevant hierarchy. Specifically, for each class $c$ to be removed, subsumption axioms will be created between the parents of $c$ and the children of $c'$. Following this is the removal of all axioms related to the unwanted classes.

Once obtaining a list of class IRIs to be removed, apply the ontology pruning following the code below:

```python
from deeponto.onto import Ontology, OntologyPruner

doid = Ontology("doid.owl")
pruner = OntologyPruner(doid)
to_be_removed_class_iris = [
    "http://purl.obolibrary.org/obo/DOID_0060158",
    "http://purl.obolibrary.org/obo/DOID_9969"
]
pruner.prune(to_be_removed_class_iris)
pruner.save_onto("doid.pruned.owl")  # save the pruned ontology locally
```

## Subsumption Mapping Construction

It is often that OM datasets include equivalence matching but **not** subsumption matching. However, it is possible to create a subsumption matching task from an equivalence matching task. Given a list of **equivalence reference mappings** (see [how to load from a local file][deeponto.align.mapping.ReferenceMapping.read_table_mappings]) in the form of $\{(c, c') | c \equiv c' \}$, the reference subsumption mappings can be created by searching for the **subsumers** of $c'$, then yielding $\{(c, c'') | c \equiv c', c' \sqsubseteq c'' \}$. We have implemented a [subsumption mapping generator][deeponto.align.mapping.SubsFromEquivMappingGenerator] for this purpose:

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

## Candidate Mapping Generation

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



## Evaluation Framework

Our evaluation protocol concerns two scenarios for OM: **global matching** for overall assessment and **local ranking** for partial assessment.
### Global Matching

As an overall assessment, given a **complete** set of reference mappings, an OM system is expected to compute a set of *true* mappings and compare against the reference mappings using Precision, Recall, and F-score metrics. With $\textsf{DeepOnto}$, the evaluation can be performed using the following code. 

> Download a <a href="../assets/example_reference_mappings.tsv" download>small fragment</a> to see the format of the prediction and reference mapping files. The three columns, `"SrcEntity"`, `"TgtEntity"`, and `"Score"` refer to the source class IRI, the target class IRI, and the matching score.

```python
from deeponto.align.evaluation import AlignmentEvaluator
from deeponto.align.mapping import ReferenceMapping, EntityMapping

# load prediction mappings and reference mappings
preds = EntityMapping.read_table_mappings(f"{experiment_dir}/bertmap/match/repaired_mappings.tsv")
refs = ReferenceMapping.read_table_mappings(f"{data_dir}/refs_equiv/full.tsv")
# compute the precision, recall and F-score metrics
results = AlignmentEvaluator.f1(preds, refs)
print(results)
```

`#!console Output:`
:   &#32;
    ```python
    {'P': 0.887, 'R': 0.879, 'F1': 0.883}
    ```

For the semi-supervised setting where a set of training mappings is provided, the training set should also be loaded and set as **ignored** (neither positive nor negative) during evaluation:

```python
train_refs = ReferenceMapping.read_table_mappings(f"{data_dir}/refs_equiv/train.tsv")
results = AlignmentEvaluator.f1(preds, refs, null_reference_mappings=train_refs)
```

As for the OAEI 2023 version, some prediction mappings could involve classes that are marked as **not used in alignment**. Therefore, we need to filter out those mappings before evaluation.

```python
from deeponto.onto import Ontology
from deeponto.align.oaei import *

# load the source and target ontologies and  
# extract classes that are marked as not used in alignment
src_onto = Ontology("src_onto_file")
tgt_onto = Ontology("tgt_onto_file")
ignored_class_index = get_ignored_class_index(src_onto)
ignored_class_index.update(get_ignored_class_index(tgt_onto))

# filter the prediction mappigns
preds = remove_ignored_mappings(preds, ignored_class_index)

# then compute the results
results = AlignmentEvaluator.f1(preds, refs, ...)
```

!!! tips

    We have encapsulated above features in the [`matching_eval`][deeponto.align.oaei.matching_eval] function in the OAEI utilities.

However,

- The scores will be biased towards high-precision, low-recall OM systems if the set of reference mappings is incomplete. 
- For efficient OM system development and debugging, an intermediate evaluation is required.

Therefore, the ranking-based evaluation protocol is present as follows.

### Local Ranking

An OM system is also expected to **distinguish the reference mapping** among a set of candidate mappings and the performance can be reflected in Hits@K and MRR metrics. 

!!! warning 

    The reference subsumption mappings are inherently incomplete, so only the ranking metircs are adopted in evaluating system performance in subsumption matching.

> Download a <a href="../assets/example_candidate_mappings.tsv" download>small fragment</a> to see the format of the reference mapping and its candidate mappings. The `"SrcEntity"` and `"TgtEntity"` columns refer to the source class IRI and the target class IRI involved in a **reference mapping**. The `"TgtCandidates"` column stores a sequence of target candidate class IRIs (**including the correct one**) used for ranking, which can be accessed by the built-in Python function `eval`.

With $\textsf{DeepOnto}$, the evaluation can be performed as follows. First, an OM system needs to assign a score to each target candidate class and save the results as a list of tuples `(tgt_cand_class_iri, matching_score)`. 

```python
from deeponto.utils import FileUtils
import pandas as pd

test_candidate_mappings = FileUtils.read_table("test.cands.tsv").values.to_list()
ranking_results = []
for src_ref_class, tgt_ref_class, tgt_cands in test_candidate_mappings:
    tgt_cands = eval(tgt_cands)  # transform string into list or sequence
    scored_cands = []
    for tgt_cand in tgt_cands:
        # assign a score to each candidate with an OM system
        ...
        scored_cands.append((tgt_cand, matching_score))
    ranking_results.append((src_ref_class, tgt_ref_class, scored_cands))
# save the scored candidate mappings in the same format as the original `test.cands.tsv`
pd.DataFrame(ranking_results, columns=["SrcEntity", "TgtEntity", "TgtCandidates"]).to_csv("scored.test.cands.tsv", sep="\t", index=False)
```

Then, the ranking evaluation results can be obtained by:

```python
from deeponto.align.oaei import *

ranking_eval("scored.test.cands.tsv")
```

!!! tips

    If matching scores are not available, the target candidate classes should be **sorted** in descending order and saved in a list, the [`ranking_eval`][deeponto.align.oaei.ranking_eval] function will compute scores according to the sorted list.


## OAEI Bio-ML 2022

Below demonstrates the data statistics for the original Bio-ML used in the OAEI 2022. In the **Category** column, *"Disease"* indicates that the Mondo data are mainly about disease concepts, while *"Body"*, *"Pharm"*, and *"Neoplas"* denote semantic types of *"Body Part, Organ, or Organ Components"*, *"Pharmacologic Substance*"*, and *"Neoplastic Process"* in UMLS, respectively. 

Note that each subsumption matching task is constructed from an equivalence matching task subject to **target ontology class deletion**, therefore `#TgtCls (subs)` is different with `#TgtCls`.

<center>
<small>

| Source | Task        | Category | #SrcCls | #TgtCls | #TgtCls (subs) | #Ref (equiv) | #Ref (subs) |
|--------|:-----------:|:--------:|:-------:|:-------:|:--------------:|:------------:|:-----------:|
| Mondo  | OMIM-ORDO   | Disease  | 9,642   | 8,838   | 8,735          | 3,721        | 103         |
| Mondo  | NCIT-DOID   | Disease  | 6,835   | 8,448   | 5,113          | 4,684        | 3,339       | 
| UMLS   | SNOMED-FMA  | Body     | 24,182  | 64,726  | 59,567         | 7,256        | 5,506       |
| UMLS   | SNOMED-NCIT | Pharm    | 16,045  | 15,250  | 12,462         | 5,803        | 4,225       |
| UMLS   | SNOMED-NCIT | Neoplas  | 11,271  | 13,956  | 13,790         | 3,804        | 213         |

</small>
</center>

The downloaded datasets (from Zenodo) include `Mondo.zip` and `UMLS.zip` for resources constructed from Mondo and UMLS, respectively. Each `.zip` file has three folders: `raw_data`, `equiv_match`, and `subs_match`, corresponding to the raw source ontologies, data for equivalence matching, and data for subsumption matching, respectively. Detailed structure is presented in the following figure. 

<br/>
<p align="center">
  <img alt="deeponto" src="../images/bio-ml-oaei-2022.svg" height="420" style="width: 70%;">
</p>


## OAEI Bio-ML 2023

The 2023 version has made several changes towards the previous version...

(to be updated)

Below demonstrates the data statistics for the OAEI 2023 version of Bio-ML, where the input ontologies are extended to the modualarizations of their pruned versions used in 2022 (available at `raw_data`), through which **structural and logical contexts** are added and the input ontologies become closer to the original ontologies. To ensure the completeness of the original reference mappings, the added ontology classes are marked as **not used in alignment** through the annotation property `use_in_alignment` with a value of `false`. OM systems can choose to use these classes for enhancement but do not need to consider them for final output mappings. Even they are considered for the final output mappings, our evaluation will ensure that they are **excluded in the metric computation** (see [Evaluation Framework](#evaluation-framework)). 

In the **Category** column, *"Disease"* indicates that the Mondo data are mainly about disease concepts, while *"Body"*, *"Pharm"*, and *"Neoplas"* denote semantic types of *"Body Part, Organ, or Organ Components"*, *"Pharmacologic Substance*"*, and *"Neoplastic Process"* in UMLS, respectively. 

The changes compared to the previous version (see [Bio-ML OAEI 2022](#bio-ml-oaei-2022)) are reflected in the `+` numbers of ontology classes. 

<center>
<small>

| Source | Task        | Category | #SrcCls | #TgtCls | #TgtCls (subs) | #Ref (equiv) | #Ref (subs)  |
|--------|:-----------:|:--------:|:-------:|:-------:|:--------------:|:------------:|:------------:|
| Mondo  | OMIM-ORDO   | Disease  | 9,648 (+6)      | 9,275 (+437)    | 9,271 (+536) | 3,721 | 103   |
| Mondo  | NCIT-DOID   | Disease  | 15,762 (+8,927) | 8,465 (+17)     | 5,722 (+609) | 4,684 | 3,339 | 
| UMLS   | SNOMED-FMA  |Body | 34,418 (+10,236)|88,955 (+24,229)|88,648 (+20,081)| 7,256  | 5,506    |
| UMLS   | SNOMED-NCIT |Pharm| 29,500 (+13,455)|22,136 (+6,886) |20,113 (+7,651) | 5,803  | 4,225    |
| UMLS   | SNOMED-NCIT | Neoplas  | 22,971 (+11,700) | 20,247 (+6291) | 20,113 (+6,323) | 3,804 | 213|

</small>
</center>

The file structure for the download datasets (from Zenodo) is simplified this year to accommodate the changes.

Detailed structure is presented in the following figure (not yet available). 

<br/>
<p align="center">
  <img alt="deeponto" src="../images/bio-ml-oaei-2023" height="420" style="width: 70%;">
</p>

## OAEI Bio-LLM 2023

As Large Language Models (LLMs) are trending in the AI community, we formulate a special sub-track for evaluating LLM-based OM systems. For efficient and insightful evaluation, we select two small yet representative subsets from the NCIT-DOID and SNOMED-FMA (Body) datasets, each consisting of 50 **matched** and 50 **unmatched** class pairs. 

We have evaluated some LLMs with several settings and submit a poster paper. The results and more detail about this track will be released when the paper review is finished.
