# Copyright 2023 Yuan He. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from typing import List, Optional
import warnings

from deeponto.onto import Ontology
from deeponto.align.mapping import ReferenceMapping, EntityMapping
from deeponto.utils import FileUtils
from deeponto.align.evaluation import AlignmentEvaluator


################################################################
###                for `use_in_alignment` annotation         ###
################################################################


def get_ignored_class_index(onto: Ontology):
    """Get an index for filtering classes that are marked as not used in alignment.

    This is indicated by the special class annotation `use_in_alignment` with the following IRI:
        http://oaei.ontologymatching.org/bio-ml/ann/use_in_alignment
    """
    ignored_class_index = defaultdict(lambda: False)
    for class_iri, class_obj in onto.owl_classes.items():
        use_in_alignment = onto.get_owl_object_annotations(
            class_obj, "http://oaei.ontologymatching.org/bio-ml/ann/use_in_alignment"
        )
        if use_in_alignment and str(use_in_alignment[0]).lower() == "false":
            ignored_class_index[class_iri] = True
    return ignored_class_index


def remove_ignored_mappings(mappings: List[EntityMapping], ignored_class_index: dict):
    """Filter prediction mappings that involve classes to be ignored."""
    results = []
    for m in mappings:
        if ignored_class_index[m.head] or ignored_class_index[m.tail]:
            continue
        results.append(m)
    return results


################################################################
###                  for matching evaluation                 ###
################################################################


def matching_eval(
    pred_maps_file: str,
    ref_maps_file: str,
    null_ref_maps_file: Optional[str] = None,
    ignored_class_index: Optional[dict] = None,
    pred_maps_threshold: Optional[float] = None,
):
    r"""Conduct **global matching** evaluation for the prediction mappings against the
    reference mappings.

    The prediction mappings are formatted the same as `full.tsv` (the full reference mappings),
    with three columns: `"SrcEntity"`, `"TgtEntity"`, and `"Score"`, indicating the source
    class IRI, the target class IRI, and the corresponding mapping score.

    An `ignored_class_index` needs to be constructed for omitting prediction mappings
    that involve a class marked as **not used in alignment**.

    Use the following code to obtain such index for both the source and target ontologies:

    ```python
    ignored_class_index = get_ignored_class_index(src_onto)
    ignored_class_index.update(get_ignored_class_index(tgt_onto))
    ```
    """
    refs = ReferenceMapping.read_table_mappings(ref_maps_file, relation="=")
    preds = EntityMapping.read_table_mappings(pred_maps_file, relation="=", threshold=pred_maps_threshold)
    if ignored_class_index:
        preds = remove_ignored_mappings(preds, ignored_class_index)
    null_refs = ReferenceMapping.read_table_mappings(null_ref_maps_file, relation="=") if null_ref_maps_file else []
    results = AlignmentEvaluator.f1(preds, refs, null_reference_mappings=null_refs)
    return results


################################################################
###                  for ranking evaluation                  ###
################################################################


def read_candidate_mappings(cand_maps_file: str, has_score: bool = True):
    r"""Load scored or already ranked candidate mappings.

    The predicted candidate mappings are formatted the same as `test.cands.tsv`, with three columns:
    `"SrcEntity"`, `"TgtEntity"`, and `"TgtCandidates"`, indicating the source reference class IRI, the
    target reference class IRI, and a list of tuples in the form of `(target_candidate_class_IRI, score)`.
    The scores are optional if the candidate mappings have been ranked.

    This method loads the candidate mappings in this format and parse them into the inputs of [`mean_reciprocal_rank`][deeponto.align.evaluation.AlignmentEvaluator.mean_reciprocal_rank]
    and [`hits_at_K`][[`mean_reciprocal_rank`][deeponto.align.evaluation.AlignmentEvaluator.hits_at_K].
    """
    all_cand_maps = FileUtils.read_table(cand_maps_file).values.tolist()
    formatted = []
    for src_ref_class, tgt_ref_class, tgt_cands in all_cand_maps:
        ref_map = ReferenceMapping(src_ref_class, tgt_ref_class, "=")
        tgt_cands = eval(tgt_cands)
        if has_score:
            cand_maps = [EntityMapping(src_ref_class, t, "=", s) for t, s in tgt_cands]
        else:
            warnings.warn("Input candidate mappings do not have a score, assume default rank in descending order.")
            cand_maps = [
                EntityMapping(src_ref_class, t, "=", (len(tgt_cands) - i) / len(tgt_cands))
                for i, t in enumerate(tgt_cands)
            ]
        cand_maps = EntityMapping.sort_entity_mappings_by_score(cand_maps)
        formatted.append((ref_map, cand_maps))
    return formatted


def ranking_eval(cand_maps_file: str, has_score: bool = True, Ks=[1, 5, 10]):
    r"""Conduct **local ranking** evaluation for the scored or ranked candidate mappings.

    See [`read_candidate_mappings`][deeponto.align.oaei.read_candidate_mappings] for the file format and loading.
    """
    formatted_cand_maps = read_candidate_mappings(cand_maps_file, has_score)
    results = {"MRR": AlignmentEvaluator.mean_reciprocal_rank(formatted_cand_maps)}
    for K in Ks:
        results[f"Hits@{K}"] = AlignmentEvaluator.hits_at_K(formatted_cand_maps, K=K)
    return results
