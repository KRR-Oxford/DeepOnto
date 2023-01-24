# Copyright 2021 Yuan He (KRR-Oxford). All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Iterable, Tuple, List
import math
from .mapping import *


class AlignmentEvaluator:
    """Class that provides evaluation metrics for alignment."""
    
    def __init__(self):
        pass

    @staticmethod
    def precision(prediction_mappings: List[EntityMapping], reference_mappings: Iterable[ReferenceMapping]) -> float:
        r"""The percentage of correct predictions.

        $$P = \frac{|\mathcal{M}_{pred} \cap \mathcal{M}_{ref}|}{|\mathcal{M}_{pred}|}$$
        """
        preds = [p.to_tuple() for p in prediction_mappings]
        refs = [r.to_tuple() for r in reference_mappings]
        return len(set(preds).intersection(set(refs))) / len(set(preds))

    @staticmethod
    def recall(prediction_mappings: List[EntityMapping], reference_mappings: Iterable[ReferenceMapping]) -> float:
        r"""The percentage of correct retrievals.

        $$R = \frac{|\mathcal{M}_{pred} \cap \mathcal{M}_{ref}|}{|\mathcal{M}_{ref}|}$$
        """
        preds = [p.to_tuple() for p in prediction_mappings]
        refs = [r.to_tuple() for r in reference_mappings]
        return len(set(preds).intersection(set(refs))) / len(set(refs))

    @staticmethod
    def f1(
        prediction_mappings: List[EntityMapping],
        reference_mappings: Iterable[ReferenceMapping],
        null_reference_mappings: Optional[Iterable] = None,
    ):
        r"""Compute the F1 score given the prediction and reference mappings.

        $$F_1 = \frac{2 P R}{P + R}$$

        `null_reference_mappings` is an additional set whose elements
        should be **ignored** in the calculation, i.e., **neither positive nor negative**.
        Specifically, both $\mathcal{M}_{pred}$ and $\mathcal{M}_{ref}$ will **substract**
        $\mathcal{M}_{null}$ from them.
        """
        preds = [p.to_tuple() for p in prediction_mappings]
        refs = [r.to_tuple() for r in reference_mappings]
        null_refs = [n.to_tuple() for n in null_reference_mappings]
        # elements in the {null_set} are removed from both {pred} and {ref} (ignored)
        if null_refs:
            preds = set(preds) - set(null_refs)
            refs = set(refs) - set(null_refs)
        P = len(set(preds).intersection(set(refs))) / len(set(preds))
        R = len(set(preds).intersection(set(refs))) / len(set(refs))
        F1 = 2 * P * R / (P + R)

        return {"P": round(P, 3), "R": round(R, 3), "F1": round(F1, 3)}

    ##################################################################################
    ###                       [Eval Case 2]: Hits@K & MRR                          ###
    ##################################################################################

    # TODO: check below algorithms after full deployment

    @staticmethod
    def hits_at_K(prediction_and_candidates: List[Tuple[EntityMapping, List[EntityMapping]]], K: int):
        r"""Compute $Hits@K$ for a list of `(prediction_mapping, candidate_mappings)` pair.

        It is computed as the number of a `prediction_mapping` existed in the first $K$ ranked `candidate_mappings`,
        divided by the total number of input pairs.

        $$Hits@K = \sum_i^N \mathbb{I}_{rank_i \leq k} / N$$
        """
        n_hits = 0
        for pred, cands in prediction_and_candidates:
            ordered_candidates = [c.to_tuple() for c in EntityMapping.sort_entity_mappings_by_score(cands, k=K)]
            if pred.to_tuple() in ordered_candidates:
                n_hits += 1
        return n_hits / len(prediction_and_candidates)

    @staticmethod
    def mean_reciprocal_rank(prediction_and_candidates: List[Tuple[EntityMapping, List[EntityMapping]]]):
        r"""Compute $MRR$ for a list of `(prediction_mapping, candidate_mappings)` pair.

        $$MRR = \sum_i^N rank_i^{-1} / N$$
        """
        sum_inverted_ranks = 0
        for pred, cands in prediction_and_candidates:
            ordered_candidates = [c.to_tuple() for c in EntityMapping.sort_entity_mappings_by_score(cands)]
            if pred.to_tuple() in ordered_candidates:
                rank = ordered_candidates.index(pred) + 1
            else:
                rank = math.inf
            sum_inverted_ranks += 1 / rank
        return sum_inverted_ranks / len(prediction_and_candidates)
