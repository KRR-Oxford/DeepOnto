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
"""Define evaluation metrics for different tasks"""

from typing import Optional, Iterable, Tuple, List

from deeponto.onto.mapping import AnchoredOntoMappings

##################################################################################
###                       [Eval Case 1]: P, R, F1                              ###
##################################################################################


def precision(pred: Iterable, ref: Iterable) -> float:
    """% of predictions are correct:
        P = TP / (TP + FP)
    """
    # True Positive = the number of (unignored) prediction mappings that are True
    tp = len(set(pred).intersection(set(ref)))
    # False Positive = the number of (unignored) prediction mappings that are False
    fp = len(set(pred)) - tp
    return tp / (tp + fp)


def recall(pred: Iterable, ref: Iterable) -> float:
    """% of correct retrieved:
        R = TP / (TP + FN)
    """
    # True Positive = the number of (unignored) reference mappings that are Positive
    tp = len(set(pred).intersection(set(ref)))
    # False Negative = the number of (unignored) reference mappings that are Negative
    fn = len(set(ref)) - tp
    return tp / (tp + fn)


def f_score(pred: Iterable, ref: Iterable, beta: float, null_ref: Optional[Iterable] = None):
    """Semantics-aware F-score with factor β where elements in the {null_set} should be ignored, 
    i.e.: neither +ve nor -ve. 
    
    Check concepts at: https://lawhy.github.io//pythagorean-means/.
    """
    # elements in the {null_set} are removed from both {pred} and {ref} (ignored)
    if null_ref:
        pred = set(pred) - set(null_ref)
        ref = set(ref) - set(null_ref)
    P = precision(pred, ref)
    R = recall(pred, ref)
    beta_sqr = beta ** 2
    F_beta = ((1 + beta_sqr) * P * R) / (beta_sqr * P + R)
    return {"P": round(P, 3), "R": round(R, 3), "f_score": round(F_beta, 3)}


def f1(pred: Iterable, ref: Iterable, null_ref: Optional[Iterable] = None):
    """Semantics-aware F1 score when precision and recall are equally important, i.e.:
        F1 = 2 * P * R / (P + R).
    """
    return f_score(pred, ref, 1.0, null_ref)


##################################################################################
###                       [Eval Case 2]: Hits@K & MRR                          ###
##################################################################################

# TODO: check below algorithms after full deployment


def hits_at_k(pred_maps: AnchoredOntoMappings, ref_tuples: List[Tuple], k: int):
    """Hits@K = # hits at top K / # testing samples 
    """
    n_hits = 0
    for src_ent, tgt_ent in ref_tuples:
        tgt2score = pred_maps.anchor2cands[src_ent, tgt_ent]
        topk_tgt_cands = list(tgt2score.keys())[:k]
        if tgt_ent in topk_tgt_cands:
            n_hits += 1
    return n_hits / len(ref_tuples)


def mean_reciprocal_rank(pred_maps: AnchoredOntoMappings, ref_tuples: List[Tuple]):
    """MRR = (\sum_i 1 / rank_i) / # testing samples
    """
    sum_inv_ranks = 0
    for src_ent, tgt_ent in ref_tuples:
        tgt2score = pred_maps.anchor2cands[src_ent, tgt_ent]
        tgt_cands = list(tgt2score.keys())
        rank = tgt_cands.index(tgt_ent) + 1
        sum_inv_ranks += 1 / rank
    return sum_inv_ranks / len(ref_tuples)
