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
"""Script for running implemented ontology matching models."""

import os
import sys

main_dir = os.getcwd().split("DeepOnto")[0] + "DeepOnto/src"
sys.path.append(main_dir)

import click

from deeponto import SavedObj
from deeponto.onto.mapping import OntoMappings
from deeponto.utils.logging import banner_msg
from deeponto.evaluation.eval_metrics import *
from deeponto.models.align import supported_modes
from deeponto.utils import print_choices


def global_match_eval(
    pred_path: str,
    ref_path: str,
    null_ref_path: Optional[str],
    threshold: float,
    consider_all_full_scored_mappings: bool = False,
):
    """Eval on Precision, Recall, and F-score (most general OM eval)
    """

    banner_msg("Eval using P, R, F-score")

    # load prediction mappings from the saved directory
    pred_maps = OntoMappings.from_saved(pred_path)
    pred = pred_maps.topKs(threshold, K=1)
    if consider_all_full_scored_mappings:
        full_score_pred = pred_maps.topKs(1.0, K=pred_maps.n_best)
        pred = list(set(pred + full_score_pred))

    if pred_maps.flag == "tgt2src":
        # reverse (head, tail) to match src2tgt
        pred = [(y, x) for (x, y) in pred]

    # load reference mappings and (opt) null mappings
    ref = OntoMappings.read_tsv_mappings(ref_path).to_tuples()
    null_ref = OntoMappings.read_tsv_mappings(null_ref_path).to_tuples() if null_ref_path else None

    results = f1(pred, ref, null_ref)
    SavedObj.print_json(results)

    return results


def pair_score_eval(pred_path: str, ref_path: str, *ks: int):
    """Eval on Hits@K, MRR (estimating OM performance) 
    """

    banner_msg("Eval using Hits@K, MRR")

    # load prediction mappings from the saved directory
    pred_maps = OntoMappings.from_saved(pred_path)

    # load reference mappings and (opt) null mappings
    ref = OntoMappings.read_tsv_mappings(ref_path, 0.0).to_tuples()

    results = dict()
    results["MRR"] = mean_reciprocal_rank(pred_maps, ref)
    for k in ks:
        results[f"Hits@{k}"] = hits_at_k(pred_maps, ref, k)
    SavedObj.print_json(results)

    return results


@click.command()
@click.option("-o", "--saved_path", type=click.Path(exists=True), default=".")
@click.option("-p", "--pred_path", type=click.Path(exists=True))
@click.option("-r", "--ref_path", type=click.Path(exists=True))
@click.option("-n", "--null_ref_path", type=click.Path(exists=True), default=None)
@click.option("-t", "--threshold", type=float, default=0.0)
@click.option("-c", "--consider_all_full_scored", type=bool, default=False)
@click.option("-k", "--hits_at", multiple=True, default=[1, 5, 10, 30, 100])
def main(
    saved_path: str,
    pred_path: str,
    ref_path: str,
    null_ref_path: Optional[str],
    threshold: float,
    consider_all_full_scored: bool,
    hits_at: List[int],
):

    banner_msg("Choose a Supported OM Mode")
    print_choices(supported_modes)
    mode = supported_modes[click.prompt("Enter a number", type=int)]

    if mode == "global_match":
        results = global_match_eval(
            pred_path, ref_path, null_ref_path, threshold, consider_all_full_scored
        )
    elif mode == "pair_score":
        results = pair_score_eval(pred_path, ref_path, *hits_at)
    else:
        raise ValueError(f"Unknown mode: {mode}, choices are: {supported_modes}.")
        
    SavedObj.save_json(results, saved_path + f"/{mode}.results.json")


if __name__ == "__main__":
    main()
