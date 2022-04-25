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
"""Script for evaluating implemented ontology matching models."""

import os
import sys

main_dir = os.getcwd().split("DeepOnto")[0] + "DeepOnto/src"
sys.path.append(main_dir)

import click

from deeponto import SavedObj
from deeponto.evaluation.align_eval import global_match_eval, local_rank_eval
from deeponto.utils.logging import banner_msg
from deeponto.evaluation.eval_metrics import *
from deeponto.models.align import eval_modes
from deeponto.utils import print_choices

@click.command()
@click.option("-o", "--saved_path", type=click.Path(exists=True), default=".")
@click.option("-p", "--pred_path", type=click.Path(exists=True))
@click.option("-r", "--ref_path", type=click.Path(exists=True))
@click.option("-n", "--null_ref_path", type=click.Path(exists=True), default=None)
@click.option("-a", "--ref_anchor_path", type=click.Path(exists=True), default=None)
@click.option("-t", "--threshold", type=float, default=0.0)
@click.option("-k", "--hits_at", multiple=True, default=[1, 5, 10, 30, 100])
@click.option("-s", "--show_more_f_scores", type=bool, default=False)
def main(
    saved_path: str,
    pred_path: str,
    ref_path: str,
    ref_anchor_path: Optional[str],
    null_ref_path: Optional[str],
    threshold: float,
    hits_at: List[int],
    show_more_f_scores: bool,
):

    banner_msg("Choose a Evaluation Mode")
    print_choices(eval_modes)
    mode = eval_modes[click.prompt("Enter a number", type=int)]

    if mode == "global_match":
        results = global_match_eval(
            pred_path, ref_path, null_ref_path, threshold, show_more_f_scores
        )
    elif mode == "local_rank":
        results = local_rank_eval(pred_path, ref_anchor_path, *hits_at)
    else:
        raise ValueError(f"Unknown mode: {mode}, choices are: {eval_modes}.")
        
    SavedObj.save_json(results, saved_path + f"/{mode}.results.json")


if __name__ == "__main__":
    main()
