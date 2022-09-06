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
from deeponto.onto.mapping import OntoMappings, AnchoredOntoMappings
from deeponto.utils import print_choices, set_seed, read_table
from deeponto.utils.logging import banner_msg
from deeponto.models.align import (
    OntoAlignPipeline,
    implemented_models,
    supported_modes,
    multi_procs_models,
)
from deeponto.config import InputConfig


@click.command()
@click.option("-o", "--saved_path", type=click.Path(exists=True), default=".")
@click.option("-s", "--src_onto_path", type=click.Path(exists=True), default=None)
@click.option("-t", "--tgt_onto_path", type=click.Path(exists=True), default=None)
@click.option("-c", "--config_path", type=click.Path(exists=True), default=None)
@click.option("-m", "--to_be_scored_mappings_path", type=click.Path(exists=True), default=None)
@click.option("-f", "--to_be_scored_mappings_flag", type=str, default=None)
def onto_match(
    saved_path: str,
    config_path: str,
    src_onto_path: str,
    tgt_onto_path: str,
    to_be_scored_mappings_path: str,
    to_be_scored_mappings_flag: str,
):
    set_seed(42)

    banner_msg("Choose a Supported OM Mode")
    print_choices(supported_modes)
    mode = supported_modes[click.prompt("Enter a number", type=int)]

    banner_msg("Choose an Implemented OM Model")
    print_choices(implemented_models)
    model_name = implemented_models[click.prompt("Enter a number", type=int)]

    num_procs = None
    if model_name in multi_procs_models and mode == "global_match":
        # TODO: multi-procs can be extended to pairwise mapping computation
        use_multi_procs = click.confirm(
            f'"{model_name}" supports multi-processing for "{mode}", enable this feature?'
        )
        if use_multi_procs:
            num_procs = click.prompt("Enter the number of processes", type=int)

    if not config_path:
        align_configs_dir = os.getcwd().split("DeepOnto")[0] + "DeepOnto/config"
        banner_msg(f"Use Default Configs for {model_name}")
        config_path = align_configs_dir + f"/{model_name}.json"
    SavedObj.print_json(InputConfig.load_config(config_path))
    print()

    align_pipeline = OntoAlignPipeline(
        model_name, saved_path, config_path, src_onto_path, tgt_onto_path
    )

    if mode == "pair_score":
        # load the to-be-confirmed mappings during pair-score mode
        assert to_be_scored_mappings_path != None
        assert to_be_scored_mappings_path.endswith(".tsv") or to_be_scored_mappings_path.endswith(
            ".csv"
        )
        assert to_be_scored_mappings_flag in ["src2tgt", "tgt2src"]
        temp_read = read_table(to_be_scored_mappings_path)
        with_anchors = "TgtCandidates" in temp_read.columns
        if not with_anchors:
            print("Loading to-be-scored mappings from table data ...")
            tbc_onto_maps = OntoMappings.read_table_mappings(
                table_mappings_path=to_be_scored_mappings_path, flag=to_be_scored_mappings_flag
            )
        else:
            print("Loading to-be-scored mappings from table data with anchors ...")
            tbc_onto_maps = AnchoredOntoMappings.read_table_mappings(
                table_mappings_path=to_be_scored_mappings_path, flag=to_be_scored_mappings_flag
            ).unscored_cand_maps()
        align_pipeline.run(mode, tbc_onto_maps, to_be_scored_mappings_flag, num_procs=num_procs)
    else:
        align_pipeline.run(mode, num_procs=num_procs)


if __name__ == "__main__":
    onto_match()
