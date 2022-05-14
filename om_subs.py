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
"""Script for constructing inter-ontology subsumption mappings."""

import os
import sys

from torch import ge

main_dir = os.getcwd().split("DeepOnto")[0] + "DeepOnto/src"
sys.path.append(main_dir)

import click

from deeponto.onto import Ontology
from deeponto.data.align import SubsumptionMappingGenerator
from deeponto.utils.logging import banner_msg
from deeponto.utils.general_utils import print_choices


@click.command()
@click.option("-o", "--saved_path", type=click.Path(exists=True), default=".")
@click.option("-s", "--src_onto_path", type=click.Path(exists=True))
@click.option("-t", "--tgt_onto_path", type=click.Path(exists=True))
@click.option("-e", "--equiv_maps_path", type=click.Path(exists=True))
@click.option(
    "-r", "--subs_relation", type=click.Choice(["<", ">"])
)  # "<" means IS-A; ">" is the inverse of IS-A
@click.option("-m", "--max_subs_ratio", type=int, default=1)
@click.option("-d", "--is_delete_equiv_tgt", type=bool, default=True)
@click.option("-h", "--max_hop", type=int, default=1)
def subs_construct(
    saved_path: str,
    src_onto_path: str,
    tgt_onto_path: str,
    equiv_maps_path: str,
    subs_relation: str,
    max_subs_ratio: int,
    is_delete_equiv_tgt: bool,
    max_hop: int
):

    banner_msg("Choose a Generation Type")
    generation_types = ["static", "online"]
    print_choices(generation_types)
    gen_type = generation_types[click.prompt("Enter a number", type=int)]
    
    src_onto = Ontology.from_new(src_onto_path)
    tgt_onto = Ontology.from_new(tgt_onto_path)
    subs_generator = SubsumptionMappingGenerator(
        src_onto,
        tgt_onto,
        subs_relation,
        equiv_maps_path,
        max_subs_ratio,
        is_delete_equiv_tgt,
        max_hop,
    )

    if gen_type == "static":
        subs_generator.static_subs_construct()
    else:
        subs_generator.online_subs_construct()

    # conduct check
    for s in subs_generator.subs_pairs:
        assert subs_generator.validate_subs(s)

    # save the subsumption mappings
    banner_msg("Saving Results")
    print(f"Generate {len(subs_generator.subs_pairs)} subsumption ({subs_generator.rel}) mappings ...")
    print("Save mappings to .tsv file ...")
    subs_generator.subs_pairs_to_tsv(saved_path)

    if is_delete_equiv_tgt:
        print("Save preserved IRIs for target ontology (for pruning) ...")
        preserved_tgt_iris = subs_generator.preserved_tgt_iris()
        with open(saved_path + "/preserved_tgt_iris.txt", "w") as f:
            f.writelines([x + "\n" for x in preserved_tgt_iris])


if __name__ == "__main__":
    subs_construct()
