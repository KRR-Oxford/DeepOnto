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
"""Script for downloading and extracting mappings from Mondo 
        (https://mondo.monarchinitiative.org/)
        
    Feb 18: this script is not useful now because MONDO team has provided mappings for the users
"""

import os
import sys

main_dir = os.getcwd().split("DeepOnto")[0] + "DeepOnto/src"
sys.path.append(main_dir)

import click

from deeponto.data.data_utils import *
from deeponto.onto.mapping import *


@click.command()
@click.option("-o", "--saved_path", type=click.Path())
def mondo(saved_path: str):
    # download the mondo ontology if not existed
    download_mondo(saved_path)
    # extract the mappings
    for map_prop, rel in [
        #("equivalent_to", "="),  # owl:equivalentTo
        ("exactMatch", "="),  # sko:exactMatch
        #("closeMatch", "="),  # sko:closeMatch
        #("narrowMatch", ">"),  # sko:narrowMatch
        #("broadMatch", "<"),  # sko:broadMatch
    ]:
        mapping_dir = f"{saved_path}/{map_prop}"
        if detect_path(mapping_dir):
            print(
                f"{map_prop} mappings have been created; "
                + "if you want to override the existing mappings,"
                + "delete the directory and re-run."
            )
            mappings = OntoMappings.from_saved(f"{saved_path}/{map_prop}")
        else:
            mappings = extract_mondo_mappings(saved_path, map_prop, rel=rel)
            mappings.save_instance(f"{saved_path}/{map_prop}")
    # split the mappings for pairwise ontologies


if __name__ == "__main__":
    mondo()
