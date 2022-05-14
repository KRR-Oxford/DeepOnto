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

from deeponto.onto.prune import OntoPruner


@click.command()
@click.option("-o", "--saved_path", type=click.Path(exists=True), default=".")
@click.option("-s", "--src_onto_path", type=click.Path(exists=True))
@click.option("-i", "--preserved_iris_path", type=click.Path(exists=True))
@click.option("-t", "--preserve_strategy", type=str, default="simplest")
def onto_prune(
    saved_path: str, src_onto_path: str, preserved_iris_path: str, preserve_strategy: str
):
    pruner = OntoPruner(saved_path, src_onto_path, preserved_iris_path, preserve_strategy)
    pruner.run()

if __name__ == "__main__":
    onto_prune()