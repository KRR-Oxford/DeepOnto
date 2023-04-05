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

import os
import sys

main_dir = os.getcwd().split("DeepOnto")[0] + "DeepOnto/src"
sys.path.append(main_dir)

from deeponto.onto import Ontology
from deeponto.align.bertmap import BERTMapPipeline, DEFAULT_CONFIG_FILE
import click

@click.command()
@click.option("-s", "--src_onto_file", type=click.Path(exists=True))
@click.option("-t", "--tgt_onto_file", type=click.Path(exists=True))
@click.option("-c", "--config_file", type=click.Path(exists=True))
@click.option("-r", "--resume_training", type=bool, default=False)
def run_bertmap(src_onto_file, tgt_onto_file, config_file, resume_training):
    
    config = BERTMapPipeline.load_bertmap_config(config_file)
    # enable automatic global matching and subsequent mapping refinement
    config.global_matching.enabled = True
    # None for both False and None
    config.bert.resume_training = None if not resume_training else resume_training
    src_onto = Ontology(src_onto_file)
    tgt_onto = Ontology(tgt_onto_file)

    BERTMapPipeline(src_onto, tgt_onto, config)


if __name__ == "__main__":
    run_bertmap()
