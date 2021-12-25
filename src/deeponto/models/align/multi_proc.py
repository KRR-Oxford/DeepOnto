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
"""Class for ontology alignment that uses multiple processes"""

import os
import sys
main_dir = os.getcwd().split("DeepOnto")[0] + "DeepOnto/src"
sys.path.append(main_dir)

from multiprocessing import Process
from deeponto.models.align import *
from deeponto.onto import  Ontology
from deeponto.onto.mapping import  EntityMappingList
from deeponto.onto.onto_text import Tokenizer, text_utils
import time

def run_multi_procs_on_align(onto_align: OntoAlign, num_procs: int):
    pass


def log_info(output_ent_mappings: EntityMappingList):
    pass

if __name__ == '__main__':
    tkz = Tokenizer.from_pretrained(text_utils.BIOCLINICAL_BERT)
    # tkz2 = Tokenizer.from_rule_based()
    ontos = []
    for name in ["fma2nci", "nci2fma", "fma2snomed", "snomed2fma", "snomed+2fma"][:2]:
        onto_file = main_dir + f"/../data/LargeBio/ontos/{name}.small.owl"
        onto = Ontology.from_new(onto_file, tokenizer=tkz)
        ontos.append(onto)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    align = EditSimilarity(src_onto=ontos[0], tgt_onto=ontos[1], tokenizer=tkz, rel="≡", saved_path="./testing/align")
    # align.compute_mappings_all()
    align.run(num_procs=3)
    # procs = []
    # for i in [10, 12]:
    #     p = Process(target=align.compute_mappings_for_ent, args=(i, ))
    #     p.start()
    #     p.join()
    #     procs.append(p)
    # print(align.src2tgt_mappings.ranked)
    # print(align.src2tgt_mappings.tops())