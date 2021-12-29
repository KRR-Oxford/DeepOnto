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
"""Class for ontology alignment, requiring:

1. Mapping computation for a cross-ontology entity pair;
2. Full alignment computation w/o selection heuristic.

"""

from itertools import cycle
from typing import List, Tuple, Optional, Iterable
from multiprocessing_on_dill import Process, Manager
from pyats.datastructures import AttrDict
import numpy as np
import os

from deeponto.onto import Ontology
from deeponto.onto.mapping import *
from deeponto.onto.text import Tokenizer, text_utils
from deeponto.utils.logging import create_logger, banner_msg


class OntoAlign:
    def __init__(
        self,
        src_onto: Ontology,
        tgt_onto: Ontology,
        tokenizer: Tokenizer,
        cand_pool_size: Optional[int] = 200,
        rel: str = "≡",
        n_best: Optional[int] = 10,
        is_trainable: bool = False,
        saved_path: str = "",
    ):

        self.src_onto = src_onto
        self.tgt_onto = tgt_onto
        self.tokenizer = tokenizer
        self.cand_pool_size = cand_pool_size
        self.rel = rel
        self.saved_path = saved_path
        self.set_mapping = lambda src_ent_name, tgt_ent_name, mapping_score: EntityMapping(
            src_ent_name, tgt_ent_name, self.rel, mapping_score
        )
        self.new_mapping_list = lambda: EntityMappingList()
        self.logger = create_logger(f"{type(self).__name__}", saved_path=self.saved_path)
        self.n_best = n_best
        self.is_trainable = is_trainable

        self.src2tgt_mappings = OntoMappings(flag="src2tgt", n_best=self.n_best, rel=self.rel)
        self.tgt2src_mappings = OntoMappings(flag="tgt2src", n_best=self.n_best, rel=self.rel)
        self.flag_set = cycle(["src2tgt", "tgt2src"])
        self.flag = next(self.flag_set)

    ##################################################################################
    ###                        compute entity pair mappings                        ###
    ##################################################################################

    def pair_score(self, ent_name_pairs: List[Tuple[str, str]]):
        """Compute entity pair mappings by fixing source and target sides, respectively
        """
        self.renew()
        fixed_src_mappings = self.ent_pairs_mappings(ent_name_pairs)
        self.switch()
        fixed_tgt_mappings = self.ent_pairs_mappings(ent_name_pairs)
        return AttrDict({"fixed_src": fixed_src_mappings, "fixed_tgt": fixed_tgt_mappings,})

    def ent_pairs_mappings(self, ent_name_pairs: List[Tuple[str, str]]):
        """Compute mappings for intput src-tgt entity pairs
        """
        prefix = self.flag.split("2")[1]  # src or tgt
        # maximum number of mappings is the number of opposite ontology classes
        max_num_mappings = len(getattr(self, f"{prefix}_onto").idx2class)
        mappings = OntoMappings(flag=self.flag, n_best=max_num_mappings, rel=self.rel)
        for src_ent_name, tgt_ent_name in ent_name_pairs:
            src_ent_id = self.src_onto.idx2class[src_ent_name]
            tgt_ent_id = self.tgt_onto.idx2class[tgt_ent_name]
            score = self.ent_pair_score(src_ent_id, tgt_ent_id)
            mappings.add(EntityMapping(src_ent_name, tgt_ent_name, self.rel, score))
        return mappings

    def ent_pair_score(self, src_ent_id: str, tgt_ent_id: str) -> float:
        """Compute mapping score between a cross-ontology entity pair
        """
        raise NotImplementedError

    ##################################################################################
    ###                        compute global mappings                             ###
    ##################################################################################

    def global_match(self, num_procs: Optional[int] = None):
        """Compute alignment for both src2tgt and tgt2src
        """
        self.renew()
        self.global_mappings_for_onto_multi_procs(
            num_procs
        ) if num_procs else self.global_mappings_for_onto()
        self.switch()
        self.global_mappings_for_onto_multi_procs(
            num_procs
        ) if num_procs else self.global_mappings_for_onto()

    def renew(self):
        """Renew alignment direction to src2tgt
        """
        while self.flag != "src2tgt":
            self.switch()

    def switch(self):
        """Switch alignment direction
        """
        self.src_onto, self.tgt_onto = self.tgt_onto, self.src_onto
        self.flag = next(self.flag_set)

    def current_global_mappings(self):
        return getattr(self, f"{self.flag}_mappings")

    def global_mappings_for_onto_multi_procs(self, num_procs: int):
        """Compute mappings for all entities in the current source ontology but distributed
        to multiple processes
        """
        # manager for collecting mappings from different procs
        manager = Manager()
        return_dict = manager.dict()
        # suggested by huggingface when doing multi-threading
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        def async_compute(proc_idx: int, return_dict: dict, src_ent_id_chunk: Iterable[int]):
            return_dict[proc_idx] = self.global_mappings_for_ent_chunk(src_ent_id_chunk)

        self.logger.info(
            f'Compute "{self.rel}" Mappings: {self.src_onto.owl.name} ==> {self.tgt_onto.owl.name}\n'
        )

        # split entity ids into {num_procs} chunks
        src_ent_id_chunks = np.array_split(list(self.src_onto.idx2class.keys()), num_procs)

        # start proc for each chunk
        jobs = []
        for i in range(num_procs):
            p = Process(target=async_compute, args=(i, return_dict, src_ent_id_chunks[i]))
            jobs.append(p)
            p.start()

        # block the main thread until all procs finished
        for p in jobs:
            p.join()

        # save the output mappings
        mappings = self.current_global_mappings()
        for ent_mappings in return_dict.values():
            mappings.add_many(*ent_mappings)
        banner_msg("Task Finished")
        mappings.save_instance(f"{self.saved_path}/global_match/{self.flag}")

    def global_mappings_for_onto(self):
        """Compute mappings for all entities in the current source ontology
        """
        self.logger.info(
            f'Compute "{self.rel}" Mappings: {self.src_onto.owl.name} ==> {self.tgt_onto.owl.name}\n'
        )
        # save the output mappings
        mappings = self.current_global_mappings()
        mappings.add_many(*self.global_mappings_for_ent_chunk(self.src_onto.idx2class.keys()))
        banner_msg("Task Finished")
        mappings.save_instance(f"{self.saved_path}/global_match/{self.flag}")

    def global_mappings_for_ent_chunk(self, src_ent_id_chunk: Iterable[int]):
        """Compute cross-ontology mappings for a chunk of source entities,
        Note: save time especially for evaluating on Hits@K, MRR, etc.
        """
        mappings_for_chunk = []
        for src_ent_id in src_ent_id_chunk:
            mappings_for_chunk += self.global_mappings_for_ent(src_ent_id)
        return mappings_for_chunk

    def global_mappings_for_ent(self, src_ent_id: int) -> EntityMappingList:
        """Compute cross-ontology mappings for a source entity
        """
        banner_msg(f"Compute Mappings for Entity {src_ent_id} ({self.flag})")
        mappings_for_ent = self.new_mapping_list()
        # TODO: followed by individual implementations
        return mappings_for_ent

    def idf_select_for_ent(self, src_ent_id: int) -> Tuple[str, float]:
        """Select candidates in target ontology for a given source entity
        """
        src_ent_labs = self.src_onto.idx2labs[src_ent_id]
        src_ent_toks = self.tokenizer.tokenize_all(src_ent_labs)
        # TODO: could have more candidate selection methods in future
        tgt_cands = self.tgt_onto.idf_select(
            src_ent_toks, self.cand_pool_size
        )  # [(ent_id, idf_score)]
        return tgt_cands

    ##################################################################################
    ###                        other auxiliary functions                           ###
    ##################################################################################

    def lab_products_for_ent(self, src_ent_id: int) -> Tuple[List[str], List[str], List[int]]:
        """Compute Catesian Product between a source entity's labels and its selected 
        target entities' labels, with each block length recorded
        """
        src_sents, tgt_sents = [], []
        product_lens = []
        src_ent_labs = self.src_onto.idx2labs[src_ent_id]
        tgt_cands = self.idf_select_for_ent(src_ent_id)
        for tgt_cand_id, _ in tgt_cands:
            tgt_ent_labs = self.tgt_onto.idx2labs[tgt_cand_id]
            src_out, tgt_out = text_utils.lab_product(src_ent_labs, tgt_ent_labs)
            assert len(src_out) == len(tgt_out)
            product_lens.append(len(src_out))
            src_sents += src_out
            tgt_sents += tgt_out
        return src_sents, tgt_sents, product_lens
