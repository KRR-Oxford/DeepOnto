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
"""Class for ontology alignment pipeline"""

from deeponto.ontology import Ontology
from deeponto.ontology.mapping import *
from deeponto.ontology.onto_text import Tokenizer, text_utils
from itertools import cycle, product
from typing import List, Tuple, Optional


class OntoAlign:
    def __init__(
        self,
        src_onto: Ontology,
        tgt_onto: Ontology,
        tokenizer: Tokenizer,
        cand_pool_size: Optional[int] = 200,
        rel: str = "=",
        n_best: Optional[int] = 10,
    ):

        self.src_onto = src_onto
        self.tgt_onto = tgt_onto
        self.tokenizer = tokenizer
        self.cand_pool_size = cand_pool_size
        self.rel = rel
        self.set_mapping = lambda src_ent_name, tgt_ent_name, mapping_score: EntityMapping(
            src_ent_name, tgt_ent_name, self.rel, mapping_score
        )
        self.n_best = n_best

        self.src2tgt_mappings = EntityRankedMappings(
            flag="src2tgt", n_best=self.n_best, rel=self.rel
        )
        self.tgt2src_mappings = EntityRankedMappings(
            flag="tgt2src", n_best=self.n_best, rel=self.rel
        )
        self.flag_set = cycle(["src2tgt", "tgt2src"])
        self.flag = next(self.flag_set)

    def switch(self):
        """Switch alignment direction
        """
        self.src_onto, self.tgt_onto = self.tgt_onto, self.src_onto
        self.flag = next(self.flag_set)

    def current_mappings(self):
        return getattr(self, f"{self.flag}_mappings")

    def compute_mappings_for_ent(self, src_ent_id: int):
        """Compute cross-ontology mappings between source and target ontologies
        """
        raise NotImplementedError

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
