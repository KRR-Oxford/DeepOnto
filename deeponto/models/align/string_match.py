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
"""Class for string matching OM system"""

from typing import Optional, List, Set
from textdistance import levenshtein
from itertools import product

from deeponto.onto.text import Tokenizer
from deeponto.onto import Ontology
from . import OntoAlignBase


class StringMatch(OntoAlignBase):
    def __init__(
        self,
        src_onto: Ontology,
        tgt_onto: Ontology,
        tokenizer: Tokenizer,
        cand_pool_size: Optional[int] = 200,
        n_best: Optional[int] = 10,
        saved_path: str = "",
        use_edit_dist: bool = False,
    ):
        super().__init__(
            src_onto=src_onto,
            tgt_onto=tgt_onto,
            tokenizer=tokenizer,
            cand_pool_size=cand_pool_size,
            rel="=",
            n_best=n_best,
            is_trainable=False,
            is_val_model_select=True,
            default_hyperparams={"threshold": 0.9, "map_type": "src2tgt"},
            saved_path=saved_path,
        )
        self.use_edit_dist = use_edit_dist

    def ent_pair_score(self, src_ent_iri: str, tgt_ent_iri: str):
        """Compute mapping score between a cross-ontology entity pair
        """
        src_ent_labs = self.src_onto.iri2labs[src_ent_iri]
        tgt_ent_labs = self.tgt_onto.iri2labs[tgt_ent_iri]
        if not self.use_edit_dist:
            mapping_score = int(len(self.overlap(src_ent_labs, tgt_ent_labs)) > 0)
        else:
            mapping_score = self.max_norm_edit_sim(src_ent_labs, tgt_ent_labs)
        return mapping_score

    def fixed_src_ent_pair_score(self, src_ent_iri: str, tgt_cand_iris: List[str]):
        """Compute mapping scores between a source entity and a batch of target entities
        """
        mappings_for_ent = super().fixed_src_ent_pair_score(src_ent_iri, tgt_cand_iris)
        for tgt_cand_iri in tgt_cand_iris:
            score = self.ent_pair_score(src_ent_iri, tgt_cand_iri)
            mappings_for_ent.append(self.set_mapping(src_ent_iri, tgt_cand_iri, score))
        mappings_for_ent = mappings_for_ent.sorted()
        self.logger.info(f"[{self.flag}: {src_ent_iri}] {mappings_for_ent}\n")
        return mappings_for_ent

    def global_mappings_for_ent(self, src_ent_iri: str):
        """Compute cross-ontology mappings for a source entity
        """
        mappings_for_ent = super().global_mappings_for_ent(src_ent_iri)
        # select target candidates and compute score for each
        tgt_cands = self.idf_select_for_ent(src_ent_iri)
        for tgt_cand_iri, _ in tgt_cands:
            mapping_score = self.ent_pair_score(src_ent_iri, tgt_cand_iri)
            if mapping_score > 0:
                # save mappings only with positive mapping scores
                mappings_for_ent.append(self.set_mapping(src_ent_iri, tgt_cand_iri, mapping_score))
        # output only the top (k=n_best) scored mappings
        n_best_mappings_for_ent = mappings_for_ent.topk(self.n_best)
        self.logger.info(f"[{self.flag}: {src_ent_iri}] {n_best_mappings_for_ent}\n")
        return n_best_mappings_for_ent

    @staticmethod
    def overlap(src_ent_labs: List[str], tgt_ent_labs: List[str]) -> Set:
        # TODO: the overlapped percentage could be a factor of judgement
        return set(src_ent_labs).intersection(set(tgt_ent_labs))

    @classmethod
    def max_norm_edit_sim(cls, src_ent_labs: List[str], tgt_ent_labs: List[str]) -> float:
        # save time from the special case of overlapped labels
        if cls.overlap(src_ent_labs, tgt_ent_labs):
            return 1.0
        label_pairs = product(src_ent_labs, tgt_ent_labs)
        sim_scores = [levenshtein.normalized_similarity(src, tgt) for src, tgt in label_pairs]
        return max(sim_scores)
