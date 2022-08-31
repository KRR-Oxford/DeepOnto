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
"""Mapping extension algorithm for ontology alignment based on locality principle,
which states: 

    Pr(Matched(A', B') | Matched(A, B)) is high for 
    SemanticallyRelated(A, A') and SemanticallyRelated(B, B') 

Compared to the original version which adds the mappings into the predicted set;
the new implementation is more like a `modification` in the sense that the selected
prediciton mappings are `modified` by the extended mappings because some of the
old mappings are replaced as new top1 outputs -- this also preservse the 1-1 property
of output mappings. 
"""
from __future__ import annotations

from typing import Optional, Callable, TYPE_CHECKING
from itertools import product
import random

# to avoid circular imports
if TYPE_CHECKING:
    from deeponto.onto import Ontology
    from logging import Logger

from deeponto.onto.mapping import OntoMappings, EntityMapping, EntityMappingList
from .graph_utils import *


class IterativeMappingExtension:
    def __init__(
        self,
        src_onto: Ontology,
        tgt_onto: Ontology,
        onto_mappings: OntoMappings,  # can be src2tgt of tgt2src
        scoring_method: Callable[[int, int], float],
        threshold: float = 0.9,  # \kappa as in BERTMap paper
        logger: Optional[Logger] = None,
    ):
        self.src_onto = src_onto
        self.tgt_onto = tgt_onto
        self.frontier = []
        self.onto_mappings = onto_mappings
        self.ent_pair_score = scoring_method
        self.threshold = threshold
        self.log = logger.info if logger else print
        self.num_iter = 0
        self.num_expanded = 0

    def run_extension(self, max_iter: int = 10):
        self.num_iter = 0
        self.num_expanded = 0
        self.log(
            f"Run iterative mapping extension algorithm for {self.onto_mappings.flag} mappings."
        )
        # initialize the frontier as mappings with score >= kappa (extension threshold)
        self.renew_frontier(self.onto_mappings)
        while self.frontier and self.num_iter < max_iter:
            new_mappings = []
            for src_ent_name, tgt_ent_name in self.frontier:
                new_mappings += self.one_hob_extend(src_ent_name, tgt_ent_name)
            self.num_iter += 1
            # add the new mappings to onto_mappings
            self.onto_mappings.add_many(*new_mappings)
            # renew the frontier by the newly expanded mapping set
            self.renew_frontier(
                OntoMappings(
                    self.onto_mappings.flag,
                    self.onto_mappings.n_best,
                    self.onto_mappings.rel,
                    self.onto_mappings.dup_strategy,
                    *new_mappings,
                )
            )
            self.log(f"Add {len(new_mappings)} from iteration {self.num_iter} ...")
        self.log(f"[ExpansionStats]: total={self.num_expanded}; num_iter={self.num_iter}")

    def renew_frontier(self, given_mappings: OntoMappings):
        """Renew frontier with original anchors or newly expanded mappings
        """
        self.frontier = given_mappings.topks(
            K=given_mappings.n_best, threshold=self.threshold, as_tuples=True
        )
        if given_mappings.flag == "tgt2src":
            # reverse (head, tail) to match src2tgt
            self.frontier = [(y, x) for (x, y) in self.frontier]

    def check_ent_pair(self, src_ent_iri: str, tgt_ent_iri: str):
        """Check if a pair of entities should be added or discarded (explored or undervalued);
        if should be added, return the corresponding Mapping
        """
        # score doesn't matter in check_existed
        temp_map = (
            EntityMapping(src_ent_iri, tgt_ent_iri, self.onto_mappings.rel, -1.0)
            if self.onto_mappings.flag == "src2tgt"
            else EntityMapping(tgt_ent_iri, src_ent_iri, self.onto_mappings.rel, -1.0)
        )
        if self.onto_mappings.is_existed_mapping(temp_map):
            return "explored"
        # if not explored before we compute the score
        try:
            score = self.ent_pair_score(src_ent_iri, tgt_ent_iri)
            if score < self.threshold:
                return "undervalued"
            else:
                temp_map.score = score
                return temp_map
        except:
            return "undervalued"

    def one_hob_extend(self, src_ent_iri: str, tgt_ent_iri: str, maximum_pairs: int = 500):
        """1-hop mapping extension, the assumption is given a highly confident mapping,
        the corresponding classes' parents and children are likely to be matched.
        """
        # get back the entity class by IRI
        src_ent = self.src_onto.owl.search(iri=src_ent_iri)[0]
        tgt_ent = self.tgt_onto.owl.search(iri=tgt_ent_iri)[0]

        cand_pairs = list(
            product(super_thing_classes_of(src_ent), super_thing_classes_of(tgt_ent))
        )  # parent pairs
        cand_pairs += list(
            product(sub_thing_classes_of(src_ent), sub_thing_classes_of(tgt_ent))
        )  # children pairs
        print(f"detect {len(cand_pairs)} pairs for extension ...")
        if len(cand_pairs) > maximum_pairs:
            temp = len(cand_pairs)
            cand_pairs = random.sample(cand_pairs, maximum_pairs)
            print(f"sample {len(cand_pairs)}/{temp} pairs for extension ...")
        num_explored = 0
        num_undervalued = 0
        num_added = 0
        new_mappings = EntityMappingList()
        for src_cand, tgt_cand in cand_pairs:
            result = self.check_ent_pair(src_cand.iri, tgt_cand.iri)
            if result == "explored":
                num_explored += 1
            elif result == "undervalued":
                num_undervalued += 1
            else:
                num_added += 1
                new_mappings.append(result)
        self.num_expanded += num_added
        self.log(
            f"[Iter {self.num_iter}][Anchor]: {src_ent_iri} {self.onto_mappings.rel} {tgt_ent_iri}\n"
            + f"\t[ExpansionStats]: total={self.num_expanded}; added={num_added}; seen={num_explored}; undervalued={num_undervalued}\n"
            + f"\t[AddedMappings]: {str(new_mappings)}"
        )
        return new_mappings
