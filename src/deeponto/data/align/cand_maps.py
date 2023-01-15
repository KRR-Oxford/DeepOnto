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
"""Provide functions for generating negative candidates"""

from typing import Optional
import random

from deeponto.onto import Ontology
from deeponto.onto.graph.graph_utils import *
from deeponto.onto.text import Tokenizer
from deeponto.onto.mapping import OntoMappings, EntityMapping, AnchoredOntoMappings, AnchorMapping
from deeponto.utils import uniqify
from deeponto.utils.logging import banner_msg
from deeponto import SavedObj, FlaggedObj


# TODO: to be updated constantly
sampling_options = ["idf", "neighbour", "random"]


class MappingCandidateGenerator(FlaggedObj):
    def __init__(
        self,
        saved_path: str,
        src_onto: Ontology,
        tgt_onto: Ontology,
        ref_mappings_path: str,
        null_ref_mappings_path: Optional[str],
        rel: str,
        tokenizer: Tokenizer,
        max_hops: int = 5,
        avoid_ancestors: bool = False,
        avoid_descendents: bool = False,
    ):

        super().__init__(src_onto, tgt_onto)
        self.rel = rel
        self.saved_path = saved_path

        # loaded mappings are served as +ve candiates (anchors)
        # note that dict cannot be used here because multiple mappings are possible
        self.pos_src2tgt = OntoMappings.read_table_mappings(
            table_mappings_path=ref_mappings_path, rel=self.rel
        ).to_tuples()
        self.pos_tgt2src = [(y, x) for (x, y) in self.pos_src2tgt]
        self.null_src2tgt = []
        self.null_tgt2src = []
        if null_ref_mappings_path:
            self.null_src2tgt = OntoMappings.read_table_mappings(
                table_mappings_path=null_ref_mappings_path, rel=self.rel
            ).to_tuples()
            self.null_tgt2src = [(y, x) for (x, y) in self.null_src2tgt]
        self.pos_src2tgt = list(set(self.pos_src2tgt) - set(self.null_src2tgt))
        self.pos_tgt2src = list(set(self.pos_tgt2src) - set(self.null_tgt2src))

        self.tokenizer = tokenizer

        # init the reference anchored mappings
        self.src2tgt_anchored_mappings = AnchoredOntoMappings(
            flag="src2tgt", n_best=None, rel=self.rel
        )
        self.src2tgt_ref_dict = defaultdict(list)
        self.init_anchors()
        self.switch()
        self.tgt2src_anchored_mappings = AnchoredOntoMappings(
            flag="tgt2src", n_best=None, rel=self.rel
        )
        self.tgt2src_ref_dict = defaultdict(list)
        self.init_anchors()
        self.renew()

        self.stats = dict()

        # arguments for neighbour sampling
        self.max_hops = max_hops
        self.avoid_ancestors = avoid_ancestors
        self.avoid_descendents = avoid_descendents

    def init_anchors(self):
        """Add all the reference anchor mappings as a candidate mapping first
        """
        # init anchor maps and anchor dict
        anchored_maps = self.current_anchor_mappings()
        ref_dict = self.current_ref_dict()
        for ref_src_ent_iri, ref_tgt_ent_iri in getattr(self, f"pos_{self.flag}"):
            anchor_map = AnchorMapping(ref_src_ent_iri, ref_tgt_ent_iri, self.rel, 0.0)
            anchor_map.add_candidate(anchor_map)
            anchored_maps.add(anchor_map)
            ref_dict[ref_src_ent_iri].append(ref_tgt_ent_iri)
        for null_ref_src_iri, null_ref_tgt_ent_iri in getattr(self, f"null_{self.flag}"):
            ref_dict[null_ref_src_iri].append(null_ref_tgt_ent_iri)

    def sample(self, **strategy2nums):
        """Sample for both src2tgt and tgt2src sides
        """
        self.renew()
        self.sample_for_all_one_side(**strategy2nums)
        self.switch()
        self.sample_for_all_one_side(**strategy2nums)
        self.renew()

    def sample_for_all_one_side(self, **strategy2nums):
        """Sample negative candidates as per specified strategy-number pairs
        """
        for ref_src_ent_iri, ref_tgt_ent_iri in getattr(self, f"pos_{self.flag}"):
            banner_msg(
                f"Generate Candidates for({self.src_onto.name_from_iri(ref_src_ent_iri)}," + 
                    f"{self.tgt_onto.name_from_iri(ref_tgt_ent_iri)})"
            )
            anchor_map = AnchorMapping(ref_src_ent_iri, ref_tgt_ent_iri, self.rel, 0.0)
            # excluding all the reference target classes for the same reference source class
            # as negative candidates
            excluded_ref_tgts = self.current_ref_dict()[ref_src_ent_iri]
            if len(excluded_ref_tgts) > 1:
                print("One-to-many reference mappings detected ...")
            tgt_cand_iris, sample_stats = self.mixed_sample(
                ref_tgt_ent_iri, excluded_ref_tgts, **strategy2nums
            )
            self.stats[str(anchor_map.to_tuple())] = sample_stats
            for tgt_cand_iri in tgt_cand_iris:
                cand_map = EntityMapping(ref_src_ent_iri, tgt_cand_iri, self.rel, 0.0)
                anchor_map.add_candidate(cand_map)
            self.current_anchor_mappings().add(anchor_map, allow_existed=False)
            print("Candidate mappings statistics:")
            SavedObj.print_json(sample_stats)

        # candidate mappings are never the null reference mappings
        assert not set(getattr(self, f"null_{self.flag}")).intersection(
            set(self.current_anchor_mappings().cand2anchors.keys())
        )
        assert set(getattr(self, f"pos_{self.flag}")).intersection(
            set(self.current_anchor_mappings().cand2anchors.keys())
        ) == set(getattr(self, f"pos_{self.flag}"))

        self.current_anchor_mappings().save_instance(f"{self.saved_path}/{self.flag}.rank")
        # self.current_anchor_mappings().unscored_cand_maps().save_instance(
        #     f"./{self.flag}.rank/for_score"
        # )
        SavedObj.save_json(self.stats, f"{self.saved_path}/{self.flag}.rank/stats.json")

    def current_anchor_mappings(self) -> AnchoredOntoMappings:
        return getattr(self, f"{self.flag}_anchored_mappings")

    def current_ref_dict(self):
        return getattr(self, f"{self.flag}_ref_dict")

    ##################################################################################
    ###                            sampling stratgies                              ###
    ##################################################################################

    def mixed_sample(self, ref_tgt_ent_iri, excluded_ref_tgts, **strategy2nums):
        """Randomly select specified candidates for each strategy;
        amend random samples if not meeting the number
        """
        all_tgt_cand_iris = []
        stats = defaultdict(lambda: 0)
        i = 0
        total_n_cands = 0
        for strategy, n_cands in strategy2nums.items():
            i += 1
            if strategy in sampling_options:
                sampler = getattr(self, f"{strategy}_sample")
                # for ith iteration, the worst case is when all n_cands are duplicated
                # or should be excluded from other reference targets so we generate
                # NOTE:  total_n_cands + n_cands + len(excluded_ref_tgts)
                # candidates first and prune the rest; another edge case is when sampled
                # candidates are not sufficient and we use random sample to meet n_cands
                cur_tgt_cands = sampler(
                    ref_tgt_ent_iri, total_n_cands + n_cands + len(excluded_ref_tgts)
                )
                # remove the duplicated candidates (and excluded refs) and prune the tail
                cur_tgt_cands = list(
                    set(cur_tgt_cands) - set(all_tgt_cand_iris) - set(excluded_ref_tgts)
                )[:n_cands]
                stats[strategy] += len(cur_tgt_cands)
                # use random samples for complementation if not enough
                while len(cur_tgt_cands) < n_cands:
                    amend_cands = self.random_sample(ref_tgt_ent_iri, n_cands - len(cur_tgt_cands))
                    amend_cands = list(
                        set(amend_cands)
                        - set(all_tgt_cand_iris)
                        - set(cur_tgt_cands)
                        - set(excluded_ref_tgts)
                    )
                    cur_tgt_cands += amend_cands
                assert len(cur_tgt_cands) == n_cands
                if strategy != "random":
                    stats["random"] += n_cands - stats[strategy]
                all_tgt_cand_iris += cur_tgt_cands
                total_n_cands += n_cands
            else:
                raise ValueError(f"Invalid sampling trategy: {strategy}.")
        assert len(all_tgt_cand_iris) == total_n_cands
        return all_tgt_cand_iris, stats

    def random_sample(self, ref_tgt_ent_iri, n_cands: int):
        """Randomly select n target candidates
        """
        ref_tgt_ent = self.tgt_onto.owl.search(iri=ref_tgt_ent_iri)[0]
        # for subsumption mapping, ancestors or descendants will be avoided
        avoided_family = self.avoided_family_of(ref_tgt_ent)
        avoided_ent_iris = [m.iri for m in avoided_family]

        all_tgt_ent_iris = list(self.tgt_onto.iri2labs.keys())
        all_tgt_ent_iris = list(set(all_tgt_ent_iris) - set(avoided_ent_iris))

        assert not ref_tgt_ent_iri in all_tgt_ent_iris
        return random.sample(all_tgt_ent_iris, n_cands)

    def idf_sample(self, ref_tgt_ent_iri: str, n_cands: int):
        """Sample negative candidates using inverted index-based idf scoring
        """
        ref_tgt_ent = self.tgt_onto.owl.search(iri=ref_tgt_ent_iri)[0]
        # for subsumption mapping, ancestors or descendants will be avoided
        avoided_family = self.avoided_family_of(ref_tgt_ent)
        avoided_ent_iris = [m.iri for m in avoided_family]

        # select n candidates for the target entity
        ref_tgt_ent_labs = self.tgt_onto.iri2labs[ref_tgt_ent_iri]
        ref_tgt_ent_toks = self.tokenizer.tokenize_all(ref_tgt_ent_labs)
        # sample 1 more candidate to prevent when reference target is included
        tgt_cands = self.tgt_onto.idf_select(
            ref_tgt_ent_toks, n_cands + len(avoided_family)
        )  # [(ent_id, idf_score)]

        # unzip to list of ids
        tgt_cand_iris = list(list(zip(*tgt_cands))[0])
        # ref-tgt is considered to be avoided as well
        tgt_cand_iris = list(set(tgt_cand_iris) - set(avoided_ent_iris))

        assert not ref_tgt_ent_iri in tgt_cand_iris
        return tgt_cand_iris[:n_cands]

    def neighbour_sample(self, ref_tgt_ent_iri, n_cands: int):
        """Sample negative candidates from nearest nearest (starting from 1-hop)
        """

        # select n candidates for the target entity
        ref_tgt_ent = self.tgt_onto.obj_from_iri(ref_tgt_ent_iri)

        # for subsumption mapping, ancestors or descendants will be avoided
        avoided_family = self.avoided_family_of(ref_tgt_ent)

        ref_tgt_neighbours = neighbours_of(ref_tgt_ent, max_hop=self.max_hops, ignore_root=True)
        tgt_cand_iris = []
        hop = 1
        # extract from the nearest neighbours until enough candidates or max hop
        while len(tgt_cand_iris) < n_cands and hop <= self.max_hops:
            neighbours_of_cur_hop = list(set(ref_tgt_neighbours[hop]) - avoided_family)
            # if by adding neighbours of current hop the require number will be met
            # we randomly pick among them
            if len(neighbours_of_cur_hop) > n_cands - len(tgt_cand_iris):
                neighbours_of_cur_hop = random.sample(
                    neighbours_of_cur_hop, n_cands - len(tgt_cand_iris)
                )
            # some neighbour
            tgt_cand_iris += [tgt_cand.iri for tgt_cand in neighbours_of_cur_hop]
            tgt_cand_iris = uniqify(tgt_cand_iris)
            hop += 1

        assert len(tgt_cand_iris) <= n_cands
        assert not ref_tgt_ent_iri in tgt_cand_iris
        return tgt_cand_iris

    def avoided_family_of(self, ent: ThingClass):
        """Compute the (direct) family members of an entity class
        which should be avoided in subsumption matching
        """
        family = [ent]
        if self.avoid_ancestors:
            print("avoid ancestors for subsumption matching ...")
            family += thing_class_ancestors_of(ent)
        if self.avoid_descendents:
            family += thing_class_descendants_of(ent)
            print("avoid descendants for subsumption matching ...")
        return set(family)
