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

from itertools import cycle
import random

from deeponto.onto import Ontology
from deeponto.onto.graph.graph_utils import *
from deeponto.onto.text import Tokenizer
from deeponto.onto.text.text_utils import unfold_iri, abbr_iri
from deeponto.onto.mapping import OntoMappings, EntityMapping, AnchoredOntoMappings
from deeponto.utils import uniqify
from deeponto.utils.logging import banner_msg
from deeponto import SavedObj


# TODO: to be updated constantly
sampling_options = ["random", "idf", "neighbour"]


class OntoAlignNegCandsSampler:
    def __init__(
        self,
        src_onto: Ontology,
        tgt_onto: Ontology,
        ref_mappings_path: str,
        rel: str,
        tokenizer: Tokenizer,
        max_hobs: int = 5,
        avoid_ancestors: bool = False,
        avoid_descendents: bool = False,
    ):

        self.src_onto = src_onto
        self.tgt_onto = tgt_onto
        self.rel = rel
        self.flag_set = cycle(["src2tgt", "tgt2src"])
        self.flag = next(self.flag_set)

        # loaded mappings are served as +ve candiates (anchors)
        # note that dict cannot be used here because multiple mappings are possible
        self.pos_src2tgt = OntoMappings.read_tsv_mappings(
            tsv_mappings_path=ref_mappings_path, rel=self.rel
        ).to_tuples()
        self.pos_tgt2src = [(y, x) for (x, y) in self.pos_src2tgt]
        self.tokenizer = tokenizer

        # init the reference anchored mappings
        self.src2tgt_anchored_mappings = AnchoredOntoMappings(
            flag="src2tgt", n_best=None, rel=self.rel
        )
        self.init_anchors()
        self.switch()
        self.tgt2src_anchored_mappings = AnchoredOntoMappings(
            flag="tgt2src", n_best=None, rel=self.rel
        )
        self.init_anchors()
        self.renew()

        self.stats = defaultdict(lambda: {"random": 0})

        # arguments for neighbour sampling
        self.max_hobs = max_hobs
        self.avoid_ancestors = avoid_ancestors
        self.avoid_descendents = avoid_descendents

    def init_anchors(self):
        """Add all the reference anchor mappings as a candidate mapping first
        """
        anchored_maps = self.current_anchor_mappings()
        for ref_src_ent_name, ref_tgt_ent_name in getattr(self, f"pos_{self.flag}"):
            anchor_map = EntityMapping(ref_src_ent_name, ref_tgt_ent_name, self.rel, 0.0)
            anchored_maps.add(anchor_map, anchor_map)

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
        for ref_src_ent_name, ref_tgt_ent_name in getattr(self, f"pos_{self.flag}"):
            banner_msg(f"CandMaps for ({ref_src_ent_name}, {ref_tgt_ent_name})")
            anchor_map = EntityMapping(ref_src_ent_name, ref_tgt_ent_name, self.rel, 0.0)
            tgt_cand_names, sample_stats = self.mixed_sample(ref_tgt_ent_name, **strategy2nums)
            self.stats[str((ref_src_ent_name, ref_tgt_ent_name))] = sample_stats
            for tgt_cand_name in tgt_cand_names:
                cand_map = EntityMapping(ref_src_ent_name, tgt_cand_name, self.rel, 0.0)
                self.current_anchor_mappings().add(anchor_map, cand_map, allow_existed=False)
            print("Candidate mappings statistics:")
            SavedObj.print_json(sample_stats)

        self.current_anchor_mappings().save_instance(f"./{self.flag}.rank/for_eval")
        self.current_anchor_mappings().unscored_cand_maps().save_instance(f"./{self.flag}.rank/for_score")
        SavedObj.save_json(self.stats, f"./{self.flag}.rank/stats.json")

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

    def current_anchor_mappings(self):
        return getattr(self, f"{self.flag}_anchored_mappings")

    ##################################################################################
    ###                            sampling stratgies                              ###
    ##################################################################################

    def mixed_sample(self, ref_tgt_ent_name, **strategy2nums):
        """Randomly select specified candidates for each strategy;
        amend random samples if not meeting the number
        """
        all_tgt_cand_names = []
        stats = defaultdict(lambda: 0)
        i = 0
        total_n_cands = 0
        for strategy, n_cands in strategy2nums.items():
            i += 1
            total_n_cands += n_cands
            if strategy in sampling_options:
                sampler = getattr(self, f"{strategy}_sample")
                # for ith iteration, the worst case is when (i-1)*n_cands are duplicated
                # so we generate i*n_cands candidates first and prune the rest
                # another edge case is when sampled candidates are not sufficient
                # but we have covered the worst case for the duplicates
                cur_tgt_cands = sampler(ref_tgt_ent_name, i * n_cands)
                # remove the duplicated candidates and prune the tail
                cur_tgt_cands = list(set(cur_tgt_cands) - set(all_tgt_cand_names))[:n_cands]
                stats[strategy] += len(cur_tgt_cands)
                # use random samples for complementation if not enough
                while len(cur_tgt_cands) < n_cands:
                    amend_cands = self.random_sample(ref_tgt_ent_name, n_cands - len(cur_tgt_cands))
                    amend_cands = list(
                        set(amend_cands) - set(all_tgt_cand_names) - set(cur_tgt_cands)
                    )
                    cur_tgt_cands += amend_cands
                assert len(cur_tgt_cands) == n_cands
                if strategy != "random":
                    stats["random"] += n_cands - stats[strategy]
                all_tgt_cand_names += cur_tgt_cands
            else:
                raise ValueError(f"Invalid sampling trategy: {strategy}.")
        assert len(all_tgt_cand_names) == total_n_cands
        return all_tgt_cand_names, stats

    def random_sample(self, ref_tgt_ent_name, n_cands: int):
        """Randomly select n target candidates
        """
        all_tgt_ent_names = list(self.tgt_onto.class2idx.keys())
        all_tgt_ent_names.remove(ref_tgt_ent_name)
        return random.sample(all_tgt_ent_names, n_cands)

    def idf_sample(self, ref_tgt_ent_name, n_cands: int):
        """Sample negative candidates using inverted index-based idf scoring
        """
        # select n candidates for the target entity
        ref_tgt_ent_id = self.tgt_onto.class2idx[ref_tgt_ent_name]
        ref_tgt_ent_labs = self.tgt_onto.idx2labs[ref_tgt_ent_id]
        ref_tgt_ent_toks = self.tokenizer.tokenize_all(ref_tgt_ent_labs)
        # sample 1 more candidate to prevent when reference target is included
        tgt_cand_ids = self.tgt_onto.idf_select(
            ref_tgt_ent_toks, n_cands + 1
        )  # [(ent_id, idf_score)]

        # unzip to list of ids
        tgt_cand_ids = list(list(zip(*tgt_cand_ids))[0])
        # remove the positive pair if included
        if ref_tgt_ent_id in tgt_cand_ids:
            tgt_cand_ids.remove(ref_tgt_ent_id)

        tgt_cand_names = [self.tgt_onto.idx2class[i] for i in tgt_cand_ids]
        assert not ref_tgt_ent_name in tgt_cand_names
        return tgt_cand_names[:n_cands]

    def neighbour_sample(self, ref_tgt_ent_name, n_cands: int):
        """Sample negative candidates from nearest nearest (starting from 1-hob)
        """

        # select n candidates for the target entity
        ref_tgt_ent = self.tgt_onto.owl.search(iri=unfold_iri(ref_tgt_ent_name))[0]
        # print(self.tgt_onto.owl.search(iri=unfold_iri(ref_tgt_ent_name)))

        # for subsumption mapping, ancestors or descendants will be avoided
        ref_tgt_ancestors = ancestors_of(ref_tgt_ent) if self.avoid_ancestors else []
        ref_tgt_descendants = descendants_of(ref_tgt_ent) if self.avoid_descendents else []
        # avoid also "self"
        avoid_neighbors = set([ref_tgt_ent] + ref_tgt_ancestors + ref_tgt_descendants)

        ref_tgt_neighbours = neighbours_of(ref_tgt_ent, max_hob=self.max_hobs, ignore_root=True)
        tgt_cand_names = []
        hob = 1
        # extract from the nearest neighbours until enough candidates or max hob
        while len(tgt_cand_names) < n_cands and hob <= self.max_hobs:
            neighbours_of_cur_hob = list(set(ref_tgt_neighbours[hob]) - avoid_neighbors)
            # if by adding neighbours of current hob the require number will be met
            # we randomly pick among them
            if len(neighbours_of_cur_hob) > n_cands - len(tgt_cand_names):
                neighbours_of_cur_hob = random.sample(
                    neighbours_of_cur_hob, n_cands - len(tgt_cand_names)
                )
            # some neighbour
            tgt_cand_names += [abbr_iri(tgt_cand.iri) for tgt_cand in neighbours_of_cur_hob]
            tgt_cand_names = uniqify(tgt_cand_names)
            hob += 1

        # tgt_cand_names = [abbr_iri(tgt_cand.iri) for tgt_cand in tgt_cands]
        assert len(tgt_cand_names) <= n_cands
        assert not ref_tgt_ent_name in tgt_cand_names
        return tgt_cand_names
