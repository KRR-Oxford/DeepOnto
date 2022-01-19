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
from pyats.datastructures import AttrDict
import random

from deeponto.onto import Ontology
from deeponto.onto.text import Tokenizer
from deeponto.onto.mapping import OntoMappings, EntityMapping


# TODO: to be updated constantly
sampling_options = ["idf"]


class OntoAlignNegCandsSampler:
    def __init__(
        self, src_onto: Ontology, tgt_onto: Ontology, tsv_mappings_path: str, tokenizer: Tokenizer
    ):
        
        self.src_onto = src_onto
        self.tgt_onto = tgt_onto
        
        # loaded mappings are served as +ve candiates (anchors)
        # note that dict cannot be used here because multiple mappings are possible
        self.pos_src2tgt = OntoMappings.read_tsv_mappings(tsv_mappings_path)
        self.pos_tgt2src = [(y, x) for (x, y) in self.pos_src2tgt]
        self.tokenizer = tokenizer

        self.flag_set = cycle(["src2tgt", "tgt2src"])
        self.flag = next(self.flag_set)

    def sample(self, n_cands: int, strategy: str):
        """Sample negative candidates using provided strategy for both sides (src and tgt)
        """
        if strategy in sampling_options:
            # implemented sample names are all in: "xx_sample"
            sampler = getattr(self, f"{strategy}_sample")
            self.renew()
            fixed_src_ent_pairs = sampler(n_cands)
            self.switch()
            fixed_tgt_ent_pairs = sampler(n_cands)
            return AttrDict({"fixed_src": fixed_src_ent_pairs, "fixed_tgt": fixed_tgt_ent_pairs})
        else:
            raise ValueError(f"Valid sampling options are: {sampling_options}")

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

    def random_select(self, n_cands: int):
        """Randomly select n target candidates
        """
        return random.sample(range(len(self.tgt_onto.class2idx)), n_cands)

    ##################################################################################
    ###                            sampling stratgies                              ###
    ##################################################################################

    def idf_sample(self, n_cands: int) -> OntoMappings:
        """Sample negative candidates using inverted index-based idf scoring
        """

        # rel "?" means to-be-confirmed
        ent_pairs = OntoMappings(flag=self.flag, n_best=n_cands, rel="?")

        for src_ent, tgt_ent in getattr(self, f"pos_{self.flag}"):
            
            # for multiple mappings, only need to add the ground truth:
            if src_ent in ent_pairs.ranked.keys():
                tbc_mapping = EntityMapping(src_ent, tgt_ent, rel="?", score=1.0)
                ent_pairs.add(tbc_mapping)
                continue

            # select n candidates for the src ent
            src_ent_id = self.src_onto.class2idx[src_ent]
            src_ent_labs = self.src_onto.idx2labs[src_ent_id]
            src_ent_toks = self.tokenizer.tokenize_all(src_ent_labs)
            tgt_cand_ids = self.tgt_onto.idf_select(src_ent_toks, n_cands)  # [(ent_id, idf_score)]

            if not tgt_cand_ids:
                # randomly select when idf select nothing
                print(f"idf select nothing => randomly select {n_cands} candidates ...")
                tgt_cand_ids = self.random_select(n_cands)
            else:
                # unzip to list of ids
                tgt_cand_ids = list(list(zip(*tgt_cand_ids))[0])
            # check if the positive pair is included
            tgt_ent_id = self.tgt_onto.class2idx[tgt_ent]
            if not tgt_ent_id in tgt_cand_ids:
                tgt_cand_ids.insert(0, tgt_ent_id)

            # store the mapping
            for tgt_cand_id in tgt_cand_ids[:n_cands]:
                tgt_cand = self.tgt_onto.idx2class[tgt_cand_id]
                score = 1.0 if tgt_cand_id == tgt_ent_id else 0.0
                tbc_mapping = EntityMapping(src_ent, tgt_cand, rel="?", score=score)
                ent_pairs.add(tbc_mapping)

        return ent_pairs
