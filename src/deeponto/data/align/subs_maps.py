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
"""Generate Subsumption Mappings from Equivalence Mappings"""

import random
from pyats.datastructures import AttrDict

from deeponto.onto.text.text_utils import unfold_iri, abbr_iri
from deeponto.onto.graph.graph_utils import super_thing_classes_of
from deeponto.onto import Ontology
from deeponto.utils import uniqify
from deeponto.onto.mapping import OntoMappings
from deeponto import FlaggedObj


class SubsumptionMappingGenerator(FlaggedObj):
    def __init__(
        self,
        src_onto: Ontology,
        tgt_onto: Ontology,
        equiv_mappings_path: str,
        map_ratio: int = 1,
        delete_equiv_src: bool = False,
        delete_equiv_tgt: bool = True,
    ):
        super().__init__(src_onto, tgt_onto)
        self.src2tgt = AttrDict()
        self.tgt2src = AttrDict()
        self.init_memory_bank()
        self.src2tgt.equivs = OntoMappings.read_tsv_mappings(equiv_mappings_path).to_tuples()
        self.switch()
        self.init_memory_bank()
        self.renew()
        self.tgt2src.equivs = [(y, x) for (x, y) in self.src2tgt.equivs]
        self.map_ratio = map_ratio
        self.delete_equiv_src = delete_equiv_src  # delete the source equiv class or not
        self.delete_equiv_tgt = delete_equiv_tgt  # delete the target equiv class or not

    def init_memory_bank(self):
        cur_bank = getattr(self, self.flag)
        cur_bank.equivs = []
        cur_bank.subs = []
        cur_bank.constructed = {self.src_onto.owl.name: [], self.tgt_onto.owl.name: []}
        cur_bank.deleted = {self.src_onto.owl.name: [], self.tgt_onto.owl.name: []}

    def generate_all_is_a_maps(self):
        """Generate ${map_raio} (direct 1-hob) subsumption mappings from the 
        equivalence mappings while fixing the source side
        """
        left, right = self.flag.split("2")[0], self.flag.split("2")[1]
        print(f"generate subsumption mappings from {right} side with {left} side fixed")
        print(
            f"delete SRC equiv class: {self.delete_equiv_src};",
            f"delete TGT equiv class: {self.delete_equiv_tgt}",
        )
        # if a target entity has been involved in a subsumption mapping
        # mark the rest of equiv mappings that involve it as invalid
        cur_bank = getattr(self, self.flag)
        for ref_src_ent_name, ref_tgt_ent_name in cur_bank.equivs:
            # skip if the src/tgt class of an equiv mapping is opted to be deleted
            # and it has been involved in a subsumption mapping
            # NOTE: do not delete what have been constructed
            if (
                ref_src_ent_name in cur_bank.constructed[self.src_onto.owl.name]
                and self.delete_equiv_src
            ) or (
                ref_tgt_ent_name in cur_bank.constructed[self.tgt_onto.owl.name]
                and self.delete_equiv_tgt
            ):
                continue
            cur_subs = self.generate_is_a_map_from_equiv_pair(ref_src_ent_name, ref_tgt_ent_name)
            cur_bank.constructed[self.src_onto.owl.name] += [src for src, _ in cur_subs]
            cur_bank.constructed[self.tgt_onto.owl.name] += [tgt for _, tgt in cur_subs]
            cur_bank.subs += cur_subs
        cur_bank.subs = uniqify(cur_bank.subs)

    def generate_is_a_map_from_equiv_pair(self, src_ent_name: str, tgt_ent_name: str):
        """Generate a (direct 1-hob) subsumption (IS-A) mapping from an equivalence mapping
        while fixing the source side
        """
        cur_bank = getattr(self, self.flag)
        # src_ent = self.src_onto.owl.search(iri=unfold_iri(src_ent_name))[0]
        tgt_ent = self.tgt_onto.owl.search(iri=unfold_iri(tgt_ent_name))[0]
        parents_of_tgt_ent = super_thing_classes_of(tgt_ent)
        
        # NOTE: do not construct what have been deleted
        valid_parent_names = []
        for p in parents_of_tgt_ent:
            p_name = abbr_iri(p.iri)
            if not p_name in cur_bank.deleted[self.tgt_onto.owl.name]:
                valid_parent_names.append(p_name)

        if len(valid_parent_names) > self.map_ratio:
            valid_parent_names = random.sample(valid_parent_names, self.map_ratio)
        subs_pairs = [(src_ent_name, p_name) for p_name in valid_parent_names]
        assert len(subs_pairs) == len(set(subs_pairs))
        if subs_pairs:
            if self.delete_equiv_src:
                cur_bank.deleted[self.src_onto.owl.name].append(src_ent_name)
            if self.delete_equiv_tgt:
                cur_bank.deleted[self.tgt_onto.owl.name].append(tgt_ent_name)
        return subs_pairs

    def deleted_equiv_src_iris(self):
        """Return iris of the source classes from the *used* equivalence mapping
        for downstream class deletion
        """
        cur_bank = getattr(self, self.flag)
        return [unfold_iri(s) for s in cur_bank.deleted[self.src_onto.owl.name]]

    def deleted_equiv_tgt_iris(self):
        """Return iris of the target classes from the *used* equivalence mapping
        for downstream class deletion
        """
        cur_bank = getattr(self, self.flag)
        return [unfold_iri(t) for t in cur_bank.deleted[self.tgt_onto.owl.name]]

    def preserved_src_iris(self):
        """Return iris of the source classes that should be preserved (not used for 
        constructing the subsumption mappings)
        """
        all_src_iris = set([unfold_iri(x) for x in self.src_onto.class2idx.keys()])
        return list(all_src_iris - set(self.deleted_equiv_src_iris()))

    def preserved_tgt_iris(self):
        """Return iris of the target classes that should be preserved (not used for 
        constructing the subsumption mappings)
        """
        all_tgt_iris = set([unfold_iri(x) for x in self.tgt_onto.class2idx.keys()])
        return list(all_tgt_iris - set(self.deleted_equiv_tgt_iris()))
