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
"""Class for ontology entity mappings."""

from __future__ import annotations

from collections import defaultdict
from typing import Optional, List, Tuple
from pyats.datastructures import AttrDict

from deeponto import SavedObj
from deeponto.utils import sort_dict_by_values, read_tsv
from deeponto.onto.text.text_utils import abbr_iri


##################################################################################
###                            entity mapping collection                       ###
##################################################################################


class AnchoredOntoMappings(SavedObj):
    def __init__(
        self,
        flag: str,
        n_best: Optional[int],
        rel: str,
        *anchor_cand_pairs: Tuple[EntityMapping, EntityMapping],
    ):
        """Store ranked (by score) mappings for each reference (head, tail) pairs:
        {
            ...
            ("anchor_head_ent_i", "anchor_tail_ent_i": {
            ...
            Sorted({
                ...
                "tail_ent_j": score(i, j) # NOTE: anchor_tail_ent_i is somewhere
                ...
                })
            ...
            }
            ...
        }
        NOTE: [anchor_head, anchor_tail] ensures uniqueness to a anchor mapping
        """
        self.flag = flag
        self.n_best = n_best
        self.rel = rel
        # save mappings in disjoint partitions to prevent overriding keys (src entities)
        self.anchor2cand = defaultdict(dict)
        self.cand2anchor = defaultdict(list)
        self.add_many(*anchor_cand_pairs)
        super().__init__(f"{self.flag}.anchored.maps")

    def __str__(self):
        self.info = AttrDict(
            {
                "flag": self.flag,
                "relation": self.rel,
                "n_best": self.n_best,
                "num_anchors": len(self.anchor2cand),
                "num_maps": len(self),
            }
        )
        return super().report(**self.info)

    def __len__(self):
        """Total number of ranked mappings
        """
        return sum([len(map_dict) for map_dict in self.anchor2cand.values()])

    def save_instance(self, saved_path):
        """save the current instance locally
        """
        super().save_instance(saved_path)
        # also save a readable format of the ranked alignment set
        anchor2cand_json = {str(k): v for k, v in self.anchor2cand.items()}
        self.save_json(anchor2cand_json, saved_path + f"/{self.saved_name}.json")

    def add(self, anchor_map: EntityMapping, cand_map: EntityMapping, allow_existed: bool = True):
        """Given an anchor mapping, add a new candidate mapping or add an existing 
        candidate mapping to update mapping score (take average) while keeping the ranking
        """
        self.validate_input(anchor_map, cand_map)
        # average the mapping scores if already existed
        if self.check_existed(anchor_map, cand_map):
            if not allow_existed:
                raise ValueError("Duplicate mapping not allowed ...")
            old_score = self.anchor2cand[anchor_map.head, anchor_map.tail][cand_map.tail]
            print(f"Found an existing mapping...")
            print(f"\t[Old]: {EntityMapping(cand_map.head, cand_map.tail, self.rel, old_score)}")
            print(
                f"\t[New]: {EntityMapping(cand_map.head, cand_map.tail, self.rel, cand_map.score)}"
            )
            new_score = (old_score + cand_map.score) / 2
            print(f"\t ==> update score to {new_score}")
            self.anchor2cand[anchor_map.head, anchor_map.tail][cand_map.tail] = new_score
        else:
            self.anchor2cand[anchor_map.head, anchor_map.tail][cand_map.tail] = cand_map.score
        self.cand2anchor[cand_map.head, cand_map.tail].append(anchor_map.tail)
        # rank according to mapping scores and preserve n_best (if specified)
        self.anchor2cand[anchor_map.head, anchor_map.tail] = sort_dict_by_values(
            self.anchor2cand[anchor_map.head, anchor_map.tail], top_k=self.n_best
        )

    def add_many(self, *anchor_cand_pairs: Tuple[EntityMapping, EntityMapping]):
        """Add a list of anchor-cand mapping pairs while keeping the ranking
        """
        for anchor_map, cand_map in anchor_cand_pairs:
            self.add(anchor_map, cand_map)

    def fill_scored_maps(self, scored_onto_maps: OntoMappings):
        """Fill mapping score from scored onto mappings
        """
        assert self.flag == scored_onto_maps.flag
        num_valid = 0
        for src_ent_name, v in scored_onto_maps.ranked.items():
            for tgt_ent_name, score in v.items():
                if self.cand2anchor[src_ent_name, tgt_ent_name]:
                    num_valid += 1
                    for anchor_tail in self.cand2anchor[src_ent_name, tgt_ent_name]:
                        self.anchor2cand[src_ent_name, anchor_tail][tgt_ent_name] = score
                    self.anchor2cand[src_ent_name, anchor_tail] = sort_dict_by_values(
                        self.anchor2cand[src_ent_name, anchor_tail], top_k=self.n_best
                    )
        print(
            f"{num_valid}/{len(scored_onto_maps)} of scored mappings are filled to corresponding anchors."
        )

    def unscored_cand_maps(self) -> OntoMappings:
        """Return all candidate mappings with no scores and anchors (so that duplicates will be merged)
        """
        unscored_cands = OntoMappings(self.flag, self.n_best, self.rel)
        for cand_tup in self.cand2anchor.keys():
            cand_map = EntityMapping(cand_tup[0], cand_tup[1], self.rel, 0.0)
            unscored_cands.add(cand_map)
        return unscored_cands

    def validate_input(self, anchor_map: EntityMapping, cand_map: EntityMapping):
        if anchor_map.rel != self.rel or cand_map.rel != self.rel:
            raise ValueError("Input mappings are not of the same type (relation).")
        if anchor_map.head != cand_map.head:
            raise ValueError(
                "Candidate mapping does not have the same head entity as the anchor mapping."
            )

    def check_existed(self, anchor_map: EntityMapping, cand_map: EntityMapping):
        return cand_map.tail in self.anchor2cand[anchor_map.head, anchor_map.tail].keys()


class OntoMappings(SavedObj):
    def __init__(self, flag: str, n_best: Optional[int], rel: str, *ent_mappings: EntityMapping):
        """Store ranked (by score) mappings for each head entity in Dict:
        {
            ...
            "head_ent_i": Sorted({
                ...
                "tail_ent_j": score(i, j)
                ...
                })
            ...
        }
        """
        self.flag = flag
        self.rel = rel
        self.n_best = n_best
        self.ranked = defaultdict(dict)
        self.add_many(*ent_mappings)
        super().__init__(f"{self.flag}.maps")

    def __str__(self):
        self.info = AttrDict(
            {
                "flag": self.flag,
                "relation": self.rel,
                "n_best": self.n_best,
                "num_heads": len(self.ranked),
                "num_maps": len(self),
            }
        )
        return super().report(**self.info)

    def __len__(self):
        """Total number of ranked mappings
        """
        return sum([len(map_dict) for map_dict in self.ranked.values()])

    def save_instance(self, saved_path):
        """save the current instance locally
        """
        super().save_instance(saved_path)
        # also save a readable format of the ranked alignment set
        self.save_json(self.ranked, saved_path + f"/{self.saved_name}.json")

    def get_mappings_of_ent(self, src_ent_name: str):
        """Return ranked mappings for a particular entry entity
        """
        ls = EntityMappingList()
        for tgt_ent_name, score in self.ranked[src_ent_name].items():
            ls.append(EntityMapping(src_ent_name, tgt_ent_name, self.rel, score))
        return ls

    def topKs(
        self, threshold: float = 0.0, K: int = 1, upper: float = 100.0
    ) -> List[Tuple[str, str]]:
        """Return the top ranked mappings for each head entity with scores >= threshold,
        output mappings are transformed to tuples
        """
        # NOTE: when K = None, slicing automatically gives the whole length
        # i.e., ls[:None] == ls[:len(ls)]
        ent_tuple_list = []
        for src_ent_name, v in self.ranked.items():
            for tgt_ent_name, score in list(v.items())[:K]:
                if score >= threshold and score < upper:
                    ent_tuple_list.append((src_ent_name, tgt_ent_name))
        return ent_tuple_list

    def to_tuples(self) -> List[Tuple[str, str]]:
        """Unravel the ranked dictionary to tuples
        """
        return self.topKs(0.0, K=self.n_best)

    def add(self, em: EntityMapping):
        """Add a new entity mapping or add an existing mapping to update mapping score (take average)
        while keeping the ranking
        """
        self.validate_input(em)
        # average the mapping scores if already existed
        if self.check_existed(em):
            old_score = self.ranked[em.head][em.tail]
            print(f"Found an existing mapping...")
            print(f"\t[Old]: {EntityMapping(em.head, em.tail, self.rel, old_score)}")
            print(f"\t[New]: {EntityMapping(em.head, em.tail, self.rel, em.score)}")
            new_score = (old_score + em.score) / 2
            print(f"\t ==> update score to {new_score}")
            self.ranked[em.head][em.tail] = new_score
        else:
            self.ranked[em.head][em.tail] = em.score
        # rank according to mapping scores and preserve n_best (if specified)
        self.ranked[em.head] = sort_dict_by_values(self.ranked[em.head], top_k=self.n_best)

    def add_many(self, *ems: EntityMapping):
        """Add a list of new mappings while keeping the ranking
        """
        for em in ems:
            self.add(em)

    def validate_input(self, em: EntityMapping):
        if em.rel != self.rel:
            raise ValueError("Input mappings are not of the same type (relation).")

    def check_existed(self, em: EntityMapping):
        return em.tail in self.ranked[em.head].keys()

    @classmethod
    def read_tsv_mappings(
        cls,
        tsv_mappings_path: str,
        threshold: Optional[float] = 0.0,
        flag: str = "src2tgt",
        rel: str = "=",
        n_best: Optional[int] = None,
    ):
        """Read mappings from tsv files and preserve mappings with scores >= threshold
        """
        df = read_tsv(tsv_mappings_path)
        onto_mappings = cls(flag=flag, n_best=n_best, rel=rel)
        for _, dp in df.iterrows():
            if dp["Score"] >= threshold:
                onto_mappings.add(
                    EntityMapping(
                        abbr_iri(dp["SrcEntity"]), abbr_iri(dp["TgtEntity"]), rel, dp["Score"]
                    )
                )
        return onto_mappings


class EntityMappingList(list):
    def append(self, em: EntityMapping):
        if isinstance(em, EntityMapping):
            super().append(em)
        else:
            raise TypeError("Only Entity Mapping can be added to the list.")

    def topKs(self, k: int):
        """Return top K scored mappings from the list
        """
        return EntityMappingList(sorted(self, key=lambda x: x.score, reverse=True))[:k]

    def sorted(self):
        """Return the sorted entity mapping list
        """
        return self.topKs(k=len(self))

    def __getitem__(self, item):
        result = list.__getitem__(self, item)
        if type(item) is slice:
            return EntityMappingList(result)
        else:
            return result

    def __repr__(self):
        info = type(self).__name__ + "(\n"
        for em in self:
            info += "  " + str(em) + ",\n"
        info += ")"
        return info


##################################################################################
###                               entity mapping                               ###
##################################################################################


class EntityMapping:
    def __init__(self, src_ent_name: str, tgt_ent_name: str, rel: str, score: float):
        self.head = src_ent_name
        self.tail = tgt_ent_name
        self.rel = rel
        self.score = score

    def to_tuple(self):
        return (self.head, self.tail)

    def __repr__(self):
        return f"EntityMapping({self.head} {self.rel} {self.tail}, {self.score})"


class EquivalenceMapping(EntityMapping):
    def __init__(self, src_ent_name: str, tgt_ent_name: str, score: float):
        super().__init__(src_ent_name, tgt_ent_name, "=", score)  # ≡


class SubsumptionMapping(EntityMapping):
    def __init__(self, src_ent_name: str, tgt_ent_name: str, score: float):
        super().__init__(src_ent_name, tgt_ent_name, "<", score)  # ⊂
