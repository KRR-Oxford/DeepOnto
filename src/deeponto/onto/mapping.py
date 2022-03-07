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


##################################################################################
###                            entity mapping collection                       ###
##################################################################################


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
        for em in ent_mappings:
            self.add(em)
        super().__init__(f"{self.flag}.maps")

    def __str__(self):
        self.info = AttrDict(
            {
                "flag": self.flag,
                "relation": self.rel,
                "n_best": self.n_best,
                "num_heads": len(self.ranked),
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

    def topKs(self, threshold: float = 0.0, K: int = 1) -> List[Tuple[str, str]]:
        """Return the top ranked mappings for each head entity with scores >= threshold,
        output mappings are transformed to tuples
        """
        ent_tuple_list = []
        for src_ent_name, v in self.ranked.items():
            for tgt_ent_name, score in list(v.items())[:K]:
                if score >= threshold:
                    ent_tuple_list.append((src_ent_name, tgt_ent_name))
        return ent_tuple_list

    def to_tuples(self) -> List[Tuple[str, str]]:
        """Unravel the ranked dictionary to tuples
        """
        return self.topKs(0.0, K=self.n_best)

    def check_type(self, em: EntityMapping):
        if em.rel != self.rel:
            raise ValueError("Input mappings are not of the same type (relation).")

    def check_existed(self, em: EntityMapping):
        return em.tail in self.ranked[em.head].keys()

    def add_many(self, *ems: EntityMapping):
        """Add a list of new mappings while keeping the ranking
        """
        for em in ems:
            self.add(em)

    def add(self, em: EntityMapping):
        """Add a new entity mapping or add an existing mapping to update mapping score (take average)
        while keeping the ranking
        """
        self.check_type(em)
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
        # truncate if n_best is specified
        if self.n_best:
            self.ranked[em.head] = sort_dict_by_values(self.ranked[em.head], top_k=self.n_best)

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
        onto_mappings = cls(flag, n_best, rel)
        for _, dp in df.iterrows():
            if dp["Score"] >= threshold:
                onto_mappings.add(EntityMapping(dp["SrcEntity"], dp["TgtEntity"], rel, dp["Score"]))
        return onto_mappings


class EntityMappingList(list):
    def append(self, em: EntityMapping):
        if isinstance(em, EntityMapping):
            super().append(em)
        else:
            raise TypeError("Only Entity Mapping can be added to the list.")

    def top_k(self, k: int):
        """Return top K scored mappings from the list
        """
        return EntityMappingList(sorted(self, key=lambda x: x.score, reverse=True))[:k]

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
