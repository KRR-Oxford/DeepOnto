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
"""Class for processing text semantics from an ontology"""

from __future__ import annotations

import networkx as nx
import itertools
import random
from typing import List, Set, Tuple, Optional
from pyats.datastructures import AttrDict
import ontoplm
from ontoplm import SavedObj
from ontoplm.utils import rand_sample_excl, uniqify


class TextSemantics(SavedObj):
    def __init__(self, transitive: bool = False):
        # note that reflexivity and symmetricality have been assumed by 
        # e.g., if a class C has labels {x, y}, we include (x, x), (y, y) 
        # for reflexivity; and (x, y), (y, x) for symmetricality.
        self.transitive = transitive
        self.indivs = []
        self.indiv_names = []
        self.indiv_synonyms_sizes = []
        self.merged = dict()
        super().__init__("txtsem")

    def __call__(self, *ontos: ontoplm.onto.Ontology):
        self.__init__(self.transitive)
        for i in range(len(ontos)):
            onto = ontos[i]
            grouped_synonyms = self.synonym_field(onto)
            self.indivs.append(
                AttrDict(
                    {
                        "info": onto.info,
                        "num_synonyms": len(grouped_synonyms),
                        "synonyms": grouped_synonyms,
                    }
                )
            )
            self.indiv_names.append(onto.owl.name)
            self.indiv_synonyms_sizes.append(len(grouped_synonyms))
        if self.transitive:
            self.merge_semantics()
        print(self)

    def __str__(self) -> str:
        self.info = AttrDict(
            {
                "transitive": self.transitive,
                "owl_names": self.indiv_names,
                "indv_synonyms_sizes": self.indiv_synonyms_sizes,
                "indv_synonyms_total": sum(self.indiv_synonyms_sizes),
            }
        )
        if self.transitive:
            self.info.merged_synonyms_size = self.merged.num_synonyms
            self.info.reduced_synonyms_size = self.info.indv_synonyms_total - self.merged.num_synonyms
        return super().report(**self.info)

    def merge_semantics(self):
        all_synonyms = [ind.synonyms for ind in self.indivs]
        merged_synonym_field = self.merge_synonym_fields(*all_synonyms)
        self.merged = AttrDict(
            {"num_synonyms": len(merged_synonym_field), "synonyms": merged_synonym_field}
        )

    def synonym_field(self, onto: ontoplm.onto.Ontology) -> List[Set[str]]:
        grouped_synonyms = []
        # group labels by their associated classes if transitivity is not considered
        if not self.transitive:
            grouped_synonyms = [set(v) for v in onto.class2labs.values()]
        # group labels using adjacency of graph
        else:
            label_pairs = []
            for v in onto.class2labs.values():
                # cannot rule out edges to self otherwise singletons will be removed
                label_pairs += itertools.product(v, v)
            grouped_synonyms = self.connected_labels(label_pairs)
        return grouped_synonyms

    @classmethod
    def merge_synonym_fields(cls, *synonym_fields_seq: List[Set[str]]):
        """with transitivity assumed, 

        Returns:
            [type]: [description]
        """
        label_pairs = []
        for grouped_synonyms in synonym_fields_seq:
            for synonym_set in grouped_synonyms:
                label_pairs += itertools.product(synonym_set, synonym_set)
        merged_grouped_synonyms = cls.connected_labels(label_pairs)
        return merged_grouped_synonyms

    @staticmethod
    def connected_labels(label_pairs: List[Tuple[str, str]]) -> List[Set[str]]:
        """build a graph for adjacency among the class labels such that
        the transitivity of synonyms is ensured

        Args:
            label_pairs (List[Tuple[str, str]]): label pairs that are synonymous

        Returns:
            List[Set[str]]: a collection of synonym sets
        """
        G = nx.Graph()
        G.add_edges_from(label_pairs)
        # nx.draw(G, with_labels = True)
        cc = list(nx.connected_components(G))
        return cc
    
    ##################################################################################
    ###                          +ve and -ve sampling                              ###
    ##################################################################################
    
    @staticmethod
    def positive_sampling(grouped_synonyms: List[Set[str]], pos_num: Optional[int] = None):
        pos_sample_pool = []
        for synonym_set in grouped_synonyms:
            synonym_pairs = list(itertools.product(synonym_set, synonym_set))
            pos_sample_pool += synonym_pairs
        pos_sample_pool = uniqify(pos_sample_pool)
        if (not pos_num) or (pos_num >= len(pos_sample_pool)):
            # return all the possible synonyms if no maximum limit
            return pos_sample_pool
        else:
            return random.sample(pos_sample_pool, pos_num)

    @staticmethod
    def random_negative_sampling(grouped_synonyms: List[Set[str]], neg_num: int):
        """soft non-synonyms are defined as label pairs from two disjoint synonym groups
        that are randomly selected
        """
        neg_sample_pool = []
        # randomly select disjoint synonym group pairs
        while len(neg_sample_pool) < neg_num:
            left, right = tuple(random.sample(grouped_synonyms, 2))
            # randomly choose one label from a synonym group
            left_label = random.choice(list(left))  
            right_label = random.choice(list(right))  
            neg_sample_pool.append((left_label, right_label))
            neg_sample_pool = uniqify(neg_sample_pool)
        return neg_sample_pool


    @staticmethod
    def disjointness_negative_sampling(onto: ontoplm.onto.Ontology, neg_num: int):
        pass
