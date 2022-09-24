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
"""Providing useful utility functions"""

import itertools
import random
from owlready2.entity import ThingClass

from deeponto.onto import Ontology
from deeponto.onto.graph.graph_utils import thing_class_ancestors_of, thing_class_descendants_of


class SubsumptionPairGenerator:
    def __init__(self, onto: Ontology, neg_ratio: int):
        self.onto = onto
        self.subs = {"entailment": [], "contradiction": []}
        self.neg_ratio = neg_ratio

    def sample(self):
        self.subs = {"entailment": [], "contradiction": []}
        self.entailment_pairs()
        self.contradiction_pairs()
        assert not set(self.subs["entailment"]).intersection(self.subs["contradiction"])

    def save(self, saved_path: str):
        """Save in json format similar to the huggingface datasets manner
        """
        data_dict = {}
        pass

    @classmethod
    def check_different_branch(cls, ent1: ThingClass, ent2: ThingClass):
        """Check if any of the two entities are from different branches
        (i.e., no intersection below Thing Class or there exists a disjointness
        among their ancestors)
        """
        ancestors1 = thing_class_ancestors_of(ent1, include_self=True)
        descendants1 = thing_class_descendants_of(ent1, include_self=True)
        ancestors2 = thing_class_ancestors_of(ent2, include_self=True)
        descendants2 = thing_class_descendants_of(ent2, include_self=True)

        # check any disjoint ancestors (including self)
        for a1, a2 in itertools.product(ancestors1, ancestors2):
            if cls.check_disjoint(a1, a2):
                return True

        # check if no intersection is found below the Thing Class
        common_ancestors = set(ancestors1).intersection(set(ancestors2))
        common_descendants = set(descendants1).intersection(set(descendants2))
        if not common_ancestors and not common_descendants:
            return True

        return False

    def entailment_pairs(self):
        """Extract all subsumption pairs that indicate entailment from left to right,
        i.e., if A is {left} => then A is {right}.
        """
        for cl in self.onto.classes:
            left_labs = self.onto.iri2labs[cl.iri]
            right_labs = []
            for subsumer in thing_class_ancestors_of(cl):
                right_labs += self.onto.iri2labs[subsumer.iri]
            for pos_pair in itertools.product(left_labs, right_labs):
                self.subs["entailment"].append(tuple(pos_pair))

    def contradiction_pairs(self):
        """Sample false subsumption pairs that indicate contradiction from left to right,
        i.e., if A is {left} !=> then A is {right}. Such pairs are classes that share NO
        ancestors. 
        """
        classes = self.onto.classes
        max_neg = len(self.subs["entailment"]) * self.neg_ratio
        max_iter = 10 * max_neg
        i = 0
        while len(self.subs["contradiction"]) < max_neg and i < max_iter:
            # NOTE: sampling method can be improved
            left, right = tuple(random.sample(classes, k=2))
            # sanitiy check to reduce false negatives
            if self.check_different_branch(left, right):
                left_labs = self.onto.iri2labs[left.iri]
                right_labs = self.onto.iri2labs[right.iri]
                neg_cands = list(itertools.product(left_labs, right_labs))
                neg_pair = random.choice(neg_cands)
                if not neg_pair in self.subs["contradiction"]:
                    self.subs["contradiction"].append(neg_pair)
            i += 1
        self.subs["contradiction"] = list(sorted(self.subs["contradiction"]))
