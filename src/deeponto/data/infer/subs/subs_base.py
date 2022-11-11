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
"""Base Class for Ontology Subsumption Inference Data Sampling"""

import itertools
import random
from typing import Callable, Optional, List
import enlighten
from collections import defaultdict

from deeponto.onto import Ontology
from deeponto.onto.logic.reasoner import OWLReasoner
from deeponto import SavedObj, OWL_THING


class SubsumptionSamplerBase:
    def __init__(
        self,
        onto: Ontology,
        neg_ratio: int = 1,
        hard_neg_ratio: Optional[int] = None,
    ):
        self.onto = onto
        self.reasoner = OWLReasoner(onto.owl_path)
        self.class_iris = list(self.reasoner.owlClasses.keys())
        self.class_iris_with_root = self.class_iris + [OWL_THING]
        self.obj_prop_iris = list(self.reasoner.owlObjectProperties.keys())
        self.subs = self.init_subs()
        self.neg_ratio = neg_ratio
        # set the hard negative ratio the same as the soft negative ratio if not specified
        self.hard_neg_ratio = hard_neg_ratio if hard_neg_ratio else neg_ratio
        # pre-compute classes that have multiple (inferred and direct) children
        self.non_single_child_classes = dict()
        self.sibling_pairs = []
        for cl_iri in self.class_iris_with_root:
            owlClass = self.reasoner.owlClass_from_iri(cl_iri)
            children = self.reasoner.subclasses_of(owlClass, direct=True)
            if len(children) >= 2:
                self.non_single_child_classes[cl_iri] = children
                self.sibling_pairs += [
                    (x, y) for x, y in itertools.product(children, children) if x != y
                ]  # all possible combinations excluding reflexive pairs
        self.sibling_pairs = list(set(self.sibling_pairs))
        # an additional sibling dictionary for customized (fixed one sample) sampling
        self.sibling_dict = defaultdict(list)
        for l, r in self.sibling_pairs:
            self.sibling_dict[l].append(r)
            self.sibling_dict[r].append(l)
        print(
            f"{len(self.non_single_child_classes)}/{len(self.class_iris_with_root)} (including Owl:Thing) has multiple (direct and inferred) children ..."
        )
        print(
            f"In total there are {len(self.sibling_pairs)} sibling class pairs (order matters) ..."
        )
