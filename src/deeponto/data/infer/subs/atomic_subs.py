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
"""Intra-ontology Atomic Subsumption Class Pair Generator for Text Inference"""

import itertools
import random
from typing import Callable, Optional

from deeponto.onto import Ontology
from . import SubsumptionSamplerBase


class AtomicSubsumptionSampler(SubsumptionSamplerBase):
    def __init__(
        self, onto: Ontology, neg_ratio: int = 1, hard_neg_ratio: Optional[int] = None,
    ):
        super().__init__(onto, neg_ratio)
        # set the hard negative ratio the same as the soft negative ratio if not specified
        self.hard_neg_ratio = hard_neg_ratio if hard_neg_ratio else neg_ratio
        
    def init_subs(self):
        return {
            "positive": [],
            "soft_negative": [],
            "hard_negative": [],
        }

    def sample(self):
        self.subs = self.init_subs()
        self.subs["positive"] = self.entailment_pairs()
        self.subs["soft_negative"] = self.contradiction_pairs(
            self.random_atomic_class_pair, max_neg_num=int(self.neg_ratio * len(self.subs["positive"]))
        )
        hard_neg_num = min(
            self.hard_neg_ratio * len(self.subs["positive"]), len(self.reasoner.sibling_pairs)
        )
        self.subs["hard_negative"] = self.contradiction_pairs(
            self.random_sibling_atomic_class_pair, max_neg_num=int(hard_neg_num)
        )
        assert not set(self.subs["positive"]).intersection(self.subs["soft_negative"])
        assert not set(self.subs["positive"]).intersection(self.subs["hard_negative"])

    def entailment_pairs(self):
        """Extract all atomic subsumption pairs that indicate entailment from left to right,
        i.e., if A is {left} => then A is {right}.
        """
        pbar = self.progress_manager.counter(desc="Extract Subsumption:", unit="pair")
        positives = []
        for cl_iri in self.reasoner.class_iris:
            owl_cl = self.reasoner.owlClasses[cl_iri]
            for subsumer_iri in self.reasoner.super_entities_of(owl_cl):
                positives.append(f"{cl_iri} <SubsumedBy> {subsumer_iri}")
                pbar.update()
        positives = list(set(sorted(positives)))
        print(f"In total {len(positives)} unique subsumption pairs are extracted.")
        return positives

    def contradiction_pairs(self, sample_a_pair: Callable, max_neg_num: int):
        """Sample false atomic subsumption pairs that indicate contradiction from left to right,
        i.e., if A is {left} !=> then A is {right}. Such pairs need to pass the negative sample
        check.
        """
        negatives = []
        max_iter = 2 * max_neg_num
        print(f"Sample false subsumption pairs with method: {sample_a_pair.__name__}.")
        # create two bars for process tracking
        added_bar = self.progress_manager.counter(
            total=max_neg_num, desc="Sampling False Subsumption", unit="pair"
        )
        iter_bar = self.progress_manager.counter(total=max_iter, desc="#Iteration", unit="it")
        i = 0
        added = 0
        while added < max_neg_num and i < max_iter:
            left_class_iri, right_class_iri = sample_a_pair()
            # skip if it doesn't draw a pair
            if not left_class_iri or not right_class_iri:
                i += 1
                iter_bar.update(1)
                continue
            left_class = self.reasoner.owlClasses[left_class_iri]
            right_class = self.reasoner.owlClasses[right_class_iri]
            # collect class iri if accepted
            if self.reasoner.check_negative_subsumption(left_class, right_class):
                neg = f"{left_class_iri} <NotSubsumedBy> {right_class_iri}"
                negatives.append(neg)
                added += 1
                added_bar.update(1)
                if added == max_neg_num:
                    negatives = list(set(sorted(negatives)))
                    added = len(negatives)
                    added_bar.count = added
            i += 1
            iter_bar.update(1)
        negatives = list(set(sorted(negatives)))
        print(f"In total {len(negatives)} unique false subsumption pairs are extracted.")
        return negatives

    def random_select_lab_pair(self, left_class_iri: str, right_class_iri: str):
        """[Auxiliary]: randomly select a pair of labels from two classes
        """
        left_labs = self.onto.iri2labs[left_class_iri]
        right_labs = self.onto.iri2labs[right_class_iri]
        # if either class labels are empty, neg_cands will be empty
        neg_cands = list(itertools.product(left_labs, right_labs))
        if neg_cands:
            return random.choice(neg_cands)
        else:
            return None
