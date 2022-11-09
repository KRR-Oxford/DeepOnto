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
import enlighten
from collections import defaultdict

from deeponto.onto import Ontology
from deeponto.onto.logic.reasoner import OWLReasoner
from deeponto import SavedObj, OWL_THING


class SubsumptionSampler:
    def __init__(self, onto_path: str, neg_ratio: int = 1, hard_neg_ratio: Optional[int] = None):
        self.reasoner = OWLReasoner(onto_path)
        self.onto = Ontology.from_new(onto_path)
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
            self.soft_neg_sample, max_neg_num=int(self.neg_ratio * len(self.subs["positive"]))
        )
        hard_neg_num = min(
            self.hard_neg_ratio * len(self.subs["positive"]), len(self.sibling_pairs)
        )
        self.subs["hard_negative"] = self.contradiction_pairs(
            self.hard_neg_sample, max_neg_num=int(hard_neg_num)
        )
        assert not set(self.subs["positive"]).intersection(self.subs["soft_negative"])
        assert not set(self.subs["positive"]).intersection(self.subs["hard_negative"])

    def save(self, saved_path: str):
        """Save in json format similar to the huggingface datasets manner
        """
        SavedObj.save_json(self.subs, saved_path)

    def entailment_pairs(self):
        """Extract all subsumption pairs that indicate entailment from left to right,
        i.e., if A is {left} => then A is {right}.
        """
        manager = enlighten.get_manager()
        pbar = manager.counter(desc="Extract Subsumption:", unit="pair")
        positives = []
        for cl_iri in self.class_iris:
            owl_cl = self.reasoner.owlClass_from_iri(cl_iri)
            for subsumer_iri in self.reasoner.superclasses_of(owl_cl):
                positives.append(f"{cl_iri} <SubsumedBy> {subsumer_iri}")
                pbar.update()
        positives = list(set(sorted(positives)))
        print(f"In total {len(positives)} unique subsumption pairs are extracted.")
        return positives

    def contradiction_pairs(self, sample_a_pair: Callable, max_neg_num: int):
        """Sample false subsumption pairs that indicate contradiction from left to right,
        i.e., if A is {left} !=> then A is {right}. Such pairs meet one of the two criteria:
            1. they are disjoint (after reasoning);
            2. they do not have a common descendant (including themselves, i.e., they do 
            not have a inferred subsumption relationship).
        """
        negatives = []
        max_iter = 2 * max_neg_num
        print(f"Sample false subsumption pairs with method: {sample_a_pair.__name__}.")
        # create two bars for process tracking
        manager = enlighten.get_manager()
        added_bar = manager.counter(
            total=max_neg_num, desc="Sampling False Subsumption", unit="pair"
        )
        iter_bar = manager.counter(total=max_iter, desc="#Iteration", unit="it")
        i = 0
        added = 0
        while added < max_neg_num and i < max_iter:
            left_class_iri, right_class_iri = sample_a_pair()
            # skip if it doesn't draw a pair
            if not left_class_iri or not right_class_iri:
                i += 1
                iter_bar.update(1)
                continue
            # collect class label if accepted
            if self.sanity_check(left_class_iri, right_class_iri):
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

    def soft_neg_sample(self):
        """[Auxiliary]: soft negatives are randomly sampled
        """
        return tuple(random.sample(self.class_iris, k=2))

    def hard_neg_sample(self):
        """[Auxiliary]: hard negatives are sampled from sibling classes (direct children of a parent class)
        """
        return tuple(random.choice(self.sibling_pairs))

    def sanity_check(self, left_class_iri: str, right_class_iri: str):
        """[Auxiliary]: sanity check for a given negative sample
        """
        accepted = False
        # get the owlapi class objects
        owl_left = self.reasoner.owlClass_from_iri(left_class_iri)
        owl_right = self.reasoner.owlClass_from_iri(right_class_iri)
        # NOTE: Test 1: check for disjointness (after reasoning)
        if self.reasoner.check_disjoint(owl_left, owl_right):
            accepted = True
        else:
            # NOTE: Test 2: check any common descendants including themselves (after reasoning)
            left_descendant_iris = self.reasoner.subclasses_of(owl_left)
            left_descendant_iris.append(left_class_iri)
            right_descendant_iris = self.reasoner.subclasses_of(owl_right)
            right_descendant_iris.append(right_class_iri)
            if not set(left_descendant_iris).intersection(set(right_descendant_iris)):
                accepted = True
        return accepted

    def neg_sample_for_class(self, class_iri: str, try_hard: bool = True):
        """Generate a customized negative sample (atomic classes) with one side fixed
        """
        accepted = False
        neg = None
        while not accepted:
            neg = None
            if try_hard:
                # print("try to sample a hard negative first ...")
                try:
                    neg = random.choice(self.sibling_dict[class_iri])
                except:
                    # print("no hard negative can be sampled ...")
                    pass
            if not neg:
                neg = random.choice(self.class_iris)
            if self.sanity_check(class_iri, neg):
                accepted = True
        return neg

    def neg_sample_for_object_property(self, property_iri: str):
        """Generate a customized negative sample (object property) with one side fixed
        """
        accepted = False
        neg = None
        while not accepted:
            prop = self.reasoner.owlObjectProperties[property_iri]
            prop_subs = self.reasoner.sub_object_properties_of(prop) + [property_iri]
            neg_iri = random.choice(self.obj_prop_iris)
            neg = self.reasoner.owlObjectProperties[neg_iri]
            neg_subs = self.reasoner.sub_object_properties_of(neg) + [neg_iri]
            if not set(prop_subs).intersection(set(neg_subs)):
                accepted = True
        return str(neg.getIRI())

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
