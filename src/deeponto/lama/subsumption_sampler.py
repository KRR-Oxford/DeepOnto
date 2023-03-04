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
from abc import abstractmethod
import itertools
import random
from collections import defaultdict
from typing import Callable, Optional
import enlighten

from deeponto.onto import Ontology


class SubsumptionSamplerBase:
    """Base Class for Sampling Subsumption Pairs."""

    def __init__(self, onto: Ontology):
        self.onto = onto
        self.progress_manager = enlighten.get_manager()

        # for faster sampling
        self.concept_iris = list(self.onto.owl_classes.keys())
        self.object_property_iris = list(self.onto.owl_object_properties.keys())
        self.sibling_concept_groups = self.onto.sibling_class_groups
        self.sibling_auxiliary_dict = defaultdict(list)
        for i, sib_group in enumerate(self.sibling_concept_groups):
            for sib in sib_group:
                self.sibling_auxiliary_dict[sib].append(i)

    def random_named_concept(self) -> str:
        """Randomly draw a named concept's IRI."""
        return random.choice(self.concept_iris)

    def random_object_property(self) -> str:
        """Randomly draw a object property's IRI."""
        return random.choice(self.object_property_iris)

    def get_siblings(self, concept_iri: str):
        """Get the sibling concepts of the given concept."""
        sibling_group = self.sibling_auxiliary_dict[concept_iri]
        sibling_group = [self.sibling_concept_groups[i] for i in sibling_group]
        sibling_group = list(itertools.chain.from_iterable(sibling_group))
        return sibling_group

    def random_sibling(self, concept_iri: str) -> str:
        """Randomly draw a sibling concept for a given concept."""
        sibling_group = self.get_siblings(concept_iri)
        if sibling_group:
            return random.choice(sibling_group)
        else:
            # not every concept has a sibling concept
            return None

    @abstractmethod
    def positive_sampling(self, num_samples: Optional[int], *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def negative_sampling(self, num_samples: Optional[int], *args, **kwargs):
        raise NotImplementedError


class AtomicSubsumptionSampler(SubsumptionSamplerBase):
    """Sampler for constructing the Atomic Subsumption Inference (SI) dataset.
    """
    def __init__(self, onto: Ontology):
        super().__init__(onto)

        # compute the sibling concept pairs for faster hard negative sampling
        self.sibling_pairs = []
        for sib_group in self.sibling_concept_groups:
            self.sibling_pairs += [(x, y) for x, y in itertools.product(sib_group, sib_group) if x != y]
        self.maximum_num_hard_negatives = len(self.sibling_pairs)

    def positive_sampling(self, num_samples: Optional[int] = None):
        r"""Sample named concept pairs that are involved in a subsumption axiom.

        An extracted pair $(C, D)$ indicates $\mathcal{O} \models C \sqsubseteq D$ where
        $\mathcal{O}$ is the input ontology.
        """
        pbar = self.progress_manager.counter(desc="Sample Positive Subsumptions", unit="pair")
        positives = []
        for concept_iri in self.concept_iris:
            owl_concept = self.onto.owl_classes[concept_iri]
            for subsumer_iri in self.onto.reasoner.super_entities_of(owl_concept, direct=False):
                positives.append((concept_iri, subsumer_iri))
                pbar.update()
        positives = list(set(sorted(positives)))
        if num_samples:
            positives = random.sample(positives, num_samples)
        print(f"Sample {len(positives)} unique positive subsumption pairs.")
        return positives

    def negative_sampling(
        self,
        negative_sample_type: str,
        num_samples: int,
        apply_assumed_disjointness_alternative: bool = True,
    ):
        r"""Sample named concept pairs that are involved in a disjoiness (assumed) axiom, which then
        implies non-subsumption.

        See the definition of **Assumed Disjoiness** [here][deeponto.onto.OntologyReasoner.check_assumed_disjoint].
        """
        if negative_sample_type == "soft":
            draw_one = lambda: tuple(random.sample(self.concept_iris, k=2))
        elif negative_sample_type == "hard":
            draw_one = lambda: random.choice(self.sibling_pairs)
        else:
            raise RuntimeError(f"{negative_sample_type} not supported.")
        
        negatives = []
        max_iter = 2 * num_samples
        
        # which method to validate the negative sample
        valid_negative = self.onto.reasoner.check_assumed_disjoint
        if apply_assumed_disjointness_alternative:
            valid_negative = self.onto.reasoner.check_assumed_disjoint_alternative

        print(f"Sample {negative_sample_type} negative subsumption pairs.")
        # create two bars for process tracking
        added_bar = self.progress_manager.counter(total=num_samples, desc="Sample Negative Subsumptions", unit="pair")
        iter_bar = self.progress_manager.counter(total=max_iter, desc="#Iteration", unit="it")
        i = 0
        added = 0
        while added < num_samples and i < max_iter:
            sub_concept_iri, super_concept_iri = draw_one()
            sub_concept = self.onto.get_owl_object_from_iri(sub_concept_iri)
            super_concept = self.onto.get_owl_object_from_iri(super_concept_iri)
            # collect class iri if accepted
            if valid_negative(sub_concept, super_concept):
                neg = (sub_concept_iri, super_concept_iri)
                negatives.append(neg)
                added += 1
                added_bar.update(1)
                if added == num_samples:
                    negatives = list(set(sorted(negatives)))
                    added = len(negatives)
                    added_bar.count = added
            i += 1
            iter_bar.update(1)
        negatives = list(set(sorted(negatives)))
        print(f"Sample {len(negatives)} unique positive subsumption pairs.")
        return negatives
