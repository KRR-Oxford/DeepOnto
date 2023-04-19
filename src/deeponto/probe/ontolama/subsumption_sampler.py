# Copyright 2021 Yuan He. All rights reserved.

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
import re

from org.semanticweb.owlapi.model import OWLAxiom  # type: ignore

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
    def positive_sampling(self, num_samples: Optional[int]):
        raise NotImplementedError

    @abstractmethod
    def negative_sampling(self, num_samples: Optional[int]):
        raise NotImplementedError


class AtomicSubsumptionSampler(SubsumptionSamplerBase):
    r"""Sampler for constructing the Atomic Subsumption Inference (SI) dataset.

    Positive samples come from the entailed subsumptions.

    Soft negative samples come from the pairs of randomly selected concepts, subject to
    passing the [assumed disjointness check][deeponto.onto.OntologyReasoner.check_assumed_disjoint].

    Hard negative samples come from the pairs of randomly selected *sibling* concepts, subject to
    passing the [assumed disjointness check][deeponto.onto.OntologyReasoner.check_assumed_disjoint].
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
            for subsumer_iri in self.onto.reasoner.get_inferred_super_entities(owl_concept, direct=False):
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


IRI = "<https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)>"

class ComplexSubsumptionSampler(SubsumptionSamplerBase):
    r"""Sampler for constructing the Complex Subsumption Inference (SI) dataset.

    To obtain complex concept expressions on both sides of the subsumption relationship
    (as a sub-concept or a super-concept), this sampler utilises the equivalence axioms
    in the form of $C \equiv C_{comp}$ where $C$ is atomic and $C_{comp}$ is complex.

    An equivalence axiom like $C \equiv C_{comp}$ is deemed as an **anchor axiom**.

    Positive samples are in the form of $C_{sub} \sqsubseteq C_{comp}$ or $C_{comp} \sqsubseteq C_{super}$
    where $C_{sub}$ is an entailed sub-concept of $C$ and $C_{comp}$, $C_{super}$ is an entailed super-concept
    of $C$ and $C_{comp}$.

    Negative samples are formed by replacing one of the named entities in the anchor axiom, the modified
    sub-concept and super-concept need to pass the [assumed disjointness check][deeponto.onto.OntologyReasoner.check_assumed_disjoint]
    to be accepted as a valid negative sample. Without loss of generality, suppose we choose $C \sqsubseteq C_{comp}$
    and replace a named entity in $C_{comp}'$ to form $C \sqsubseteq C_{comp}'$, then $C$ and $C_{comp}'$ is a valid
    negative only if they satisfy the assumed disjointness check.
    """

    def __init__(self, onto: Ontology):
        super().__init__(onto)
        self.anchor_axioms = self.onto.get_equivalence_axioms("Classes")

    def positive_sampling_from_anchor(self, anchor_axiom: OWLAxiom):
        """Returns all positive subsumption pairs extracted from an anchor equivalence axiom."""
        sub_axiom = list(anchor_axiom.asOWLSubClassOfAxioms())[0]
        atomic_concept, complex_concept = sub_axiom.getSubClass(), sub_axiom.getSuperClass()
        # determine which is the atomic concept
        if complex_concept.isClassExpressionLiteral():
            atomic_concept, complex_concept = complex_concept, atomic_concept

        # intialise the positive samples from the anchor equivalence axiom
        positives = list(anchor_axiom.asOWLSubClassOfAxioms())
        for super_concept_iri in self.onto.reasoner.get_inferred_super_entities(atomic_concept, direct=False):
            positives.append(
                self.onto.owl_data_factory.getOWLSubClassOfAxiom(
                    complex_concept, self.onto.get_owl_object_from_iri(super_concept_iri)
                )
            )
        for sub_concept_iri in self.onto.reasoner.get_inferred_sub_entities(atomic_concept, direct=False):
            positives.append(
                self.onto.owl_data_factory.getOWLSubClassOfAxiom(
                    self.onto.get_owl_object_from_iri(sub_concept_iri), complex_concept
                )
            )
        
        # TESTING
        # for p in positives:
        #     assert self.onto.reasoner.owl_reasoner.isEntailed(p)    
        
        return list(set(sorted(positives)))

    def positive_sampling(self, num_samples_per_anchor: Optional[int] = 10):
        r"""Sample positive subsumption axioms that involve one atomic and one complex concepts.

        An extracted pair $(C, D)$ indicates $\mathcal{O} \models C \sqsubseteq D$ where
        $\mathcal{O}$ is the input ontology.
        """
        print(f"Maximum number of positive samples for each anchor is set to {num_samples_per_anchor}.")
        pbar = self.progress_manager.counter(desc="Sample Positive Subsumptions from", unit="anchor axiom")
        positives = dict()
        for anchor in self.anchor_axioms:
            positives_from_anchor = self.positive_sampling_from_anchor(anchor)
            if num_samples_per_anchor and num_samples_per_anchor < len(positives_from_anchor):
                positives_from_anchor = random.sample(positives_from_anchor, k = num_samples_per_anchor)
            positives[str(anchor)] = positives_from_anchor
            pbar.update()
        # positives = list(set(sorted(positives)))
        print(f"Sample {sum([len(v) for v in positives.values()])} unique positive subsumption pairs.")
        return positives
    
    def negative_sampling(self, num_samples_per_anchor: Optional[int] = 10):
        r"""Sample negative subsumption axioms that involve one atomic and one complex concepts.

        An extracted pair $(C, D)$ indicates $C$ and $D$ pass the [assumed disjointness check][deeponto.onto.OntologyReasoner.check_assumed_disjoint].
        """
        print(f"Maximum number of negative samples for each anchor is set to {num_samples_per_anchor}.")
        pbar = self.progress_manager.counter(desc="Sample Negative Subsumptions from", unit="anchor axiom")
        negatives = dict()
        for anchor in self.anchor_axioms:
            negatives_from_anchor = []
            i, max_iter = 0, num_samples_per_anchor + 2
            while i < max_iter and len(negatives_from_anchor) < num_samples_per_anchor:
                corrupted_anchor = self.random_corrupt(anchor)
                corrupted_sub_axiom = random.choice(list(corrupted_anchor.asOWLSubClassOfAxioms()))
                sub_concept, super_concept = corrupted_sub_axiom.getSubClass(), corrupted_sub_axiom.getSuperClass()
                if self.onto.reasoner.check_assumed_disjoint_alternative(sub_concept, super_concept):
                    negatives_from_anchor.append(corrupted_sub_axiom)
                i += 1
            negatives[str(anchor)] = list(set(sorted(negatives_from_anchor)))
            pbar.update()
        # negatives = list(set(sorted(negatives)))
        print(f"Sample {sum([len(v) for v in negatives.values()])} unique positive subsumption pairs.")
        return negatives

    def random_corrupt(self, axiom: OWLAxiom):
        """Randomly change an IRI in the input axiom and return a new one.
        """
        replaced_iri = random.choice(re.findall(IRI, str(axiom)))[1:-1]
        replaced_entity = self.onto.get_owl_object_from_iri(replaced_iri)
        replacement_iri = None
        if self.onto.get_entity_type(replaced_entity) == "Classes":
            replacement_iri = self.random_named_concept()
        elif self.onto.get_entity_type(replaced_entity) == "ObjectProperties":
            replacement_iri = self.random_object_property()
        else:
            # NOTE: to extend to other types of entities in future
            raise RuntimeError("Unknown type of axiom.")
        return self.onto.replace_entity(axiom, replaced_iri, replacement_iri)
