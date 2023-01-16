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

import random
from typing import Optional
import re

from deeponto.onto import Ontology
from deeponto.onto.logic.parser.equiv_parser import IRI as IRI_Pattern
from deeponto.onto.logic.parser.equiv_parser import ALL_PATTERNS, OWLEquivAxiomParser
from . import SubsumptionSamplerBase

from org.semanticweb.owlapi.model import OWLObject  # type: ignore


class ComplexSubsumptionSampler(SubsumptionSamplerBase):
    def __init__(self, onto: Ontology, obj_prop_path: Optional[str] = None, neg_ratio: int = 1):
        super().__init__(onto, neg_ratio)
        self.parser = OWLEquivAxiomParser(onto.owl_path, obj_prop_path)
        self.equiv_axioms = []
        for pa in ALL_PATTERNS:
            self.equiv_axioms += [
                ea for ea in self.reasoner.equiv_axioms if self.parser.fit(pa, str(ea))
            ]
        self.equiv_axioms = list(set(self.equiv_axioms))
        # count how many equiv_axioms have been used as they fit the patterns
        self.equiv_percent = float(len(self.equiv_axioms) / len(self.reasoner.equiv_axioms))

    def init_subs(self):
        return {"positive": [], "negative": []}

    def sample(self, max_num_per_equiv: Optional[int] = None, is_test: bool = False, to_str: bool = True):
        self.subs = self.init_subs()
        added_bar = self.progress_manager.counter(
            total=len(self.equiv_axioms), desc="Sample Complex Subs for Equivs", unit="per equiv axiom"
        )
        for ea in self.equiv_axioms:
            pos_ea = self.entailment_pairs_from_equiv_axiom(ea)
            if max_num_per_equiv:
                num = min(len(pos_ea), max_num_per_equiv)  # to prevent ever slow negative checking
            else:
                num = len(pos_ea)
            pos_ea = list(random.sample(pos_ea, num))
            neg_ea = self.contradiction_pairs_from_equiv_axiom(ea, num)
            self.subs["positive"] += pos_ea
            self.subs["negative"] += neg_ea
            added_bar.update()
            if is_test:
                break
        if to_str:
            self.subs["positive"] = [str(x) for x in list(set(self.subs["positive"]))]
            self.subs["negative"] = [str(x) for x in list(set(self.subs["negative"]))]

    def entailment_pairs_from_equiv_axiom(self, equiv_axiom: OWLObject):
        """Extract subsumptions that involve the complex class part of an equivalence axiom:
        C ≡ ComplexC => SubClassOf(C) ⊑ ComplexC and ComplexC ⊑ SuperClassOf(C)
        """
        results = []
        subs = list(equiv_axiom.asOWLSubClassOfAxioms())
        atom = subs[0].getSubClass()
        comp = subs[0].getSuperClass()
        # Not sure [C ⊑ ComplexC, ComplexC ⊑ C] which is the first element
        try:
            atom.getIRI()
        except:
            atom, comp = comp, atom
            atom.getIRI()
        # ensure atom is correctly extracted
        assert self.reasoner.owlClasses[str(atom.getIRI())]
        for super_iri in self.reasoner.super_entities_of(atom):
            pos = self.reasoner.owlDataFactory.getOWLSubClassOfAxiom(
                comp, self.reasoner.getOWLObjectFromIRI(super_iri)
            )
            results.append(pos)
        for sub_iri in self.reasoner.sub_entities_of(atom):
            pos = self.reasoner.owlDataFactory.getOWLSubClassOfAxiom(
                self.reasoner.getOWLObjectFromIRI(sub_iri), comp
            )
            results.append(pos)
        results += subs  # the two subsumptions from equivalence axiom are positive samples as well
        return list(set(results))

    def contradiction_pairs_from_equiv_axiom(self, equiv_axiom: OWLObject, max_neg_num: int):
        """Extract negative subsumptions that involve the complex class part of an equivalence axiom:
        C ≡ ComplexC => randomly corrupt one of the IRI in it and pass the negative sample check
        """
        max_iter = (
            max_neg_num + 2
        )  # as this process is rather slow but the generated samples are often fine
        results = []
        i = 0
        while len(results) < max_neg_num and i < max_iter:
            corrupted_axiom = self.random_corrupt(equiv_axiom)
            subs = corrupted_axiom.asOWLSubClassOfAxioms()
            selected_sub = random.choice(list(subs))
            if self.reasoner.check_negative_subsumption(
                selected_sub.getSubClass(), selected_sub.getSuperClass()
            ):
                results.append(selected_sub)
        return list(set(results))

    def random_corrupt(self, axiom: OWLObject):
        """Randomly change an IRI in the input axiom and return a new one
        """
        replace = random.choice(re.findall(IRI_Pattern, str(axiom)))[1:-1]
        replacement = None
        if self.reasoner.owlClasses[replace]:
            replacement = self.random_atomic_class()
        elif self.reasoner.owlObjectProperties[replace]:
            replacement = self.random_object_property()
        else:
            # NOTE: to extend to other types of entities in future
            raise RuntimeError("Unknown type of Axiom")
        return self.reasoner.replace_entity(axiom, replace, replacement)

