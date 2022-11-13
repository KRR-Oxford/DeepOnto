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
"""Reasoner Class based on OWLAPI"""

import os
import itertools
from collections import defaultdict
from deeponto import init_jvm, OWL_THING, OWL_NOTHING, OWL_BOTTOM_OBJECT_PROP, OWL_TOP_OBJECT_PROP

init_jvm("2g")

from java.io import *  # type: ignore
from java.util import *  # type: ignore
from org.semanticweb.owlapi.apibinding import OWLManager  # type: ignore
from org.semanticweb.owlapi.model import IRI, OWLObject, OWLClassExpression, OWLObjectPropertyExpression  # type: ignore
from org.semanticweb.HermiT import ReasonerFactory  # type: ignore
from yacs.config import CfgNode


class OWLReasoner:
    def __init__(self, onto_path: str):
        print("Perform reasoning on the input ontology ...\n")
        # the ontology object based on OWLAPI
        self.owlManager = OWLManager.createOWLOntologyManager()
        self.owlPath = "file:///" + os.path.abspath(onto_path)
        self.owlOnto = self.owlManager.loadOntologyFromOntologyDocument(IRI.create(self.owlPath))
        # the reasoner based on OWLAPI
        self.reasonerFactory = ReasonerFactory()
        self.reasoner = self.reasonerFactory.createReasoner(self.owlOnto)
        # save the OWLAPI classes for convenience
        self.owlClasses, self.class_iris = self.getOWLObjects("Classes")
        self.class_iris_with_root = self.class_iris + [OWL_THING]
        # save the OWLAPI object properties for convenience
        self.owlObjectProperties, self.obj_prop_iris = self.getOWLObjects("ObjectProperties")
        # for creating axioms
        self.owlDataFactory = OWLManager.getOWLDataFactory()
        # determine which top and bottom to be used
        self.top_bottom_dict = CfgNode(
            {
                "Classes": {"TOP": OWL_THING, "BOTTOM": OWL_NOTHING},
                "ObjectProperties": {"TOP": OWL_TOP_OBJECT_PROP, "BOTTOM": OWL_BOTTOM_OBJECT_PROP},
            }
        )

        # hidden properties computed as needed
        self._equiv_axioms = None
        self._non_single_child_classes = None
        self._sib_pairs = None
        self._sid_dict = None

    def getOWLObjects(self, which: str):
        owlObjects = dict()
        source = getattr(self.owlOnto, f"get{which}InSignature")
        for cl in source():
            owlObjects[str(cl.getIRI())] = cl
        return owlObjects, list(owlObjects.keys())

    @property
    def OWLThing(self):
        roots = self.reasoner.getTopClassNode().getEntities()
        for r in roots:
            if str(r.getIRI()) == OWL_THING:
                return r

    @property
    def OWLNothing(self):
        bottoms = self.reasoner.getBottomClassNode().getEntities()
        for b in bottoms:
            if str(b.getIRI()) == OWL_NOTHING:
                return b

    @property
    def equiv_axioms(self):
        """Return all the equivalence axioms in an ontology
        NOTE: (checked with protege already)
        """
        if not self._equiv_axioms:
            self._equiv_axioms = []
            for cl in self.owlOnto.getClassesInSignature():
                self._equiv_axioms += self.owlOnto.getEquivalentClassesAxioms(cl)
            self._equiv_axioms = list(set(self._equiv_axioms))
        return self._equiv_axioms

    @property
    def sibling_pairs(self):
        if not self._sib_pairs:
            self._non_single_child_classes = dict()
            self._sib_pairs = []
            for cl_iri in self.class_iris_with_root:
                if cl_iri == OWL_THING:
                    owlClass = self.OWLThing
                else:
                    owlClass = self.owlClasses[cl_iri]
                children = self.subs_of(owlClass, direct=True)
                if len(children) >= 2:
                    self._non_single_child_classes[cl_iri] = children
                    self._sib_pairs += [
                        (x, y) for x, y in itertools.product(children, children) if x != y
                    ]  # all possible combinations excluding reflexive pairs
            self._sib_pairs = list(set(self._sib_pairs))
            # an additional sibling dictionary for customized (fixed one sample) sampling
            self._sib_dict = defaultdict(list)
            for l, r in self._sib_pairs:
                self._sib_dict[l].append(r)
                self._sib_dict[r].append(l)
            print(
                f"{len(self._non_single_child_classes)}/{len(self.owlClasses)+1} (including Owl:Thing) has multiple (direct and inferred) children ..."
            )
            print(f"In total there are {len(self._sib_pairs)} sibling class pairs")
        return self._sib_pairs

    @property
    def sibling_dict(self):
        if not self._sib_dict:
            self.sibling_pairs
        return self._sib_dict

    @staticmethod
    def determine(owlObjectExp: OWLObject, is_singular: bool = False):
        if isinstance(owlObjectExp, OWLClassExpression):
            return "Classes" if not is_singular else "Class"
        elif isinstance(owlObjectExp, OWLObjectPropertyExpression):
            return "ObjectProperties" if not is_singular else "ObjectProperty"
        else:
            # NOTE: add further options in future
            pass

    def supers_of(self, owlObjectExp: OWLClassExpression, direct: bool = False):
        """Return the atomic (named) superclasses of a given OWLAPI class, either direct or inferred
        """
        ent_type = self.determine(owlObjectExp)
        get_super = f"getSuper{ent_type}"
        TOP = self.top_bottom_dict[ent_type].TOP
        super_entities = getattr(self.reasoner, get_super)(owlObjectExp, direct).getFlattened()
        super_entity_iris = [str(s.getIRI()) for s in super_entities]
        # the root node is owl#Thing
        if TOP in super_entity_iris:
            super_entity_iris.remove(TOP)
        return super_entity_iris

    def subs_of(self, owlObjectExp: OWLClassExpression, direct: bool = False):
        """Return the atomic (named) superclasses of a given OWLAPI class, either direct or inferred
        """
        ent_type = self.determine(owlObjectExp)
        get_sub = f"getSub{ent_type}"
        BOTTOM = self.top_bottom_dict[ent_type].BOTTOM
        sub_entities = getattr(self.reasoner, get_sub)(owlObjectExp, direct).getFlattened()
        sub_entity_iris = [str(s.getIRI()) for s in sub_entities]
        # the root node is owl#Thing
        if BOTTOM in sub_entity_iris:
            sub_entity_iris.remove(BOTTOM)
        return sub_entity_iris

    def check_disjoint(self, owlObjectExp1: OWLObject, owlObjectExp2: OWLObject):
        """Check if two class expressions are disjoint according to the reasoner
        """
        ent_type = self.determine(owlObjectExp1)
        assert ent_type == self.determine(owlObjectExp2)
        disjoint_axiom = getattr(self.owlDataFactory, f"getOWLDisjoint{ent_type}Axiom")(
            [owlObjectExp1, owlObjectExp2]
        )
        return self.reasoner.isEntailed(disjoint_axiom)

    def check_subsumption(self, subOwlObject: OWLObject, superOwlObject: OWLObject):
        """Check if the first class is subsumed by the second class according to the reasoner
        """
        ent_type = self.determine(subOwlObject, is_singular=True)
        assert ent_type == self.determine(superOwlObject, is_singular=True)
        sub_axiom = getattr(self.owlDataFactory, f"getOWLSub{ent_type}OfAxiom")(subOwlObject, superOwlObject)
        return self.reasoner.isEntailed(sub_axiom)
    
    def check_common_descendants(self, owlObjectExp1: OWLObject, owlObjectExp2: OWLObject):
        """Check if two OWLObjects have a common decendant
        """
        ent_type = self.determine(owlObjectExp1)
        assert ent_type == self.determine(owlObjectExp2)
        subs1 = self.subs_of(owlObjectExp1)
        subs2 = self.subs_of(owlObjectExp2)
        return set(subs1).intersection(set(subs2))

    def check_negative_subsumption(self, owlObjectExp1: OWLObject, owlObjectExp2: OWLObject):
        """[Auxiliary]: sanity check for a given negative sample
        """
        ent_type = self.determine(owlObjectExp1)
        assert ent_type == self.determine(owlObjectExp2)
        accepted = False
        # NOTE: Test 1: check for disjointness (after reasoning)
        if self.check_disjoint(owlObjectExp1, owlObjectExp2):
            accepted = True
        else:
            # NOTE: Test 2: check any common descendants and mutual subsumption
            has_common_descendants = self.check_common_descendants(owlObjectExp1, owlObjectExp2)
            has_subsumption = self.check_subsumption(owlObjectExp1, owlObjectExp2) or self.check_subsumption(owlObjectExp2, owlObjectExp1)
            if (not has_common_descendants) and (not has_subsumption):
                accepted = True        
        return accepted
