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
from deeponto import init_jvm, OWL_THING, OWL_NOTHING, OWL_BOTTOM_OBJECT_PROP, OWL_TOP_OBJECT_PROP

init_jvm("2g")

from java.io import *  # type: ignore
from java.util import *  # type: ignore
from org.semanticweb.owlapi.apibinding import OWLManager  # type: ignore
from org.semanticweb.owlapi.model import IRI  # type: ignore
from org.semanticweb.HermiT import ReasonerFactory  # type: ignore


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
        self.owlClasses = dict()
        for cl in self.owlOnto.getClassesInSignature():
            self.owlClasses[str(cl.getIRI())] = cl
        # save the OWLAPI object properties for convenience
        self.owlObjectProperties = dict()
        for obj in self.owlOnto.getObjectPropertiesInSignature():
            if str(obj.getIRI()) != OWL_TOP_OBJECT_PROP:
                self.owlObjectProperties[str(obj.getIRI())] = obj
        # for creating axioms
        self.owlDataFactory = OWLManager.getOWLDataFactory()

    def owlClass_from_iri(self, iri: str):
        try:
            return self.owlClasses[iri]
        except:
            if iri == OWL_THING:
                return self.OWLThing
            else:
                raise ValueError(f"Class IRI {iri} not found ...")

    @property
    def OWLThing(self):
        roots = self.reasoner.getTopClassNode().getEntities()
        for r in roots:
            if str(r.getIRI()) == OWL_THING:
                return r

    @property
    def equiv_axioms(self):
        """Return all the equivalence axioms in an ontology
        NOTE: (checked with protege already)
        """
        equivs = []
        for cl in self.owlOnto.getClassesInSignature():
            equivs += [str(x) for x in self.owlOnto.getEquivalentClassesAxioms(cl)]
        return list(set(equivs))

    def superclasses_of(self, owlClass, direct: bool = False):
        """Return the atomic (named) superclasses of a given OWLAPI class, either direct or inferred
        """
        superclasses = self.reasoner.getSuperClasses(owlClass, direct).getFlattened()
        superclass_iris = [str(s.getIRI()) for s in superclasses]
        # the root node is owl#Thing
        if OWL_THING in superclass_iris:
            superclass_iris.remove(OWL_THING)
        return superclass_iris

    def subclasses_of(self, owlClass, direct: bool = False):
        """Return the atomic (named) subclasses of a given OWLAPI class, either direct or inferred
        """
        subclasses = self.reasoner.getSubClasses(owlClass, direct).getFlattened()
        subclass_iris = [str(s.getIRI()) for s in subclasses]
        # the leaf node is owl#Nothing
        if OWL_NOTHING in subclass_iris:
            subclass_iris.remove(OWL_NOTHING)
        return subclass_iris

    def super_object_properties_of(self, owlObjProp, direct: bool = False):
        """Return the super-object properties of a given OWLAPI object property, either direct or inferred
        """
        super_obj_props = self.reasoner.getSuperObjectProperties(owlObjProp, direct).getFlattened()
        super_obj_iris = [str(s.getIRI()) for s in super_obj_props]
        # the root node is owl#TopObjectProperty
        if OWL_TOP_OBJECT_PROP in super_obj_iris:
            super_obj_iris.remove(OWL_TOP_OBJECT_PROP)
        return super_obj_iris

    def sub_object_properties_of(self, owlObjProp, direct: bool = False):
        """Return the sub-object properties of a given OWLAPI object property, either direct or inferred
        """
        sub_obj_props = self.reasoner.getSubObjectProperties(owlObjProp, direct).getFlattened()
        sub_obj_iris = [str(s.getIRI()) for s in sub_obj_props]
        # the leaf node is owl#BottomObjectProperty
        if OWL_BOTTOM_OBJECT_PROP in sub_obj_iris:
            sub_obj_iris.remove(OWL_BOTTOM_OBJECT_PROP)
        return sub_obj_iris

    def check_disjoint(self, owlClass1, owlClass2):
        """Check if two entity classes are disjoint according to the reasoner
        """
        disjoint_axiom = self.owlDataFactory.getOWLDisjointClassesAxiom([owlClass1, owlClass2])
        return self.reasoner.isEntailed(disjoint_axiom)
