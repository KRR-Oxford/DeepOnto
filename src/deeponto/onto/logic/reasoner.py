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
from deeponto import init_jvm
init_jvm("2g")

from java.io import *  # type: ignore
from java.util import *  # type: ignore
from org.semanticweb.owlapi.apibinding import OWLManager  # type: ignore
from org.semanticweb.owlapi.model import IRI # type: ignore
from org.semanticweb.HermiT import ReasonerFactory  # type: ignore


class OWLAPIReasoner:
    def __init__(self, onto_path: str):
        # the ontology object based on OWLAPI
        self.owlManager = OWLManager.createOWLOntologyManager()
        self.owlPath = "file:///" + os.path.abspath(onto_path)
        self.owlOnto = self.owlManager.loadOntologyFromOntologyDocument(IRI.create(self.owlPath))
        # the reasoner based on OWLAPI
        self.reasonerFactory = ReasonerFactory()
        self.reasoner = self.reasonerFactory.createReasoner(self.owlOnto)
        # save the OWLAPI classes for convenience
        self.owlClasses = dict()
        for cl in self.owlOnto.getClassesInSignature(True):
            self.owlClasses[str(cl.getIRI())] = cl
        # for creating axioms
        self.owlDataFactory = OWLManager.getOWLDataFactory()
        
    def owlClass_from_iri(self, iri: str):
        return self.owlClasses[iri]

    def superclasses_of(self, owlClass, direct: bool = False):
        """Return the named superclasses of a given OWLAPI class, either direct or inferred
        """
        superclasses = self.reasoner.getSuperClasses(owlClass, direct).getFlattened()
        superclass_iris = [str(s.getIRI()) for s in superclasses]
        return superclass_iris

    def subclasses_of(self, owlClass, direct: bool = False):
        """Return the named subclasses of a given OWLAPI class, either direct or inferred
        """
        subclasses = self.reasoner.getSubClasses(owlClass, direct).getFlattened()
        subclass_iris = [str(s.getIRI()) for s in subclasses]
        subclass_iris.remove("http://www.w3.org/2002/07/owl#Nothing")
        return subclass_iris

    def check_disjoint(self, owlClass1, owlClass2):
        """Check if two entity classes are disjoint according to the reasoner
        """
        disjoint_axiom = self.owlDataFactory.getOWLDisjointClassesAxiom([owlClass1, owlClass2])
        return self.reasoner.isEntailed(disjoint_axiom)
        
