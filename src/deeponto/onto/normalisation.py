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

from __future__ import annotations

from . import Ontology

from org.semanticweb.owlapi.model import IRI, OWLClassExpression  # type: ignore


class OntologyNormaliser:
    r"""Class for ontology normalisation.
    
    Attributes:
        onto (Ontology): The input ontology to be normalised.
        temp_super_class_index (Dict[OWLCLassExpression, OWLClass]): A dictionary in the form of `{complex_sub_class: temp_super_class}`, which means
            `temp_super_class` is created during the normalisation of a complex subsumption axiom that has `complex_sub_class` as the sub-class.
    """
    
    def __init__(self, onto: Ontology):
        self.onto = onto
        self.temp_super_class_index = dict()  # index of temporary super-classes for complex subsumption axioms' sub-classes.
        self._temp_class_num = 0
        
        
    def get_temp_super_class(self, complex_sub_class: OWLClassExpression):
        """Get the temporary super class of a complex class (which acts as the sub-class of some 
        complex subsumption axiom). Randomise a new IRI if not existed.
        """
        # assign a new temporary super class if not existed
        if not complex_sub_class in self.temp_super_class_index.keys():
            temp_class = self.onto.owl_data_factory.getOWLClass(IRI.create(f"http://TEMP_CLASS_{self._temp_class_num}"))
            self._temp_class_num += 1
            self.temp_super_class_index[complex_sub_class] = temp_class
            
        return self.temp_super_class_index[complex_sub_class]
        
    
    def split_complex_subsumption_axiom(self, complex_subsumption_axiom: OWLClassExpression):
        r"""Split a complex subsumption axiom $C \sqsubseteq D$ ($C$ and $D$ are both complex classes)
        into $C \sqsubseteq X$ and $X \sqsubseteq D$ where $X$ is a temporary class.
        
        If $C$ or $D$ is not complex, return a singleton set that contains the input axiom.

        Args:
            complex_subsumption_axiom (OWLClassExpression): A subsumption axiom with complex classes on both sides.
        Returns:
            (Set[OWLClassExpression]): A set of resulting axioms.
        """
        # check if this axiom is indeed a complex subsumption axiom
        # a complex class has no IRI
        is_complex_sub_class =  not self.onto.reasoner.has_iri(complex_subsumption_axiom.getSubClass())
        is_complex_super_class = not self.onto.reasoner.has_iri(complex_subsumption_axiom.getSuperClass())
        if is_complex_sub_class and is_complex_super_class:
            temp_class = self.onto.owl_data_factory.getOWLClass(IRI.create("http://TEMP_CLASS"))
        else:
            return set([complex_subsumption_axiom])
