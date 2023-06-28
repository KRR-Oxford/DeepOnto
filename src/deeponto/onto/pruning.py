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

import itertools
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from . import Ontology

from org.semanticweb.owlapi.util import OWLEntityRemover  # type: ignore
from java.util import Collections  # type: ignore

class OntologyPruner:
    r"""Class for in-place ontology pruning.
    
    Attributes:
        onto (Ontology): The input ontology to be pruned. Note that the pruning process is in-place.
    """
    
    def __init__(self, onto: Ontology):
        self.onto = onto
        self._pruning_applied = None
        
    def save_onto(self, save_path: str):
        """Save the pruned ontology file to the given path."""
        print(f"{self._pruning_applied} pruning algorithm has been applied.")
        print(f"Save the pruned ontology file to {save_path}.")
        return self.onto.save_onto(save_path)
        
    def prune(self, class_iris_to_be_removed: List[str]):
        r"""Apply ontology pruning while preserving the relevant hierarchy.

        !!! credit "paper"

            This refers to the ontology pruning algorithm introduced in the paper:
            [*Machine Learning-Friendly Biomedical Datasets for Equivalence and Subsumption Ontology Matching (ISWC 2022)*](https://link.springer.com/chapter/10.1007/978-3-031-19433-7_33).

        For each class $c$ to be pruned, subsumption axioms will be created between $c$'s parents and children so as to preserve the
        relevant hierarchy.

        Args:
            class_iris_to_be_removed (List[str]): Classes with IRIs in this list will be pruned and the relevant hierarchy will be repaired.
        """

        # create the subsumption axioms first
        for cl_iri in class_iris_to_be_removed:
            cl = self.onto.get_owl_object_from_iri(cl_iri)
            cl_parents = self.onto.get_asserted_parents(cl)
            cl_children = self.onto.get_asserted_children(cl)
            for parent, child in itertools.product(cl_parents, cl_children):
                sub_axiom = self.onto.owl_data_factory.getOWLSubClassOfAxiom(child, parent)
                self.onto.add_axiom(sub_axiom)

        # apply pruning
        class_remover = OWLEntityRemover(Collections.singleton(self.onto.owl_onto))
        for cl_iri in class_iris_to_be_removed:
            cl = self.onto.get_owl_object_from_iri(cl_iri)
            cl.accept(class_remover)
        self.onto.owl_manager.applyChanges(class_remover.getChanges())

        # remove IRIs in dictionaries?
        # TODO Test it
        
        # self._pruning_applied = "min_hierarchy"
