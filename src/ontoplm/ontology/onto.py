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
"""Class for handling the ontology in owlready2 format"""

from __future__ import annotations

from lxml import etree, builder
from typing import Optional, List
from owlready2 import get_ontology

from ontoplm import SavedObj
from ontoplm.ontology.onto_text.text_utils import ents_labs_from_props
from .onto_text import abbr_iri


class Ontology(SavedObj):
    def __init__(self, onto_path: str):
        self.owl = get_ontology(f"file://{onto_path}").load()

    @classmethod
    def from_new(cls, onto_path: str, lab_props: List[str] = ["label"]) -> Ontology:
        onto = cls(onto_path)
        # {class_iri: class_number}; {class_number: class_iri}
        onto.class2idx, onto.idx2class = onto.assign_class_numbers(onto.owl)
        # {class_number: [labels]}
        onto.lab_probs = lab_props
        onto.class2labs = ents_labs_from_props(onto.owl.classes(), onto.class2idx, onto.lab_probs)
        onto.num_classes = len(onto.class2idx)
        return onto

    @classmethod
    def from_saved(cls, saved_path: str) -> Optional[Ontology]:
        pass
    
    def save_instance(self, saved_path: str):
        
        return 

    def __str__(self) -> str:
        xml = builder.ElementMaker()
        name = type(self).__name__
        root = getattr(xml, name)(num_class=str(self.num_class))
        string = etree.tostring(root, pretty_print=True).decode()
        return string

    @staticmethod
    def assign_class_numbers(owl_onto):
        """ assign numbers for each class in an owlready2 ontology
        """
        cl_iris = [abbr_iri(cl.iri) for cl in owl_onto.classes()]
        cl_idx = list(range(len(cl_iris)))
        class2idx = dict(zip(cl_iris, cl_idx))
        idx2class = dict(zip(cl_idx, cl_iris))
        assert len(class2idx) == len(idx2class)
        return class2idx, idx2class
