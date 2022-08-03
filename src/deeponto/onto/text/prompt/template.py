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
"""Utilities for prompt learning based on OpenPrompt"""

from __future__ import annotations

from openprompt.data_utils import InputExample
from openprompt.prompts import ManualTemplate
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptForClassification, PromptForGeneration
from openprompt import PromptDataLoader
from openprompt.plms import load_plm

from typing import TYPE_CHECKING, Union, List, Optional, Callable, Dict

if TYPE_CHECKING:
    from deeponto.onto import Ontology
    from owlready2.entity import ThingClass
    from owlready2.class_construct import Restriction

from deeponto.onto.graph.graph_utils import *
import itertools


class OntoPattern:
    def __init__(self, pattern: Callable[[str, str], str]):
        self.pattern = pattern

    def __call__(self, text_a: str, text_b: str):
        return self.pattern(text_a, text_b)

    def __repr__(self):
        return self.pattern("<?X>", "<?Y>")


class OntoTemplate:
    def __init__(self, onto: Ontology):
        self.onto = onto
        self._gci_pattern = OntoPattern(lambda et, st: f"{et} is a kind of {st}")
        self._equiv_pattern = OntoPattern(lambda et1, et2: f"{et1} is the same as {et2}")

        # memorize used patterns
        self.patterns = {"gci": [], "equiv": []}
        self.patterns["gci"].append(self._gci_pattern)
        self.patterns["equiv"].append(self._equiv_pattern)

    @property
    def gci_pattern(self):
        return self._gci_pattern

    @gci_pattern.setter
    def gci_pattern(self, new_pattern: Optional[OntoPattern]):
        if new_pattern:
            self._gci_pattern = new_pattern
            self.patterns.gci.append(new_pattern)

    @property
    def equiv_pattern(self):
        return self._equiv_pattern

    @equiv_pattern.setter
    def equiv_pattern(self, new_pattern: Optional[OntoPattern]):
        if new_pattern:
            self._equiv_pattern = new_pattern
            self.patterns.equiv.append(new_pattern)

    def parse_named_class(self, ent: ThingClass) -> List[str]:
        """A named class is represented by all of its labels (aliases)
        """
        return self.onto.search_ent_labs(ent)

    def parse_onto_class(self, ent: Union[ThingClass, Restriction]) -> List[str]:
        """An ontology class is either a named class or a restriction,
        e.g., A.some(B). For the named class, return its aliases; for
        the complex axioms that involve restrictions, parse them using
        the DL2NL (descriptional logic to natural language text) tool.
        """
        if isinstance(ent, ThingClass):
            return self.parse_named_class(ent)
        else:
            raise NotImplementedError

    def parse_gci(
        self,
        ent: Union[ThingClass, Restriction],
        subsumer: Union[ThingClass, Restriction],
        gci_pattern: Optional[OntoPattern] = None,
    ) -> List[str]:
        """Transform a GCI (C ⊆ D) into natural language texts (a list
        as a class could have multiple descritpions)
        """
        # conduct subsumption check
        if not subsumer in ent.is_a:
            raise ValueError(f"{ent} does not have a subsumer {subsumer}!")
        # parse both the entity and its subsumer into NL texts
        ent_texts = self.parse_onto_class(ent)
        subsumer_texts = self.parse_onto_class(subsumer)
        gci_texts = []
        # default GCI text: "is a kind of"
        self.gci_pattern = gci_pattern
        for ent_t, sub_t in itertools.product(ent_texts, subsumer_texts):
            gci_text = self.gci_pattern(ent_t, sub_t)
            gci_texts.append(gci_text)
        return gci_texts

    # TODO: the main function to be adjusted
    def parse_contexts(
        self,
        ent: Union[ThingClass, Restriction],
        gci_pattern: OntoPattern = None,
        equiv_pattern: OntoPattern = None,
    ) -> Dict:
        """Providing ontology contexts for a given class and wrapping into NL texts.
        """
        # reset patterns if any new patterns
        self.gci_pattern = gci_pattern
        self.equiv_pattern = equiv_pattern
        # contexts are defined as below
        contexts = {"ent": [], "parent": [], "child": [], "branch": []}
        # texts for the entity class
        ent_texts = self.parse_onto_class(ent)
        for et1, et2 in itertools.product(ent_texts, ent_texts):
            contexts.ent.append(self.equiv_pattern(et1, et2))
        # texts for the parents of the entity class
        parent_texts = list(
            itertools.chain.from_iterable([self.parse_onto_class(p) for p in ent.is_a])
        )
        for et, pt in itertools.product(ent_texts, parent_texts):
            contexts.parent.append(self.gci_pattern(et, pt))
        # texts for the children of the entity class
        child_texts = list(
            itertools.chain.from_iterable([self.parse_onto_class(c) for c in ent.subclasses()])
        )
        for et, ct in itertools.product(ent_texts, child_texts):
            contexts.child.append(self.gci_pattern(ct, et))
        # texts for the branch information of the entity class
        branch_texts = list(
            itertools.chain.from_iterable([self.parse_onto_class(b) for b in branch_head_of(ent)])
        )
        for et, bt in itertools.product(ent_texts, branch_texts):
            contexts.branch.append(self.gci_pattern(et, bt))

        return contexts

    # TODO: the main function to be adjusted
    def present(
        self,
        ent: Union[ThingClass, Restriction],
        gci_pattern: OntoPattern = None,
        equiv_pattern: OntoPattern = None,
    ):
        ent_contexts = self.parse_contexts(ent, gci_pattern, equiv_pattern)

