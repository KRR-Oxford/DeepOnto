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
"""A parser for the OWL axioms (from OWLAPI) based on
        1. parentheses matching;
        2. tenary tree search

E.g., 'EquivalentClasses(<http://purl.obolibrary.org/obo/FOODON_00001083> ObjectIntersectionOf(<http://purl.obolibrary.org/obo/FOODON_00001133> ObjectSomeValuesFrom(<http://purl.obolibrary.org/obo/RO_0001000> <http://purl.obolibrary.org/obo/FOODON_03411012>)) )'
====> [0, [0, [6, 54], [55, [61, 109], [110, [116, 159], [160, 208], 209], 210], 212], 213]
"""

from deeponto.utils.tree import RangeNode


class OWLAxiomParserBase:
    def __init__(self):
        # abbrevations all have length 5
        self.abbr = {
            "ObjectComplementOf": "[NEG]",
            "ObjectSomeValuesFrom": "[EXT]",
            "ObjectAllValuesFrom": "[ALL]",
            "ObjectUnionOf": "[ORR]",
            "ObjectIntersectionOf": "[AND]",
            "EquivalentClasses": "[EQU]",
            "SubClassOf": "[SUB]",  # self-defined not confirmed in OWLAPI yet
            "SuperClassOf": "[SUP]", # self-defined not confirmed in OWLAPI yet
        }

    def abbr_axiom_text(self, axiom_text: str):
        """Abbreviate logical operand expressions in an OWLAPI axiom text

        Parameters
        ----------
        axiom_text : str
            string representation of an OWLAPI axiom

        Returns
        -------
        str
            modified expression of an OWLAPI axiom where the logical operands are abbreviated
        """
        for k, v in self.abbr.items():
            axiom_text = axiom_text.replace(k, v)
        return axiom_text

    def parse(self, axiom_text: str) -> RangeNode:
        """Parse the OWL Axiom based on the tenary tree search (RangeNode) and the 
        parentheses matching algorithms

        Parameters
        ----------
        axiom_text : str
            string representation of an OWLAPI axiom

        Returns
        -------
        RangeNode
            a tree stucture parsed from the OWLAPI axiom text
        """
        axiom_text = self.abbr_axiom_text(axiom_text)
        # print("To parse the following (transformed) axiom text:\n", axiom_text)
        # parse complex patterns first
        cur_parsed = self._parse(axiom_text)
        # parse the IRI patterns latter
        return self._parse(axiom_text, cur_parsed, for_iri=True)

    @classmethod
    def _parse(
        cls, axiom_text: str, cur_parsed: RangeNode = None, for_iri: bool = False
    ) -> RangeNode:
        """Parse the OWL Axiom based on the tenary tree search (RangeNode) and the 
        parentheses matching algorithms

        Parameters
        ----------
        axiom_text : str
            string representation of an OWLAPI axiom
        cur_parsed : RangeNode, optional
            intermediate tree structure parsed from an OWLAPI axiom text, by default None
        for_iri : bool, optional
            parse regarding the IRI strings instead of the logical operands or not, by default False

        Returns
        -------
        RangeNode
            a tree stucture parsed from the OWLAPI axiom text

        Raises
        ------
        RuntimeError
            raise if the OWLAPI axiom text is invalid 
        """
        if not cur_parsed:
            # a root node that covers the entire sentence
            parsed = RangeNode(0, len(axiom_text) + 1, type="Root", text=axiom_text)
        else:
            parsed = cur_parsed
        stack = []
        left_par = "("
        right_par = ")"
        if for_iri:
            left_par = "<"
            right_par = ">"

        for i, c in enumerate(axiom_text):
            if c == left_par:
                stack.append(i)
            if c == right_par:
                try:
                    start = stack.pop()
                    end = i
                    if not for_iri:
                        # the first five characters refer to the complex pattern type
                        node = RangeNode(
                            start - 5,
                            end + 1,
                            type=axiom_text[start - 5 : start],
                            text=axiom_text[start - 5 : end + 1],
                        )
                        parsed.insert_child(node)
                    else:
                        # no preceding characters for just atomic class (IRI)
                        node = RangeNode(
                            start, end + 1, type="IRI", text=axiom_text[start : end + 1]
                        )
                        parsed.insert_child(node)
                except IndexError:
                    print("Too many closing parentheses")

        if stack:  # check if stack is empty afterwards
            raise RuntimeError("Too many opening parentheses")

        return parsed
