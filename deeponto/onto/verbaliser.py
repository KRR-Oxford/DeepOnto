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
from __future__ import annotations

from typing import List, Optional
from anytree import NodeMixin, RenderTree


ABBREVIATION_DICT = {
    "ObjectComplementOf": "[NEG]",  # negation
    "ObjectSomeValuesFrom": "[EXT]",  # existential restriction
    "ObjectAllValuesFrom": "[ALL]",  # universal restriction
    "ObjectUnionOf": "[ORR]",  # disjunction
    "ObjectIntersectionOf": "[AND]",  # conjunction
    "EquivalentClasses": "[EQU]",  # equivalence
    "SubClassOf": "[SUB]",  # subsumed by
    "SuperClassOf": "[SUP]",  # subsumes
}


class OntologyAxiomParser:
    r"""A parser for the OWL axioms (from [`org.semanticweb.owlapi.model.OWLAxiom`](http://owlcs.github.io/owlapi/apidocs_5/org/semanticweb/owlapi/model/OWLAxiom.html)).
    
    To keep the Java import in the main [`Ontology`][deeponto.onto.Ontology] class, 
    this parser does not deal with `OWLAxiom` directly but instead its **string representation**.
    
    Due to the OWLAPI axiom syntax, this parser relies on two components:
    
    1. Parentheses matching;
    2. Tenary tree search ([`RangeNode`][deeponto.onto.verbaliser.RangeNode]).
    
    As a result, it will return a `RangeNode` that specifies the sub-formula in a tree structure.
    
    Examples:

        Suppose the input `OWLAxiom` has the string representation: 
        
        ```python
        >>> str(owl_axiom)
        >>> 'EquivalentClasses(<http://purl.obolibrary.org/obo/FOODON_00001083> ObjectIntersectionOf(<http://purl.obolibrary.org/obo/FOODON_00001133> ObjectSomeValuesFrom(<http://purl.obolibrary.org/obo/RO_0001000> <http://purl.obolibrary.org/obo/FOODON_03411012>)) )'
        ```
        
        After apply the parser, a `RangeNode` will be returned which can be printed as:
        
        ```python
        >>> axiom_parser = OntologyAxiomParser()
        >>> axiom_parser.parse(str(owl_axiom))
        >>> [0, [0, [6, 54], [55, [61, 109], [110, [116, 159], [160, 208], 209], 210], 212], 213]
        ```
        
        These numbers indicate the index ranges in the **abbreviated** (see `abbr_axiom_text`) axiom text 
        that correspond to a sub-formula. For example, the range `[6, 54]` corresponds to the IRI statement
        `<http://purl.obolibrary.org/obo/FOODON_00001083>`.
        
        To access this entity IRI, do:
        
        ```python
        >>> axiom_parser.parse(str(owl_axiom)).children[0].children[0].text
        >>> '<http://purl.obolibrary.org/obo/FOODON_00001083>'
        ```
        
        Note that the whole sentence is accessed by the first `.children[0]` which corresponds to the range
        `[0, 212]` (the root node the outermost range `[0, 213]`). Then, the first child `[0, 212]` is `[6, 54]`,
        whose text can be accessed by a subsequent `.children[0].text`. You can also see the type of any node
        by `node.type`. For this example, it will return `'IRI'`.
    """
    def __init__(self):
        pass

    def abbr_axiom_text(self, axiom_text: str):
        r"""Abbreviate the string representations of logical operators to a 
        fixed length (easier for parsing).
        
        The abbreviations are as follows:
        
        ```python
        {
            "ObjectComplementOf": "[NEG]",  # negation
            "ObjectSomeValuesFrom": "[EXT]",  # existential restriction
            "ObjectAllValuesFrom": "[ALL]",  # universal restriction
            "ObjectUnionOf": "[ORR]",  # disjunction
            "ObjectIntersectionOf": "[AND]",  # conjunction
            "EquivalentClasses": "[EQU]",  # equivalence
            "SubClassOf": "[SUB]",  # subsumed by
            "SuperClassOf": "[SUP]",  # subsumes
        }
        ```

        Args:
            axiom_text (str): The string representation of an OWLAPI axiom.

        Returns:
            (str): The modified string representation of an OWLAPI axiom where the logical operators are abbreviated.
        """
        for k, v in ABBREVIATION_DICT.items():
            axiom_text = axiom_text.replace(k, v)
        return axiom_text

    def parse(self, axiom_text: str) -> RangeNode:
        r"""Parse an `OWLAxiom` into a `RangeNode`. 
        
        This is the main entry for using the parser, which relies on the [`parse_by_parentheses`][deeponto.onto.verbaliser.OntologyAxiomParser.parse_by_parentheses]
        method below.

        Args:
            axiom_text (str): The string representation of an OWLAPI axiom.

        Returns:
            (RangeNode): A parsed syntactic tree given what parentheses to be matched. 
        """
        axiom_text = self.abbr_axiom_text(axiom_text)
        # print("To parse the following (transformed) axiom text:\n", axiom_text)
        # parse complex patterns first
        cur_parsed = self.parse_by_parentheses(axiom_text)
        # parse the IRI patterns latter
        return self.parse_by_parentheses(axiom_text, cur_parsed, for_iri=True)

    @classmethod
    def parse_by_parentheses(
        cls, axiom_text: str, already_parsed: RangeNode = None, for_iri: bool = False
    ) -> RangeNode:
        """Parse an `OWLAxiom` based on parentheses matching into a `RangeNode`. 
        
        This function needs to be applied twice to get a fully parsed `RangeNode` because IRIs have
        a different parenthesis pattern.

        Args:
            axiom_text (str): The string representation of an OWLAPI axiom.
            already_parsed (RangeNode, optional): A partially parsed `RangeNode` to continue with. Defaults to `None`.
            for_iri (bool, optional): Parentheses are by default `()` but will be changed to `<>` for IRIs. Defaults to `False`.

        Raises:
            RuntimeError: Raised when the input axiom text is nor properly formatted.

        Returns:
            (RangeNode): A parsed syntactic tree given what parentheses to be matched. 
        """
        if not already_parsed:
            # a root node that covers the entire sentence
            parsed = RangeNode(0, len(axiom_text) + 1, type="Root", text=axiom_text)
        else:
            parsed = already_parsed
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


class RangeNode(NodeMixin):
    """A tree implementation for ranges (without partial overlap).

        - parent node's range fully covers child node's range, e.g., `[1, 10]` is a parent of `[2, 5]`.
        - partial overlap between ranges not allowed, e.g., `[2, 4]` and `[3, 5]` cannot appear in the same `RangeNodeTree`.
        - non-overlap ranges are on different branches.
        - child nodes are ordered according to their relative positions.
    """

    def __init__(self, start, end, **kwargs):
        if start >= end:
            raise RuntimeError("invalid start and end positions ...")
        self.start = start
        self.end = end
        for k, v in kwargs.items():
            setattr(self, k, v)
        super().__init__()

    # def __eq__(self, other: RangeNode):
    #     return self.start == other.start and self.end == other.end

    def __gt__(self, other: RangeNode):
        r"""Modified compare function for a range.

        !!! note
        
            There are three kinds of comparisons:

            - $R_1 \leq R_2$: if range $R_1$ is completely contained in range $R_2$.
            - $R_1 \gt R_2$: if range $R_2$ is completely contained in range $R_1$.
            - `"irrelevant"`: if range $R_1$ and range $R_2$ have no overlap.

        NOTE that partial overlap is not allowed.
        """
        if other.start <= self.start and self.end <= other.end:
            return False
        elif self.start <= other.start and other.end <= self.end:
            return True
        elif other.end < self.start or self.end < other.start:
            # print("compared ranges are irrelevant ...")
            return "irrelevant"
        else:
            raise RuntimeError("compared ranges have partial overlap ...")

    @staticmethod
    def sort_by_start(nodes: List[RangeNode]):
        """A sorting function that sorts the nodes by their starting positions."""
        temp = {sib: sib.start for sib in nodes}
        return list(dict(sorted(temp.items(), key=lambda item: item[1])).keys())

    def insert_child(self, node: RangeNode):
        r"""Inserting a child `RangeNode`.
        
        Child nodes have a smaller (inclusive) range, e.g., `[2, 5]` is a child of `[1, 6]`.
        """
        if node > self:
            raise RuntimeError("invalid child node")
        if node.start == self.start and node.end == self.end:
            # duplicated node
            return
        # print(self.children)
        if self.children:
            inserted = False
            for ch in self.children:
                if (node < ch) is True:
                    # print("further down")
                    ch.insert_child(node)
                    inserted = True
                    break
                elif (node > ch) is True:
                    # print("insert in between")
                    ch.parent = node
                    # NOTE: should not break here as it could be parent of multiple children !
                    # break
            if not inserted:
                self.children = list(self.children) + [node]
                self.children = self.sort_by_start(self.children)
        else:
            node.parent = self
            self.children = [node]

    def __repr__(self):
        # only present downwards (down, left, right)
        printed = f"[{self.start}, {self.end}]"
        if self.children:
            printed = f"[{self.start}, {str(list(self.children))[1:-1]}, {self.end}]"
        return printed

    def print_tree(self):
        """Pretty printing in the tree structure."""
        print(RenderTree(self))
