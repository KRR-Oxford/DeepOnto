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
from IPython.display import Image
from anytree.dotexport import RenderTreeGraph
import math


ABBREVIATION_DICT = {
    "ObjectComplementOf": "[NEG]",  # negation
    "ObjectSomeValuesFrom": "[EX.]",  # existential restriction
    "ObjectAllValuesFrom": "[ALL]",  # universal restriction
    "ObjectUnionOf": "[OR.]",  # disjunction
    "ObjectIntersectionOf": "[AND]",  # conjunction
    "EquivalentClasses": "[EQV]",  # equivalence
    "SubClassOf": "[SUB]",  # subsumed by
    "SuperClassOf": "[SUP]",  # subsumes
}

class OntologyAxiomParser:
    r"""A syntactic parser for the OWL axioms (from [`org.semanticweb.owlapi.model.OWLAxiom`](http://owlcs.github.io/owlapi/apidocs_5/org/semanticweb/owlapi/model/OWLAxiom.html)).
    
    To keep the Java import in the main [`Ontology`][deeponto.onto.Ontology] class, 
    this parser does not deal with `OWLAxiom` directly but instead its **string representation**.
    
    Due to the OWLAPI axiom syntax, this parser relies on two components:
    
    1. Parentheses matching;
    2. Tree construction ([`RangeNode`][deeponto.onto.verbaliser.RangeNode]).
    
    As a result, it will return a [`RangeNode`][deeponto.onto.verbaliser.RangeNode] that
    specifies the sub-formulas (and their respective **positions in the string representation**) 
    in a tree structure.
    
    Examples:

        Suppose the input `OWLAxiom` has the string representation: 
        
        ```python
        >>> str(owl_axiom)
        >>> 'EquivalentClasses(<http://purl.obolibrary.org/obo/FOODON_00001707> ObjectIntersectionOf(<http://purl.obolibrary.org/obo/FOODON_00002044> ObjectSomeValuesFrom(<http://purl.obolibrary.org/obo/RO_0001000> <http://purl.obolibrary.org/obo/FOODON_03412116>)) )'
        ```
        
        After apply the parser, a [`RangeNode`][deeponto.onto.verbaliser.RangeNode] will be returned which can be rentered as:
        
        ```python
        axiom_parser = OntologyAxiomParser()
        axiom_parser.parse(str(owl_axiom)).render_tree()
        ```
        
        `#!console Output:`
        :   &#32;
            ```python
            Root@[0:inf]
            └── EQV@[0:212]
                ├── FOODON_00001707@[6:54]
                └── AND@[55:210]
                    ├── FOODON_00002044@[61:109]
                    └── EX.@[110:209]
                        ├── RO_0001000@[116:159]
                        └── FOODON_03412116@[160:208]
            ```
        
        Or, if `graphviz` (installed by e.g., `sudo apt install graphviz`) is available, 
        you can visualise the tree as an image by:
        
        ```python
        axiom_parser.parse(str(owl_axiom)).render_image()
        ```
        
        `#!console Output:`
        
        <p align="center">
            <img alt="range_node" src="../../../assets/example_range_node.png" style="padding: 30px 50px">
        </p>

        
        The name for each node has the form `{node_type}@[{start}:{end}]`, which means a node of the type `{node_type}` is
        located at the range `[{start}:{end}]` in the **abbreviated** axiom string  (see [`abbr_axiom_text`][deeponto.onto.verbaliser.OntologyAxiomParser.abbr_axiom_text] 
        below). 
        
        The leaf nodes are IRIs and they are represented by the last segment (split by `"/"`) of the whole IRI.
        
        Child nodes can be accessed by `.children`, the string representation of the sub-formula in this node can be 
        accessed by `.text`. For example:
        
        ```python
        parser.parse(str(owl_axiom)).children[0].children[1].text
        ```
        
        `#!console Output:`
        :   &#32;
            ```python
            '[AND](<http://purl.obolibrary.org/obo/FOODON_00002044> [EX.](<http://purl.obolibrary.org/obo/RO_0001000> <http://purl.obolibrary.org/obo/FOODON_03412116>))'
            ```
        
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
            "ObjectSomeValuesFrom": "[EX.]",  # existential restriction
            "ObjectAllValuesFrom": "[ALL]",  # universal restriction
            "ObjectUnionOf": "[OR.]",  # disjunction
            "ObjectIntersectionOf": "[AND]",  # conjunction
            "EquivalentClasses": "[EQV]",  # equivalence
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
        r"""Parse an `OWLAxiom` into a [`RangeNode`][deeponto.onto.verbaliser.RangeNode]. 
        
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
        r"""Parse an `OWLAxiom` based on parentheses matching into a [`RangeNode`][deeponto.onto.verbaliser.RangeNode]. 
        
        This function needs to be applied twice to get a fully parsed [`RangeNode`][deeponto.onto.verbaliser.RangeNode] because IRIs have
        a different parenthesis pattern.

        Args:
            axiom_text (str): The string representation of an OWLAPI axiom.
            already_parsed (RangeNode, optional): A partially parsed [`RangeNode`][deeponto.onto.verbaliser.RangeNode] to continue with. Defaults to `None`.
            for_iri (bool, optional): Parentheses are by default `()` but will be changed to `<>` for IRIs. Defaults to `False`.

        Raises:
            RuntimeError: Raised when the input axiom text is nor properly formatted.

        Returns:
            (RangeNode): A parsed syntactic tree given what parentheses to be matched. 
        """
        if not already_parsed:
            # a root node that covers the entire sentence
            parsed = RangeNode(0, math.inf, name=f"Root", text=axiom_text)
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
                        # the first character is actually "["
                        real_start = start - 5
                        axiom_type = axiom_text[real_start + 1: start - 1]
                        node = RangeNode(
                            real_start,
                            end + 1,
                            name=f"{axiom_type}",
                            text=axiom_text[real_start : end + 1],
                        )
                        parsed.insert_child(node)
                    else:
                        # no preceding characters for just atomic class (IRI)
                        abbr_iri = axiom_text[start : end + 1].split("/")[-1].rstrip(">")
                        node = RangeNode(
                            start, end + 1, name=abbr_iri, text=axiom_text[start : end + 1]
                        )
                        parsed.insert_child(node)
                except IndexError:
                    print("Too many closing parentheses")

        if stack:  # check if stack is empty afterwards
            raise RuntimeError("Too many opening parentheses")

        return parsed


class RangeNode(NodeMixin):
    r"""A tree implementation for ranges (without partial overlap).

        - Parent node's range fully covers child node's range, e.g., `[1, 10]` is a parent of `[2, 5]`.
        - Partial overlap between ranges are not allowed, e.g., `[2, 4]` and `[3, 5]` cannot appear in the same `RangeNodeTree`.
        - Non-overlap ranges are on different branches (irrelevant).
        - Child nodes are ordered according to their relative positions.
    """

    def __init__(self, start, end, name=None, **kwargs):
        if start >= end:
            raise RuntimeError("invalid start and end positions ...")
        self.start = start
        self.end = end
        self.name = "Root" if not name else name
        self.name = f"{self.name}@[{self.start}:{self.end}]"  # add start and ent to the name
        for k, v in kwargs.items():
            setattr(self, k, v)
        super().__init__()

    # def __eq__(self, other: RangeNode):
    #     """Two ranges are equal if they have the same `start` and `end`.
    #     """
    #     return self.start == other.start and self.end == other.end

    def __gt__(self, other: RangeNode):
        r"""Compare two ranges if they have a different `start` and/or a different `end`.
        
        - $R_1 \lt R_2$: if range $R_1$ is completely contained in range $R_2$, and $R_1 \neq R_2$. 
        - $R_1 \gt R_2$: if range $R_2$ is completely contained in range $R_1$,  and $R_1 \neq R_2$.
        - `"irrelevant"`: if range $R_1$ and range $R_2$ have no overlap.
        
        !!! warning
        
            Partial overlap is not allowed.
        """
        # ranges inside
        if self.start <= other.start and other.end <= self.end:
            return True
        
        # ranges outside
        if other.start <= self.start and self.end <= other.end:
            return False
        
        if other.end < self.start or self.end < other.start:
            return "irrelevant"

        raise RuntimeError("Compared ranges have a partial overlap.")

    @staticmethod
    def sort_by_start(nodes: List[RangeNode]):
        """A sorting function that sorts the nodes by their starting positions."""
        temp = {sib: sib.start for sib in nodes}
        return list(dict(sorted(temp.items(), key=lambda item: item[1])).keys())

    def insert_child(self, node: RangeNode):
        r"""Inserting a child [`RangeNode`][deeponto.onto.verbaliser.RangeNode].
        
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
                # NOTE: the equal case is when two nodes are exactly the same, no operation needed
            if not inserted:
                self.children = list(self.children) + [node]
                self.children = self.sort_by_start(self.children)
        else:
            node.parent = self
            self.children = [node]
    
    def __repr__(self):
        return f"{self.name}"

    def render_tree(self):
        """Render the whole tree.
        """
        return RenderTree(self)
        
    def render_image(self):
        """Calling this function will generate a temporary `range_node.png` file
        which will be displayed.
        
        To make this visualisation work, you need to install `graphviz` by, e.g.,
        
        ```bash
        sudo apt install graphviz
        ```
        """
        RenderTreeGraph(self).to_picture("range_node.png")
        return Image("range_node.png")
