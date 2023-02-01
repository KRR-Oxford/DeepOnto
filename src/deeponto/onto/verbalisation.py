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

import os
import spacy
from typing import List, Union
from collections import defaultdict
from anytree import NodeMixin, RenderTree
from IPython.display import Image
from anytree.dotexport import RenderTreeGraph
import math
from yacs.config import CfgNode

from . import Ontology
from org.semanticweb.owlapi.model import OWLObject, OWLClassExpression, OWLAxiom  # type: ignore


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

RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"


# download en_core_web_sm for object property 
try:
    spacy.load("en_core_web_sm")
except:
    print("Download `en_core_web_sm` for pos tagger.")
    os.system("python -m spacy download en_core_web_sm")


class OntologyVerbaliser:
    r"""A rule-based natural language verbaliser for the OWL logical expressions, e.g., [`OWLAxiom`](http://owlcs.github.io/owlapi/apidocs_5/org/semanticweb/owlapi/model/OWLAxiom.html)
    and [`OWLClassExpression`](https://owlcs.github.io/owlapi/apidocs_4/org/semanticweb/owlapi/model/OWLClassExpression.html).

    This is not a full-fledged ontology verbaliser but can already deal with most kinds of class expressions.

    Attributes:
        onto (Ontology): An ontology whose entities are to be verbalised.
        parser (OntologySyntaxParser): A syntax parser for the string representation of an `OWLObject`.
        vocab (Dict[str, List[str]]): A dictionary with `(entity_iri, entity_name)` pairs, by default
            the names are retrieved from $\texttt{rdfs:label}$.
    """

    def __init__(self, onto: Ontology):
        self.onto = onto
        self.parser = OntologySyntaxParser()
        self.nlp = spacy.load("en_core_web_sm")

        # build the default vocabulary for entities
        self.vocab = dict()
        for entity_type in ["Classes", "ObjectProperties", "DataProperties"]:
            entity_annotations, _ = self.onto.build_annotation_index(entity_type=entity_type, apply_lowercasing=False)
            self.vocab.update(**entity_annotations)
        literal_or_iri = lambda k, v: list(v)[0] if v else k  # set vocab to IRI if no string available
        self.vocab = {k: literal_or_iri(k, v) for k, v in self.vocab.items()}  # only set one name for each entity

    def update_entity_name(self, entity_iri: str, entity_name: str):
        """Update the name of an entity in `self.vocab`.

        If you want to change the name of a specific entity, you should call this
        function before applying verbalisation.
        """
        self.vocab[entity_iri] = entity_name

    def verbalise_class_expression(self, class_expression: Union[OWLClassExpression, RangeNode]):
        r"""Verbalise a class expression (`OWLClassExpression`) or its parsed form (in `RangeNode`).
        
        Currently supported types of class expressions are:
        
        - **Atomic** (named) classes, which have IRIs;
        - **Negation** (by simply adding a **"not"**);
        - **Existential** ($\exists r.C$) and **Universal** ($\forall r.C$) restrictions;
        - **Conjunction** ($C \sqcap D \sqcap ...$) and **Disjunction** ($C \sqcup D \sqcup ...$) statements.
        
        Note that the supported types can be **nested to any level** (i.e., $C$ and $D$ can be complex class expressions). 
        
        Extra cares are imposed for more natural verbalisation and less redundancy:
        
        - merging restriction statements that have the **same object property**;
        - if **object property** that starts with an **adjective** or a **noun**, add **"it"** to the head;
        - the keyword **"only"** is inserted between the object property and the class expression for an **universal restriction** statement;
        - the keyword **"some"** is inserted between the object property and the class expression for an **existential restriction** statement;
        - in a conjunction or disjunction statement, if there is **no atomic class**, then the keyword for the 
        anonymous class **"something"** is appended to the head.

        Args:
            class_expression (Union[OWLClassExpression, RangeNode]): A class expression to be verbalised.

        Raises:
            RuntimeError: Occurs when the class expression is not in one of the supported types.

        Returns:
            (CfgNode): A nested dictionary that presents the details of verbalisation. The verbalised string
                can be accessed with the key `["verbal"]`.
        """

        if not isinstance(class_expression, RangeNode):
            parsed_class_expression = self.parser.parse(class_expression).children[0]  # skip the root node
        else:
            parsed_class_expression = class_expression

        # for a singleton IRI
        if parsed_class_expression.is_iri:
            iri = parsed_class_expression.text.lstrip("<").rstrip(">")
            return CfgNode({"verbal": self.vocab[iri], "iri": iri, "type": "IRI"})
        
        if parsed_class_expression.name.startswith("NEG"):
            # negation only has one child
            cl = self.verbalise_class_expression(parsed_class_expression.children[0])
            return CfgNode({"verbal": "not " + cl.verbal, "class": cl, "type": "NEG"})

        # for existential and universal restrictions
        if parsed_class_expression.name.startswith("EX.") or parsed_class_expression.name.startswith("ALL"):
            return self._verbalise_restriction(parsed_class_expression)

        # for conjunction and disjunction
        if parsed_class_expression.name.startswith("AND") or parsed_class_expression.name.startswith("OR"):
            return self._verbalise_junction(parsed_class_expression)

        raise RuntimeError("Input class expression is not in one of the supported types.")

    def _verbalise_restriction(self, restriction_node: RangeNode, add_something: bool = True):
        """Verbalise a (parsed) class expression in the form of existential or universal restriction."""

        try:
            assert restriction_node.name.startswith("EX.") or restriction_node.name.startswith("ALL")
            assert len(restriction_node.children) == 2
        except:
            raise RuntimeError("Input range node is not related to a existential or universal restriction statement.")
            
        quantifier_word = "some" if restriction_node.name.startswith("EX.") else "only"

        object_property = restriction_node.children[0]
        assert object_property.is_iri
        object_property = self.verbalise_class_expression(object_property)
        
        
        # NOTE modify the object property's verbalisation with rules
        doc = self.nlp(object_property.verbal)
        # Rule 1. Add "is" if the object property starts with a NOUN, ADJ, or passive VERB
        if doc[0].pos_ == 'NOUN' or doc[0].pos_ == 'ADJ' or (doc[0].pos_ == 'VERB' and doc[0].text.endswith("ed")):
            object_property.verbal = "is " + object_property.verbal  
        
        

        class_expression = restriction_node.children[1]
        class_expression = self.verbalise_class_expression(class_expression.text)

        verbal = f"{object_property.verbal} {quantifier_word} {class_expression.verbal}"
            
        verbal = verbal.lstrip()
        if add_something:
            verbal = f"something that " + verbal

        return CfgNode(
            {
                "verbal": verbal,
                "property": object_property,
                "class": class_expression,
                "type": restriction_node.name[:3],
            }
        )

    def _verbalise_junction(self, junction_node: RangeNode):
        """Verbalise a (parsed) class expression in the form of conjunction or disjunction."""

        try:
            assert junction_node.name.startswith("AND") or junction_node.name.startswith("OR.")
        except:
            raise RuntimeError("Input range node is not related to a conjunction or disjunction statement.")
            
        junction_word = "and" if junction_node.name.startswith("AND") else "or"

        # collect restriction nodes for merging
        existential_restriction_children = defaultdict(list)
        universal_restriction_children = defaultdict(list)
        other_children = []
        for child in junction_node.children:
            if child.name.startswith("EX."):
                child = self._verbalise_restriction(child, add_something=False)
                existential_restriction_children[child.property.verbal].append(child)
            elif child.name.startswith("ALL"):
                child = self._verbalise_restriction(child, add_something=False)
                universal_restriction_children[child.property.verbal].append(child)
            else:
                other_children.append(self.verbalise_class_expression(child))

        merged_children = []
        for v in list(existential_restriction_children.values()) + list(universal_restriction_children.values()):
            # restriction = v[0].type
            if len(v) > 1:
                merged_child = CfgNode(dict())
                merged_child.update(v[0])  # initialised with the first one
                merged_child["class"] = CfgNode(
                    {"verbal": v[0]["class"].verbal, "classes": [v[0]["class"]], "type": junction_node.name[:3]}
                )

                for i in range(1, len(v)):
                    # v 0.5.2 fix for len(v) > 1 case
                    merged_child.verbal += f" {junction_word} " + v[i]["class"].verbal  # update grouped concepts with property
                    merged_child["class"].verbal += f" {junction_word} " + v[i]["class"].verbal  # update grouped concepts
                    merged_child["class"].classes.append(v[i]["class"])
                merged_children.append(merged_child)
                # print(merged_children)
            else:
                merged_children.append(v[0])

        results = CfgNode(
            {
                "verbal": "",
                "classes": other_children + merged_children,
                "type": junction_node.name[:3],
            }
        )

        # add the preceeding "something that" if there are only restrictions
        if not other_children:
            results.verbal += " something that " + f" {junction_word} ".join(
                c.verbal for c in merged_children
            )
            results.verbal.lstrip()
        else:
            results.verbal += f" {junction_word} ".join(c.verbal for c in other_children)
            if merged_children:
                if junction_word == "and":
                    # sea food and non-vergetarian product that derives from shark and goldfish
                    results.verbal += " that " + f" {junction_word} ".join(c.verbal for c in merged_children)
                elif junction_word == "or":
                    # sea food or non-vergetarian product or something that derives from shark or goldfish
                    results.verbal += " or something that " + f" {junction_word} ".join(c.verbal for c in merged_children)

        return results

    def verbalise_equivalence_axiom(self, equivalence_axiom: OWLAxiom):
        #TODO
        pass

    def verbalise_subsumption_axiom(self, subclassof_axiom: OWLAxiom):
        #TODO
        pass


class OntologySyntaxParser:
    r"""A syntax parser for the OWL logical expressions, e.g., [`OWLAxiom`](http://owlcs.github.io/owlapi/apidocs_5/org/semanticweb/owlapi/model/OWLAxiom.html)
    and [`OWLClassExpression`](https://owlcs.github.io/owlapi/apidocs_4/org/semanticweb/owlapi/model/OWLClassExpression.html).

    It makes use of the string representation (based on Manchester Syntax) defined in the OWLAPI. In Python,
    such string can be accessed by simply using `#!python str(some_owl_object)`.

    To keep the Java import in the main [`Ontology`][deeponto.onto.Ontology] class,
    this parser does not deal with `OWLAxiom` directly but instead its **string representation**.

    Due to the `OWLObject` syntax, this parser relies on two components:

    1. Parentheses matching;
    2. Tree construction ([`RangeNode`][deeponto.onto.verbalisation.RangeNode]).

    As a result, it will return a [`RangeNode`][deeponto.onto.verbalisation.RangeNode] that
    specifies the sub-formulas (and their respective **positions in the string representation**)
    in a tree structure.

    Examples:

        Suppose the input is an `OWLAxiom` that has the string representation:

        ```python
        >>> str(owl_axiom)
        >>> 'EquivalentClasses(<http://purl.obolibrary.org/obo/FOODON_00001707> ObjectIntersectionOf(<http://purl.obolibrary.org/obo/FOODON_00002044> ObjectSomeValuesFrom(<http://purl.obolibrary.org/obo/RO_0001000> <http://purl.obolibrary.org/obo/FOODON_03412116>)) )'
        ```

        This corresponds to the following logical expression:

        $$
        CephalopodFoodProduct \equiv MolluskFoodProduct \sqcap \exists derivesFrom.Cephalopod
        $$

        After apply the parser, a [`RangeNode`][deeponto.onto.verbalisation.RangeNode] will be returned which can be rentered as:

        ```python
        axiom_parser = OntologySyntaxParser()
        print(axiom_parser.parse(str(owl_axiom)).render_tree())
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
        located at the range `[{start}:{end}]` in the **abbreviated** expression  (see [`abbreviate_owl_expression`][deeponto.onto.verbalisation.OntologySyntaxParser.abbreviate_owl_expression]
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

    def abbreviate_owl_expression(self, owl_expression: str):
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
            owl_expression (str): The string representation of an `OWLObject`.

        Returns:
            (str): The modified string representation of this `OWLObject` where the logical operators are abbreviated.
        """
        for k, v in ABBREVIATION_DICT.items():
            owl_expression = owl_expression.replace(k, v)
        return owl_expression

    def parse(self, owl_expression: Union[str, OWLObject]) -> RangeNode:
        r"""Parse an `OWLAxiom` into a [`RangeNode`][deeponto.onto.verbalisation.RangeNode].

        This is the main entry for using the parser, which relies on the [`parse_by_parentheses`][deeponto.onto.verbalisation.OntologySyntaxParser.parse_by_parentheses]
        method below.

        Args:
            owl_expression (Union[str, OWLObject]): The string representation of an `OWLObject` or the `OWLObject` itself.

        Returns:
            (RangeNode): A parsed syntactic tree given what parentheses to be matched.
        """
        if not isinstance(owl_expression, str):
            owl_expression = str(owl_expression)
        owl_expression = self.abbreviate_owl_expression(owl_expression)
        # print("To parse the following (transformed) axiom text:\n", owl_expression)
        # parse complex patterns first
        cur_parsed = self.parse_by_parentheses(owl_expression)
        # parse the IRI patterns latter
        return self.parse_by_parentheses(owl_expression, cur_parsed, for_iri=True)

    @classmethod
    def parse_by_parentheses(
        cls, owl_expression: str, already_parsed: RangeNode = None, for_iri: bool = False
    ) -> RangeNode:
        r"""Parse an `OWLAxiom` based on parentheses matching into a [`RangeNode`][deeponto.onto.verbalisation.RangeNode].

        This function needs to be applied twice to get a fully parsed [`RangeNode`][deeponto.onto.verbalisation.RangeNode] because IRIs have
        a different parenthesis pattern.

        Args:
            owl_expression (str): The string representation of an `OWLObject`.
            already_parsed (RangeNode, optional): A partially parsed [`RangeNode`][deeponto.onto.verbalisation.RangeNode] to continue with. Defaults to `None`.
            for_iri (bool, optional): Parentheses are by default `()` but will be changed to `<>` for IRIs. Defaults to `False`.

        Raises:
            RuntimeError: Raised when the input axiom text is nor properly formatted.

        Returns:
            (RangeNode): A parsed syntactic tree given what parentheses to be matched.
        """
        if not already_parsed:
            # a root node that covers the entire sentence
            parsed = RangeNode(0, math.inf, name=f"Root", text=owl_expression, is_iri=False)
        else:
            parsed = already_parsed
        stack = []
        left_par = "("
        right_par = ")"
        if for_iri:
            left_par = "<"
            right_par = ">"

        for i, c in enumerate(owl_expression):
            if c == left_par:
                stack.append(i)
            if c == right_par:
                try:
                    start = stack.pop()
                    end = i
                    if not for_iri:
                        # the first character is actually "["
                        real_start = start - 5
                        axiom_type = owl_expression[real_start + 1 : start - 1]
                        node = RangeNode(
                            real_start,
                            end + 1,
                            name=f"{axiom_type}",
                            text=owl_expression[real_start : end + 1],
                            is_iri=False,
                        )
                        parsed.insert_child(node)
                    else:
                        # no preceding characters for just atomic class (IRI)
                        abbr_iri = owl_expression[start : end + 1].split("/")[-1].rstrip(">")
                        node = RangeNode(
                            start, end + 1, name=abbr_iri, text=owl_expression[start : end + 1], is_iri=True
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
        r"""Inserting a child [`RangeNode`][deeponto.onto.verbalisation.RangeNode].

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
        """Render the whole tree."""
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
