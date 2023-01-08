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
"""Derive OWL Equivalence Axioms of certain patterns (see OntoLAMA)"""

import re
import pandas as pd
from collections import defaultdict
from typing import Optional
from deeponto.onto.logic.parser import OWLAxiomParserBase
from deeponto.onto import Ontology
from deeponto.utils.tree import RangeNode

IRI = "<https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)>"
AND_ATOMS = f"\[AND\]\((?:{IRI}| )+?\)"
EXT_ATOM = f"\[EXT\]\({IRI} {IRI}\)"
EXT_AND_ATOMS = f"\[EXT\]\({IRI} {AND_ATOMS}\)"
AND_MIXED = f"\[AND\]\((?:{IRI}|{EXT_ATOM}|{EXT_AND_ATOMS}| )*?\)"
ALL_PATTERNS = [AND_ATOMS, EXT_ATOM, EXT_AND_ATOMS, AND_MIXED]


class OWLEquivAxiomParser(OWLAxiomParserBase):
    def __init__(self, owl_path: str, obj_prop_path: Optional[str] = None):
        super().__init__()
        self.owl_path = owl_path
        self.onto = None
        self.obj_prop_df = pd.read_csv(obj_prop_path, index_col=0) if obj_prop_path else None

    def fit(self, pattern: str, axiom_text: str):
        pattern = f"^\[(EQU|SUB|SUP)\]\(({IRI}) ({pattern})( )*\)$"
        return re.findall(pattern, self.abbr_axiom_text(axiom_text))

    @staticmethod
    def print_pattern(pattern: str, *args):
        print(pattern.replace(IRI, "<IRI>"), *args)

    def parse_atom_class(self, iri: str):
        if not self.onto:
            print("Load the input ontology ...")
            self.onto = Ontology.from_new(self.owl_path)
        # just one exceptional case where the rdf:label is missing
        # if iri == "<http://purl.obolibrary.org/obo/NCBITaxon_32224>":
        #     return "eriobotrya japonica"
        return self.onto.iri2labs[iri[1:-1]][0]

    def parse_obj_prop(self, iri: str, is_plural: bool):
        iri = iri[1:-1]
        if self.obj_prop_df is None:
            obj_prop = self.onto.obj_from_iri(iri)
            return obj_prop.label[0]
        if not is_plural:
            return self.obj_prop_df.loc[iri]["CorrectedLabelSingleObject"]
        else:
            return self.obj_prop_df.loc[iri]["CorrectedLabelMultiObject"]

    def parse_and_atoms(self, node: RangeNode):
        assert re.findall(AND_ATOMS, node.text)
        return " and ".join([self.parse_atom_class(ch.text) for ch in node.children])

    def parse_ext_atom(self, node: RangeNode, as_tuple: bool = False):
        assert re.findall(EXT_ATOM, node.text)
        assert len(node.children) == 2  # children = [objectProperty, atomClass]
        obj_prop = self.parse_obj_prop(node.children[0].text, is_plural=False)
        obj = self.parse_atom_class(node.children[1].text)
        # e.g., "derives from soyabean"
        if not as_tuple:
            return f"something that {obj_prop} {obj}"
        else:
            return node.children[0].text, obj

    def parse_ext_and_atoms(self, node: RangeNode, as_tuple: bool = False):
        assert re.findall(EXT_AND_ATOMS, node.text)
        assert len(node.children) == 2  # children = [objectProperty, AndAtoms]
        obj_prop = self.parse_obj_prop(node.children[0].text, is_plural=True)
        multi_objs = self.parse_and_atoms(node.children[1])
        # e.g., "has participants of soyabean and sunflower"
        if not as_tuple:
            return f"something that {obj_prop} {multi_objs}"
        else:
            return node.children[0].text, multi_objs

    def parse_and_mixed(self, node: RangeNode):
        assert re.findall(AND_MIXED, node.text)
        atoms = []
        exts = defaultdict(list)
        for ch in node.children:
            # we need to add start and end signs to see full matching
            if re.findall(f"^{IRI}$", ch.text):
                atoms.append(self.parse_atom_class(ch.text))
            elif re.findall(f"^{EXT_ATOM}$", ch.text):
                obj_prop, obj = self.parse_ext_atom(ch, as_tuple=True)
                exts[obj_prop].append(obj)
            elif re.findall(f"^{EXT_AND_ATOMS}$", ch.text):
                obj_prop, objs = self.parse_ext_and_atoms(ch, as_tuple=True)
                exts[obj_prop].append(objs)
            else:
                print("pattern not recognized ...")
        # assemble the properties and objects in exts
        parsed_exts = []
        for p, o in exts.items():
            o = " and ".join(o)
            if " and " in o:
                p = self.parse_obj_prop(p, is_plural=True)
            else:
                p = self.parse_obj_prop(p, is_plural=False)
            parsed_exts.append(str(p) + " " + str(o))
        if atoms:
            return " and ".join(atoms) + " that " + " and ".join(parsed_exts)
        else:
            return "something that " + " and ".join(parsed_exts)

    def parse_equiv(self, axiom_text: str, keep_atom_iri: bool = False):
        """To parse the equivalence axiom text
        
        The way of extracting children follow the structure of an equivalence axiom text
        where the root node has one child which covers the whole equivalence part (.children[0])
        subsequently, there are two children where the .children[0] is the atomic class on the left hand side
        and .children[1] is the complex class on the right hand side of the equivalence axiom.
        Therefore, root.children[0].children[1] will lead to the complex part for parsing

        Parameters
        ----------
        axiom_text : str
            the equivalence axiom text to be parsed
        keep_atom_iri : bool, optional
            keep IRI or directly render a label, by default False

        Returns
        -------
        Tuple
            (parsed atomic class, parsed complex class)

        Raises
        ------
        RuntimeError
            raise if the pattern of the axiom text is not supported yet
        """
        atom, comp = super().parse(axiom_text).children[0].children
        if not keep_atom_iri:
            atom = self.parse_atom_class(atom.text)
        else:
            atom = atom.text
        if self.fit(AND_ATOMS, axiom_text):
            comp = self.parse_and_atoms(comp)
        elif self.fit(EXT_ATOM, axiom_text):
            comp = self.parse_ext_atom(comp)
        elif self.fit(EXT_AND_ATOMS, axiom_text):
            comp = self.parse_ext_and_atoms(comp)
        elif self.fit(AND_MIXED, axiom_text):
            comp = self.parse_and_mixed(comp)
        else:
            raise RuntimeError("Pattern not supported yet ...")
        return atom, comp

    def parse_sub(self, axiom_text: str, keep_atom_iri: bool = False):
        """To parse the subsumption axiom text in the similar way as to parse the equivalence axiom

        Parameters
        ----------
        axiom_text : str
            the subsumption axiom text to be parsed
        keep_atom_iri : bool, optional
            keep IRI or directly render a label, by default False

        Returns
        -------
        Tuple
            (parsed atomic class, parsed complex class)

        Raises
        ------
        RuntimeError
            raise if the pattern of the axiom text is not supported yet
        """
        atom, comp = self.parse_equiv(axiom_text, keep_atom_iri)
        # keep the entailment order (premise => hypothesis)
        if axiom_text.startswith("SubClassOf"):
            return atom, comp
        elif axiom_text.startswith("SuperClassOf"):
            return comp, atom
