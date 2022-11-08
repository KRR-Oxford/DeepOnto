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
"""Logical Pattern Class based on OWLAPI"""

import re
from yacs.config import CfgNode

NEG = "ObjectComplementOf"
SOME = "ObjectSomeValuesFrom"
ALL = "ObjectAllValuesFrom"
OR = "ObjectUnionOf"
AND = "ObjectIntersectionOf"
EQUIV = "EquivalentClasses"
IRI = "<https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)>"

OP_DICT = {NEG: "NEG", SOME: "SOME", ALL: "ALL", OR: "OR", AND: "AND", EQUIV: "EQUIV", IRI: "IRI"}

# four patterns considered in the OntoLAMA paper
AND_ATOMS = f"{AND}\(((?:{IRI}| )+?)\)"
SOME_ATOM = f"{SOME}\(({IRI}) ({IRI})\)"
SOME_AND_ATOMS = f"{SOME}\(({IRI}) ({AND_ATOMS})\)"
AND_MIX = f"{AND}\(((?:{IRI}|{SOME_ATOM}|{SOME_AND_ATOMS}| )*?)\)"


class OWLPattern:
    def __init__(self, pattern_str: str):
        self.pattern_str = pattern_str

    def fit(self, axiom: str):
        return re.findall(self.pattern_str, axiom)

    def parse(self, axiom: str):
        return re.findall(self.pattern_str, axiom)

    def __repr__(self):
        rep = self.pattern_str
        for k, v in OP_DICT.items():
            rep = rep.replace(k, v)
        # rep.replace("\\(", "(").replace("\\)", ")")
        return rep


class OWLPatternAtom(OWLPattern):
    def __init__(self):
        super().__init__(IRI)

    def parse(self, axiom: str):
        return re.findall(IRI, axiom)[0]


class OWLPatternConjAtoms(OWLPattern):
    def __init__(self):
        super().__init__(AND_ATOMS)

    def parse(self, axiom: str):
        """Return atomic classes from the conjunction expression:
                A1 ⊓ A2 ⊓ ...
        """
        raw = super().parse(axiom)
        return [
            {"pattern": "AND_ATOMS", "conj": [x for x in re.findall(IRI, r)]} for r in raw
        ]


class OWLPatternExistAtom(OWLPattern):
    def __init__(self):
        super().__init__(SOME_ATOM)

    def parse(self, axiom: str):
        """Return relation and atom in an existential restriction:
                C ≡ ∃R.A1
        """
        raw = super().parse(axiom)
        return [{"pattern": "SOME_ATOM", "rel": r[0], "conj": [r[1]]} for r in raw]


class OWLPatternExistConjAtoms(OWLPattern):
    def __init__(self):
        super().__init__(SOME_AND_ATOMS)

    def parse(self, axiom: str):
        """Return relation and list of atoms in an existential restriction:
                ∃R.(A1 ⊓ A2 ⊓ ...)
        """
        raw = super().parse(axiom)
        results = []
        for rel, conj in raw:
            conj = OWLPatternConjAtoms().parse(conj)["conj"]
            results.append({"pattern": "SOME_AND_ATOMS", "rel": rel, "conj": conj})
        return results


class OWLPatternMixedConj(OWLPattern):
    def __init__(self):
        super().__init__(AND_MIX)
        # order matters
        self.patterns = [
            OWLPatternExistConjAtoms(),
            OWLPatternExistAtom(),
            OWLPatternConjAtoms(),
            OWLPatternAtom(),
        ]

    def parse(self, axiom: str):
        """Return """
        mixed = re.findall(f"{AND}\((({IRI}|{SOME_ATOM}|{SOME_AND_ATOMS}| )*?)\)", axiom)[0][0]
        mixed = [x[0] for x in re.findall(f"({IRI}|{SOME_ATOM}|{SOME_AND_ATOMS})", mixed)]
        results = []
        for m in mixed:
            # print(m)
            for p in self.patterns:
                if p.fit(m):
                    results.append(p.parse(m))
                    # print(p.parse(m))
                    break
        return results


class OWLEquivPattern(OWLPattern):
    def __init__(self, complex_pattern: OWLPattern):
        self.complex_pattern = complex_pattern
        super().__init__(f"^{EQUIV}\(({IRI}) ({complex_pattern.pattern_str}) \)$")

    def parse(self, axiom: str):
        """For an equivalence axiom: <AtomicClass> \equiv <ComplexClass>,
        return IRI for <AtomicClass> and the whole string for <ComplexClass>
        """
        if not self.fit(axiom):
            raise RuntimeError("The inpu axiom cannot be parsed by this pattern ...")
        raw = super().parse(axiom)[0]
        # the first two elements are related to the outer scope of parentheses
        atom, comp = raw[0], raw[1]
        return atom, self.complex_pattern.parse(comp)


ONTOLAMA_PATTERNS = [
    # OWLEquivPattern(OWLPatternConjAtoms()),  # included by the last pattern
    OWLEquivPattern(OWLPatternExistConjAtoms()),  # non-existing for foodon
    OWLEquivPattern(OWLPatternExistAtom()),
    OWLEquivPattern(OWLPatternMixedConj()),
]
