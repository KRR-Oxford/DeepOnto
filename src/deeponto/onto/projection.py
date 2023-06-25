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
# Credit to: https://github.com/bio-ontology-research-group/mowl/blob/docs/mowl/projection/owl2vec_star/model.py

from __future__ import annotations

from . import Ontology

from org.mowl.Projectors import OWL2VecStarProjector as Projector #type:ignore
from org.semanticweb.owlapi.model import OWLOntology #type:ignore


class OntologyProjector:

    r'''Class for ontology projection -- transforming ontology axioms into triples.

    !!! note "Credit"

        The code of this class originates from the [mOWL library](https://mowl.readthedocs.io/en/latest/index.html).
        
    Attributes:
        bidirectional_taxonomy (bool): If `True` then per each `SubClass` edge one `SuperClass` edge will
            be generated. Defaults to `False`.
        only_taxonomy (bool): If `True`, then projection will only include `subClass` edges. Defaults to `False`.
        include_literals (bool): If `True` the projection will also include triples involving data property
            assertions and annotations. Defaults to `False`.
    '''

    def __init__(self, bidirectional_taxonomy: bool=False, only_taxonomy: bool=False, include_literals: bool=False):
        self.bidirectional_taxonomy = bidirectional_taxonomy
        self.include_literals = include_literals
        self.only_taxonomy = only_taxonomy
        self.projector = Projector(self.bidirectional_taxonomy, self.only_taxonomy,
                                   self.include_literals)

    def project(self, ontology: Ontology):
        """The projection algorithm implemented in OWL2Vec*.

        Args:
            ontology (Ontology): An ontology to be processed.

        Returns:
            (Set): Set of triples after projection.
        """
        ontology = ontology.owl_onto
        if not isinstance(ontology, OWLOntology):
            raise TypeError(
                "Input ontology must be of type `org.semanticweb.owlapi.model.OWLOntology`.")
        edges = self.projector.project(ontology)
        triples = [(str(e.src()), str(e.rel()), str(e.dst())) for e in edges if str(e.dst()) != ""]
        return set(triples)
