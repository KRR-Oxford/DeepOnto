# Copyright 2023 Yuan He. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.\


from __future__ import annotations

from typing import List
import itertools
import networkx as nx
from nltk.corpus import wordnet as wn

from . import Ontology, OntologyReasoner, RDFS_LABEL


class Taxonomy:
    r"""Class for building the taxonomy over structured data.

    Attributes:
        nodes (list): A list of entity ids.
        edges (list): A list of `(child, parent)` pairs.
        graph (networkx.DiGraph): A directed graph that represents the taxonomy.
    """

    def __init__(self, edges: list):
        self.edges = edges
        self.graph = nx.DiGraph(self.edges)
        self.nodes = list(self.graph.nodes)

    def get_node_attributes(self, entity_id: str):
        """Get the attributes of the given entity."""
        return self.graph.nodes[entity_id]

    def get_parents(self, entity_id: str, apply_transitivity: bool = False):
        r"""Get the set of parents for a given entity."""
        if not apply_transitivity:
            return set(self.graph.successors(entity_id))
        else:
            return set(itertools.chain.from_iterable(nx.dfs_successors(self.graph, entity_id).values()))

    def get_children(self, entity_id: str, apply_transitivity: bool = False):
        r"""Get the set of children for a given entity."""
        if not apply_transitivity:
            return set(self.graph.predecessors(entity_id))
        else:
            # NOTE: the nx.dfs_predecessors does not give desirable results
            frontier = list(self.get_children(entity_id))
            explored = set()
            descendants = frontier
            while frontier:
                for candidate in frontier:
                    descendants += list(self.get_children(candidate))
                explored.update(frontier)
                frontier = set(descendants) - explored
            return set(descendants)

    def get_descendant_graph(self, entity_id: str):
        r"""Create a descendant graph (`networkx.DiGraph`) for a given entity."""
        edges = []
        descendants = self.get_children(entity_id, apply_transitivity=True)
        for desc in descendants:
            for desc_hyper in self.get_children(desc):
                edges.append((desc, desc_hyper))
        return nx.DiGraph(edges)


class OntologyTaxonym(Taxonomy):
    r"""Class for building the taxonym (subsumption graph) from an ontology.

    The nodes of this graph are **named classes** only, but the hierarchy is enriched by a simple structural reasoner.

    Attributes:
        onto (Ontology): The input ontology to build the taxonym.
        structural_reasoner (OntologyReasoner): A simple structural reasoner for completing the hierarchy.
        nodes (list): A list of named class IRIs.
        edges (list): A list of class subsumption pairs.
        graph (networkx.DiGraph): A directed subsumption graph.
    """

    def __init__(self, onto: Ontology):
        self.onto = onto
        # simple structural reasoner used for completing the hierarchy
        self.structural_reasoner = OntologyReasoner(self.onto, "struct")
        subsumption_pairs = []
        for cl_iri, cl in self.onto.owl_classes:
            # NOTE: this is different from using self.onto.get_asserted_parents which does not conduct simple reasoning
            for named_parent in self.structural_reasoner.get_inferred_super_entities(cl, direct=True):
                subsumption_pairs.append((cl_iri, named_parent))
        super().__init__(edges=subsumption_pairs)

        # set node annotations
        for class_iri in self.nodes:
            owl_class = self.onto.get_owl_object(class_iri)
            for annotation_property_iri in self.onto.owl_annotation_properties.keys():
                self.graph.nodes[class_iri][annotation_property_iri] = self.onto.get_annotations(
                    owl_class, annotation_property_iri
                )

    def get_parents(self, class_iri: str, apply_transitivity: bool = False):
        r"""Get the set of parents for a given class.

        It is worth noting that this method with transitivity applied can be deemed as simple structural reasoning.
        For more advanced logical reasoning, use the DL reasoner `self.onto.reasoner` instead.
        """
        return super().get_parents(class_iri, apply_transitivity)

    def get_children(self, class_iri: str, apply_transitivity: bool = False):
        r"""Get the set of children for a given class.

        It is worth noting that this method with transitivity applied can be deemed as simple structural reasoning.
        For more advanced logical reasoning, use the DL reasoner `self.onto.reasoner` instead.
        """
        return super().get_children(class_iri, apply_transitivity)

    def get_descendant_graph(self, class_iri: str):
        r"""Create a descendant graph (`networkx.DiGraph`) for a given ontology class."""
        super().get_descendant_graph(class_iri)


class WordnetTaxonym(Taxonomy):
    r"""Class for the building the taxonym (hypernym graph) from wordnet.

    Attributes:
        pos (str): The pos-tag of entities to be extracted from wordnet.
        nodes (list): A list of entity ids extracted from wordnet.
        edges (list): A list of hyponym-hypernym pairs.
        graph (networkx.DiGraph): A directed hypernym graph.
    """

    def __init__(self, pos: str = "n", include_membership: bool = False):
        r"""Initialise the wordnet taxonomy.

        Args:
            pos (str): The pos-tag of entities to be extracted from wordnet.
            include_membership (bool): Whether to include `instance_hypernyms` or not (e.g., London is an instance of City).  Defaults to `False`.
        """

        self.pos = pos
        synsets = self.fetch_synsets(pos=pos)
        hypernym_pairs = self.fetch_hypernyms(synsets, include_membership)
        super().__init__(edges=hypernym_pairs)

        # set node annotations
        for synset in synsets:
            self.graph.nodes[synset.name()]["name"] = synset.name().split(".")[0].replace("_", " ")
            self.graph.nodes[synset.name()]["definition"] = synset.definition()

    @staticmethod
    def fetch_synsets(pos: str = "n"):
        """Get synsets of certain pos-tag from wordnet."""
        words = wn.words()
        synsets = set()
        for word in words:
            synsets.update(wn.synsets(word, pos=pos))
        print(len(synsets), f'synsets (pos="{pos}") fetched.')
        return synsets

    @staticmethod
    def fetch_hypernyms(synsets: set, include_membership: bool = False):
        """Get hyponym-hypernym pairs from a given set of wordnet synsets."""
        hypernyms = []
        for synset in synsets:
            for h_synset in synset.hypernyms():
                hypernyms.append((synset.name(), h_synset.name()))
            if include_membership:
                for h_synset in synset.instance_hypernyms():
                    hypernyms.append((synset.name(), h_synset.name()))
        print(len(hypernyms), f"hypernyms fetched.")
        return hypernyms
