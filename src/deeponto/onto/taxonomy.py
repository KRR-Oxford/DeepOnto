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

from typing import Optional
import itertools
import networkx as nx
import numpy as np
from nltk.corpus import wordnet as wn

from . import Ontology, OntologyReasoner

import logging
logger = logging.getLogger(__name__)


RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"


class Taxonomy:
    r"""Class for building the taxonomy over structured data.

    Attributes:
        nodes (list): A list of entity ids.
        edges (list): A list of `(parent, child)` pairs.
        graph (networkx.DiGraph): A directed graph that represents the taxonomy.
        root_node (Optional[str]): Optional root node id. Defaults to `None`.
    """

    def __init__(self, edges: list, root_node: Optional[str] = None):
        self.edges = edges
        self.graph = nx.DiGraph(self.edges)
        self.nodes = list(self.graph.nodes)
        self.root_node = root_node

    def get_node_attributes(self, entity_id: str):
        """Get the attributes of the given entity."""
        return self.graph.nodes[entity_id]

    def get_children(self, entity_id: str, apply_transitivity: bool = False):
        r"""Get the set of children for a given entity."""
        if not apply_transitivity:
            return set(self.graph.successors(entity_id))
        else:
            return set(itertools.chain.from_iterable(nx.dfs_successors(self.graph, entity_id).values()))

    def get_parents(self, entity_id: str, apply_transitivity: bool = False):
        r"""Get the set of parents for a given entity."""
        if not apply_transitivity:
            return set(self.graph.predecessors(entity_id))
        else:
            # NOTE: the nx.dfs_predecessors does not give desirable results
            frontier = list(self.get_parents(entity_id))
            explored = set()
            descendants = frontier
            while frontier:
                for candidate in frontier:
                    descendants += list(self.get_parents(candidate))
                explored.update(frontier)
                frontier = set(descendants) - explored
            return set(descendants)

    def get_descendant_graph(self, entity_id: str):
        r"""Create a descendant graph (`networkx.DiGraph`) for a given entity."""
        descendants = self.get_children(entity_id, apply_transitivity=True)
        return self.graph.subgraph(list(descendants))

    def get_shortest_node_depth(self, entity_id: str):
        """Get the shortest depth of the given entity in the taxonomy."""
        if not self.root_node:
            raise RuntimeError("No root node specified.")
        return nx.shortest_path_length(self.graph, self.root_node, entity_id)

    def get_longest_node_depth(self, entity_id: str):
        """Get the longest depth of the given entity in the taxonomy."""
        if not self.root_node:
            raise RuntimeError("No root node specified.")
        return max([len(p) for p in nx.all_simple_paths(self.graph, self.root_node, entity_id)])

    def get_lowest_common_ancestor(self, entity_id1: str, entity_id2: str):
        """Get the lowest common ancestor of the given two entities."""
        return nx.lowest_common_ancestor(self.graph, entity_id1, entity_id2)


class OntologyTaxonomy(Taxonomy):
    r"""Class for building the taxonomy (top-down subsumption graph) from an ontology.

    The nodes of this graph are **named classes** only, but the hierarchy is enriched (beyond asserted axioms) by an ontology reasoner.

    Attributes:
        onto (Ontology): The input ontology to build the taxonomy.
        reasoner_type (str): The type of reasoner used. Defaults to `"struct"`. Options are `["hermit", "elk", "struct"]`.
        reasoner (OntologyReasoner): An ontology reasoner used for completing the hierarchy.
            If the `reasoner_type` is the same as `onto.reasoner_type`, then re-use `onto.reasoner`; otherwise, create a new one.
        root_node (str): The root node that represents `owl:Thing`.
        nodes (list): A list of named class IRIs.
        edges (list): A list of `(parent, child)` class pairs. That is, if $C \sqsubseteq D$, then $(D, C)$ will be added as an edge.
        graph (networkx.DiGraph): A directed subsumption graph.
    """

    def __init__(self, onto: Ontology, reasoner_type: str = "struct"):
        self.onto = onto
        # the reasoner is used for completing the hierarchy
        self.reasoner_type = reasoner_type
        # re-use onto.reasoner if the reasoner type is the same; otherwise create a new one
        self.reasoner = (
            self.onto.reasoner
            if reasoner_type == self.onto.reasoner_type
            else OntologyReasoner(self.onto, reasoner_type)
        )
        root_node = "owl:Thing"
        subsumption_pairs = []
        for cl_iri, cl in self.onto.owl_classes.items():
            # NOTE: this is different from using self.onto.get_asserted_parents which does not conduct simple reasoning
            named_parents = self.reasoner.get_inferred_super_entities(cl, direct=True)
            if not named_parents:
                # if no parents then add root node as the parent
                named_parents.append(root_node)
            for named_parent in named_parents:
                subsumption_pairs.append((named_parent, cl_iri))
        super().__init__(edges=subsumption_pairs, root_node=root_node)

        # set node annotations (rdfs:label)
        for class_iri in self.nodes:
            if class_iri == self.root_node:
                self.graph.nodes[class_iri]["label"] = "Thing"
            else:
                owl_class = self.onto.get_owl_object(class_iri)
                self.graph.nodes[class_iri]["label"] = self.onto.get_annotations(owl_class, RDFS_LABEL)

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

    def get_shortest_node_depth(self, class_iri: str):
        """Get the shortest depth of the given named class in the taxonomy."""
        return nx.shortest_path_length(self.graph, self.root_node, class_iri)

    def get_longest_node_depth(self, class_iri: str):
        """Get the longest depth of the given named class in the taxonomy."""
        return max([len(p) for p in nx.all_simple_paths(self.graph, self.root_node, class_iri)])

    def get_lowest_common_ancestor(self, class_iri1: str, class_iri2: str):
        """Get the lowest common ancestor of the given two named classes."""
        return super().get_lowest_common_ancestor(class_iri1, class_iri2)


class WordnetTaxonomy(Taxonomy):
    r"""Class for the building the taxonomy (hypernym graph) from wordnet.

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
            try:
                self.graph.nodes[synset.name()]["name"] = synset.name().split(".")[0].replace("_", " ")
                self.graph.nodes[synset.name()]["definition"] = synset.definition()
            except:
                continue

    @staticmethod
    def fetch_synsets(pos: str = "n"):
        """Get synsets of certain pos-tag from wordnet."""
        words = wn.words()
        synsets = set()
        for word in words:
            synsets.update(wn.synsets(word, pos=pos))
        logger.info(f'{len(synsets)} synsets (pos="{pos}") fetched.')
        return synsets

    @staticmethod
    def fetch_hypernyms(synsets: set, include_membership: bool = False):
        """Get hypernym-hyponym pairs from a given set of wordnet synsets."""
        hypernym_hyponym_pairs = []
        for synset in synsets:
            for h_synset in synset.hypernyms():
                hypernym_hyponym_pairs.append((h_synset.name(), synset.name()))
            if include_membership:
                for h_synset in synset.instance_hypernyms():
                    hypernym_hyponym_pairs.append((h_synset.name(), synset.name()))
        logger.info(f"{len(hypernym_hyponym_pairs)} hypernym-hyponym pairs fetched.")
        return hypernym_hyponym_pairs


class TaxonomyNegativeSampler:
    r"""Class for the efficient negative sampling with buffer over the taxonomy.

    Attributes:
        taxonomy (str): The taxonomy for negative sampling.
        entity_weights (Optional[dict]): A dictionary with the taxonomy entities as keys and their corresponding weights as values. Defaults to `None`.
    """

    def __init__(self, taxonomy: Taxonomy, entity_weights: Optional[dict] = None):
        self.taxonomy = taxonomy
        self.entities = self.taxonomy.nodes
        # uniform distribution if weights not provided
        self.entity_weights = entity_weights

        self._entity_probs = None
        if self.entity_weights:
            self._entity_probs = np.array([self.entity_weights[e] for e in self.entities])
            self._entity_probs = self._entity_probs / self._entity_probs.sum()
        self._buffer = []
        self._default_buffer_size = 10000

    def fill(self, buffer_size: Optional[int] = None):
        """Buffer a large collection of entities sampled with replacement for faster negative sampling."""
        buffer_size = buffer_size if buffer_size else self._default_buffer_size
        if self._entity_probs:
            self._buffer = np.random.choice(self.entities, size=buffer_size, p=self._entity_probs)
        else:
            self._buffer = np.random.choice(self.entities, size=buffer_size)

    def sample(self, entity_id: str, n_samples: int, buffer_size: Optional[int] = None):
        """Sample N negative samples for a given entity with replacement."""
        negative_samples = []
        positive_samples = self.taxonomy.get_parents(entity_id, True)
        while len(negative_samples) < n_samples:
            if len(self._buffer) < n_samples:
                self.fill(buffer_size)
            negative_samples += list(filter(lambda x: x not in positive_samples, self._buffer[:n_samples]))
            self._buffer = self._buffer[n_samples:]  # remove the samples from the buffer
        return negative_samples[:n_samples]
