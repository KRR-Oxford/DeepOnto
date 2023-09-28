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
# limitations under the License.

from typing import Optional
import itertools
from nltk.corpus import wordnet as wn
import networkx as nx
import numpy as np


class WordnetTaxonym:
    """Class for the building the taxonym (noun hypernym graph) from wordnet."""

    def __init__(self, pos: str = "n", include_membership: bool = False):
        synsets = self.fetch_synsets(pos=pos)
        self.entity_index = dict()
        for synset in synsets:
            self.entity_index[synset.name()] = {
                "name": synset.name().split(".")[0].replace("_", " "),
                "definition": synset.definition(),
            }
        self.edges = self.fetch_hypernyms(synsets, include_membership)
        self.graph = nx.DiGraph(self.edges)
        self.entities = list(self.graph.nodes)

    def get_descendant_graph(self, top_entity_id: str):
        r"""Create a descendant graph for a given entity."""
        edges = []
        descendants = self.get_hyponyms(top_entity_id, apply_transitivity=True)
        for desc in descendants:
            for desc_hyper in self.get_hypernyms(desc):
                edges.append((desc, desc_hyper))
        return nx.DiGraph(edges)

    def get_hypernyms(self, entity_id: str, apply_transitivity: bool = False):
        """Get a set of super-entities (hypernyms) for a given entity."""
        if not apply_transitivity:
            return set(self.graph.successors(entity_id))
        else:
            return set(itertools.chain.from_iterable(nx.dfs_successors(self.graph, entity_id).values()))

    def get_hyponyms(self, entity_id: str, apply_transitivity: bool = False):
        """Get a set of sub-entities (hyponyms) for a given entity."""
        if not apply_transitivity:
            return set(self.graph.predecessors(entity_id))
        else:
            # NOTE: the nx.dfs_predecessors does not give desirable results
            frontier = list(self.get_hyponyms(entity_id))
            explored = set()
            descendants = frontier
            while frontier:
                for candidate in frontier:
                    descendants += list(self.get_hyponyms(candidate))
                explored.update(frontier)
                frontier = set(descendants) - explored
            return set(descendants)

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
