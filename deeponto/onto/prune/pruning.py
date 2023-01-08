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
"""Provide utilities for pruning an ontology to a sub-ontology"""

from __future__ import annotations

from typing import List
from owlready2 import ThingClass
import os

import deeponto
from deeponto.utils import run_jar
from deeponto.onto import Ontology

# strategies for hierarchy preservation
preserve_strategies = ["simplest", "logic_module"]


class OntoPruner:
    def __init__(
        self,
        saved_path: str,
        onto_path: str,
        preserved_iris_path: str,
        preserve_strategy: str = "simplest",
    ):
        self.saved_path = saved_path
        self.onto_path = onto_path
        self.pruned_onto = None
        self.preserved_iris_path = preserved_iris_path
        self.preserve_strategy = preserve_strategy

        self.onto = Ontology.from_new(self.onto_path)
        with open(preserved_iris_path, "r") as f:
            self.preserved_iris = [i.replace("\n", "") for i in f.readlines()]
        
    def run(self):
        """Run pruning
        """
        if self.preserve_strategy == "simplest":
            self.pruned_onto = self.simple_prune()
        elif self.preserve_strategy == "logic_module":
            # TODO: pruning with logic module 
            pass
        else:
            raise ValueError(f"Unknown preserve strategy: {self.preserve_strategy}")

    def simple_prune(self):
        """The simplest pruning method: isolating non-preserved classes
        and link their parents and children recursively.
        """
        # perform class isolation and parent-child linking
        for cl in self.onto.owl.classes():
            if not cl.iri in self.preserved_iris:
                self.isolate_class(cl, True)

        # save the isolated ontology
        isolated_onto_path = self.saved_path + f"/{self.onto.owl.name}.kept.owl"
        self.onto.owl.save(isolated_onto_path)

        # use java program for pruning
        self.jar_preserve(
            self.onto.owl.name, isolated_onto_path, self.preserved_iris_path, self.saved_path
        )
        
        os.remove(isolated_onto_path)

    @staticmethod
    def jar_preserve(onto_name: str, onto_path: str, preserved_iris_path: str, saved_path: str):
        """Launch the jar program for class deletion
        """
        jar = deeponto.__file__.replace("__init__.py", "onto/prune/preserve_iris.jar")
        command = f"java -jar {jar} {onto_name} {onto_path} {preserved_iris_path} {saved_path}"
        run_jar(command)

    @staticmethod
    def overlap_count(onto: Ontology, preserved_iris: List[str]):
        """Count how many classes are included in the input class names
        """
        class_iris = [cl.iri for cl in onto.owl.classes()]
        return len(set(class_iris).intersection(preserved_iris))

    @staticmethod
    def isolate_class(cl: ThingClass, keep_hierarchy: bool = True):
        """Isolate an entity class from its parents and chilren and optionally
        reconstruct the hierarchy between them (parent <subs> child)
        """
        # save the children and parents first
        children = list(set(cl.subclasses()))
        parents = list(set(cl.is_a))
        # for each child, delete the class itself as a parent
        # and add the class's parents as parents
        # e.g., if A <subs> B <subs> C and B needs to be removed
        # then construct A <subs> C and let B alone (use OWLAPI to delete)
        if keep_hierarchy:
            print(f"Link the parents and children of the isolated class: {cl.iri}")
            for ch in children:
                ch.is_a += parents
                if cl in ch.is_a:
                    ch.is_a.remove(cl)  # isolate cl from its child
                else:
                    print('cl', cl, 'not in child\'s parents:', ch.is_a)
        # isolate cl from its parents
        cl.is_a.clear()
