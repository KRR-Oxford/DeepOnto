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

from typing import TYPE_CHECKING, List
from deeponto.onto.text.text_utils import abbr_iri
from owlready2 import destroy_entity

# to avoid circular imports
if TYPE_CHECKING:
    from deeponto.onto import Ontology


def overlap_count(onto: Ontology, preserved_class_names: List[str]):
    """Count how many classes are included in the input class names
    """
    class_name_set = [abbr_iri(cl.iri) for cl in onto.owl.classes()]
    return len(set(class_name_set).intersection(preserved_class_names))


def preserve_classes(
    onto: Ontology,
    preserved_class_names: List[str],
    keep_hierarchy: bool = True,
    apply_destroy: bool = True,
):
    """Preserve only the provided entity classes while keeping the relative hierarchy 
    by linking the parents and children of the deleted classes
    """
    count = 0
    original_class_num = len(list(onto.owl.classes()))
    num_preserved = overlap_count(onto, preserved_class_names)
    print("Number of classes that should be preserved:", num_preserved)
    num_destroyed = original_class_num - num_preserved
    # while loop is used for circumvent owlready2 exception
    while len(list(onto.owl.classes())) > num_preserved:
        try:
            for cl in onto.owl.classes():
                if not abbr_iri(cl.iri) in preserved_class_names:
                    if keep_hierarchy:
                        print(f"Link SubClasses and SuperClasses of the about-to-delete class: {abbr_iri(cl.iri)}")
                        # save the children and parents first
                        children = list(set(cl.subclasses()))
                        parents = list(set(cl.is_a))
                        # for each child, delete the class itself as a parent
                        # and add the class's parents as parents
                        # e.g., if A <subs> B <subs> C and B needs to be removed
                        # then construct A <subs> C and let B alone (use OWLAPI to delete)
                        for ch in children:
                            ch.is_a += parents
                            ch.is_a.remove(cl)  # isolate cl from its child
                        # isolate cl from its parents
                        cl.is_a.clear()
                    if apply_destroy:
                        # owlready2 destroy_entity has some unknown issue reporting
                        # use OWLAPI instead to delete the class to fulfill pruning
                        destroy_entity(cl)
                    count += 1
            if not apply_destroy:
                break
        except:
            print(f"Process: {count}/{num_destroyed}")
        # print(f"Process: {count}/{num_destroyed}")
    print(f"Finished: {count}/{num_destroyed}")
    return onto
