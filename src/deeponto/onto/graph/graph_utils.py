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
"""Utility functions for exploring an ontology graph"""

from owlready2.entity import EntityClass
from typing import List
from collections import defaultdict
import math

from deeponto import SavedObj


##################################################################################
###                                subsumption                                 ###
##################################################################################


def superclasses_of(ent: EntityClass, ignore_root: bool = True) -> List[EntityClass]:
    """ return super-classes of an entity class but excluding non-entity classes 
    such as existential axioms
    """
    supclasses = set()
    for supclass in ent.is_a:
        # ignore the root class Thing
        if isinstance(supclass, EntityClass):
            if ignore_root and supclass.name == "Thing":
                continue
            else:
                supclasses.add(supclass)
    return list(supclasses)


def subclasses_of(ent: EntityClass) -> List[EntityClass]:
    """ return sub-classes of an entity class but excluding non-entity classes 
    such as existential axioms
    """
    subclasses = set()
    for subclass in ent.subclasses():
        if isinstance(subclass, EntityClass):
            subclasses.add(subclass)
    return list(subclasses)


def depth_max(ent: EntityClass) -> int:
    """ get te maximum depth of a class to the root
    """
    supclasses = superclasses_of(ent=ent)
    if len(supclasses) == 0:
        return 0
    d_max = 0
    for super_c in supclasses:
        super_d = depth_max(ent=super_c)
        if super_d > d_max:
            d_max = super_d
    return d_max + 1


def depth_min(ent: EntityClass) -> int:
    """Get te minimum depth of a class to the root
    """
    supclasses = superclasses_of(ent=ent)
    if len(supclasses) == 0:
        return 0
    d_min = math.inf
    for super_c in supclasses:
        super_d = depth_min(ent=super_c)
        if super_d < d_min:
            d_min = super_d
    return d_min + 1


def ancestors_of(ent: EntityClass):
    """Return all the ancestors of a class (except for the root ThingClass)
    """
    ancestors = superclasses_of(ent)
    for parent in ancestors:
        ancestors += ancestors_of(parent)
    return ancestors


def descendants_of(ent: EntityClass):
    """Return all the descendents of a class
    """
    descendants = subclasses_of(ent)
    for child in descendants:
        descendants += descendants_of(child)
    return descendants


def neighbours_of(anchor_ent: EntityClass, max_hob: int = 5, ignore_root: bool = True):
    """Compute neighbours of an anchor entity up to max_hob
    in Breadth First Search style which ensures determined outputs
    """
    neighbours = defaultdict(list)
    frontier = [anchor_ent]
    explored = []
    hob = 1

    while hob <= max_hob:
        cur_hob_neighbours = []
        for ent in frontier:
            cur_hob_neighbours += superclasses_of(ent, ignore_root) + subclasses_of(ent)
            cur_hob_neighbours = list(set(cur_hob_neighbours))
            explored.append(ent)
        neighbours[hob] = cur_hob_neighbours
        frontier = list(set(cur_hob_neighbours) - set(explored))
        hob += 1

    stats = {k: len(v) for k, v in neighbours.items()}
    print(f"Numbers of neighbours at each hob away from entity: {str(anchor_ent)}")
    SavedObj.print_json(stats)

    return neighbours
