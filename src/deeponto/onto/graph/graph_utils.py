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
    ancestors  = superclasses_of(ent)
    for parent in ancestors:
        ancestors += ancestors_of(parent)
    return ancestors

def descendants_of(ent: EntityClass):
    """Return all the descendents of a class
    """
    descendants  = subclasses_of(ent)
    for child in descendants:
        descendants += descendants_of(child)
    return descendants


def neighbours_of(
    anchor_ent: EntityClass,
    explored: list = [],
    hob: int = 1,
    max_hob: int = 5,
):
    """Compute neighbours of an anchor entity up to max_hob
    """
    neighbours = defaultdict(list)
    
    # return if have explored
    if anchor_ent in explored:
        return neighbours
    
    # explore neighbours by 1-hob
    neighbours[hob] += superclasses_of(anchor_ent) + subclasses_of(anchor_ent)
    neighbours[hob] = list(set(neighbours[hob]))
    explored.append(anchor_ent)
    
    # return if too far from the anchor
    if hob == max_hob:
        return neighbours
    
    # for each 1-hob neighbour, add their 1-hob neighbours as 2-hob neighbours
    # recursion applied to seek for neighbours up to max_hob
    for n in neighbours[hob]:
        neighbours.update(neighbours_of(n, explored, hob + 1))
    
    return neighbours
