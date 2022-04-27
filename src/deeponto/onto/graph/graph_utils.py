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

from owlready2.entity import ThingClass
from typing import List
from collections import defaultdict
import math

from deeponto import SavedObj


##################################################################################
###                                subsumption                                 ###
##################################################################################


def super_thing_classes_of(ent: ThingClass, ignore_root: bool = True) -> List[ThingClass]:
    """ return super-classes of an entity class but excluding non-entity classes 
    such as existential axioms
    """
    supclasses = set()
    for supclass in ent.is_a:
        # ignore the root class Thing
        if isinstance(supclass, ThingClass):
            if ignore_root and supclass.name == "Thing":
                continue
            else:
                supclasses.add(supclass)
    return list(supclasses)


def sub_thing_classes_of(ent: ThingClass) -> List[ThingClass]:
    """ return sub-classes of an entity class but excluding non-entity classes 
    such as existential axioms
    """
    subclasses = set()
    for subclass in ent.subclasses():
        if isinstance(subclass, ThingClass):
            subclasses.add(subclass)
    return list(subclasses)


def depth_max(ent: ThingClass) -> int:
    """ get te maximum depth of a class (including only named 
    classes in the path) to the root
    """
    supclasses = super_thing_classes_of(ent=ent)
    # with only Thing Class
    if len(supclasses) == 0:
        return 1
    d_max = 0
    for super_c in supclasses:
        super_d = depth_max(ent=super_c)
        if super_d > d_max:
            d_max = super_d
    return d_max + 1


def depth_min(ent: ThingClass) -> int:
    """Get te minimum depth of a class (including only named 
    classes in the path) to the root
    """
    supclasses = super_thing_classes_of(ent=ent)
    # with only Thing Class
    if len(supclasses) == 0:
        return 1
    d_min = math.inf
    for super_c in supclasses:
        super_d = depth_min(ent=super_c)
        if super_d < d_min:
            d_min = super_d
    return d_min + 1


#TODO: debug cannot assert isistance
def thing_class_ancestors_of(ent: ThingClass, include_self: bool=False):
    """Return all the ancestors (restricted to EntityClass) of a class 
    (except for the root ThingClass)
    """
    ancestors = [a for a in ent.ancestors() if isinstance(a, ThingClass)]
    if not include_self:
        ancestors.remove(ent)
    return list(set(ancestors))


def thing_class_descendants_of(ent: ThingClass, include_self: bool=False):
    """Return all the descendents (restricted to EntityClass) of a class 
    """
    descendants = [a for a in ent.descendants() if isinstance(a, ThingClass)]
    if not include_self:
        descendants.remove(ent)
    return list(set(descendants))


def neighbours_of(anchor_ent: ThingClass, max_hop: int = 5, ignore_root: bool = True):
    """Compute neighbours of an anchor entity up to max_hop
    in Breadth First Search style which ensures determined outputs
    """
    neighbours = defaultdict(list)
    frontier = [anchor_ent]
    explored = []
    hop = 1

    while hop <= max_hop:
        cur_hop_neighbours = []
        for ent in frontier:
            cur_hop_neighbours += super_thing_classes_of(ent, ignore_root) + sub_thing_classes_of(ent)
            cur_hop_neighbours = list(set(cur_hop_neighbours))
            explored.append(ent)
        neighbours[hop] = cur_hop_neighbours
        frontier = list(set(cur_hop_neighbours) - set(explored))
        hop += 1

    stats = {k: len(v) for k, v in neighbours.items()}
    print(f"Numbers of neighbours at each hop away from entity: {str(anchor_ent)}")
    SavedObj.print_json(stats)

    return neighbours
