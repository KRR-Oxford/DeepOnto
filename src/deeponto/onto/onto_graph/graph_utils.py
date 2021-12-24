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

from owlready2.entity import EntityClass
from typing import List
import math

##################################################################################
###                                subsumption                                 ###
##################################################################################


def superclasses_of(ent: EntityClass) -> List[EntityClass]:
    """ return super-classes of an entity class but excluding non-entity classes such as restrictions
    """
    supclasses = set()
    for supclass in ent.is_a:
        # ignore the root class Thing
        if isinstance(supclass, EntityClass) and supclass.name != "Thing":
            supclasses.add(supclass)
    return list(supclasses)


def subclasses_of(ent: EntityClass) -> List[EntityClass]:
    """ return sub-classes of an entity class but excluding non-entity classes such as restrictions
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
