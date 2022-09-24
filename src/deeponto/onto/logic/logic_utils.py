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

from owlready2 import default_world
from owlready2.entity import ThingClass


##################################################################################
###                         owlready2 simple check                             ###
##################################################################################


def disjoints_from_iri(iri: str):
    """Return the asserted disjoints (in IRIs) of an entity or property object defined in owlready2 given its IRI
    """
    ent = default_world[iri]
    dis_ent_iris = []
    for d in ent.disjoints():
        # AllDisjoint([ent, dis_ent])
        dis_ent = list(map(lambda x: x.iri, d.entities))
        dis_ent.remove(ent.iri)
        assert len(dis_ent) == 1
        dis_ent_iris += dis_ent
    return list(set(dis_ent_iris))


def check_disjoint(ent1: ThingClass, ent2: ThingClass):
    """Check if two entities are disjoint (direct assertion)
    """
    ds1 = disjoints_from_iri(ent1.iri)
    ds2 = disjoints_from_iri(ent2.iri)
    if ent1.iri in ds2:
        assert ent2.iri in ds1
        return True
    return False
