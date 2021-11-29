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
from owlready2.prop import IndividualValueList
from typing import Iterable, List, Dict, Tuple, Set

from collections import defaultdict
from itertools import chain
import networkx as nx

from ontoplm.utils import uniqify
from ontoplm.onto.iris import namespaces, inv_namespaces


##################################################################################
###                          process entity iris                               ###
##################################################################################


def abbr_iri(ent_iri: str, sep="#"):
    """ return the abbreviated iri of an entity given the base iri of its ontology
    e.g., onto_iri#fragment => onto_prefix:fragment
    """
    if sep == "#" or sep == "/":
        # split on the last occurrence of "#" or "/""
        # e.g.1. http://www.ihtsdo.org/snomed#concept -> http://www.ihtsdo.org/snomed#
        # e.g.2. http://snomed.info/id/228178000 -> http://snomed.info/id/
        base_iri = ent_iri.replace(ent_iri.split(sep)[-1], "")

    if namespaces[base_iri] != "":
        return ent_iri.replace(base_iri, namespaces[base_iri])
    else:
        # change nothing if no abbreviation is available
        return ent_iri


def unfold_iri(ent_abbr_iri: str):
    """ unfold the abbreviated iri of an entity given the base iri of its ontology
    e.g., onto_iri#fragment <= onto_prefix:fragment
    """
    base_abbr_iri = ent_abbr_iri.split(":")[0] + ":"
    if inv_namespaces[base_abbr_iri] != "":
        return ent_abbr_iri.replace(base_abbr_iri, inv_namespaces[base_abbr_iri])
        # change nothing if no full iri is available
    else:
        return ent_abbr_iri


##################################################################################
###                          process entity labels                             ###
##################################################################################


def ents_labs_from_props(
    ents: Iterable[EntityClass], ents2idx: Dict, lab_props: List[str] = ["label"]
):
    """ extract unique and cleaned labels (for a group of entities) given the input annotational properties;
        entities are represented by their numbers according to {ents2idx}
    """
    ents_labels = defaultdict(list)
    num_labels = 0
    for ent in ents:
        ent_labs = ent_labs_from_props(ent, lab_props)
        ents_labels[ents2idx[abbr_iri(ent.iri)]] = ent_labs
        num_labels += len(ent_labs)
    return ents_labels, num_labels


def ent_labs_from_props(ent: EntityClass, lab_props: List[str] = ["label"]):
    """ extract unique and cleaned labels (for an entity) given the input annotational properties
    """
    ent_labels = list(chain(*[prep_labs(ent, lp) for lp in lab_props]))
    return uniqify(ent_labels)


def prep_labs(ent: EntityClass, lab_prop: str) -> List[str]:
    """ preprocess the texts of a class given by a particular property including
    underscores removal and lower-casing.

    Args:
        ent : class entity
        lab_prop (str): name of the property linked to a label e.g., "label"

    Returns:
        list: cleaned labels of the input entity
    """
    raw_labels = getattr(ent, lab_prop)
    assert type(raw_labels) is IndividualValueList
    cleaned_labels = [lab.lower().replace("_", " ") for lab in raw_labels]
    return cleaned_labels


##################################################################################
###                         useful tokenizer paths                             ###
##################################################################################

BASIC_BERT = "bert-base-uncased"
BIOCLINICAL_BERT = "emilyalsentzer/Bio_ClinicalBERT"
