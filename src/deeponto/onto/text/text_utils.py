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

from owlready2.entity import ThingClass
from owlready2.prop import IndividualValueList
from typing import Iterable, List, Dict, Tuple

from collections import defaultdict
from itertools import chain, product
import math

from deeponto.utils import uniqify
from deeponto.onto.iris import namespaces, inv_namespaces

##################################################################################
###                         useful tokenizer paths                             ###
##################################################################################

BASIC_BERT = "bert-base-uncased"
BIOCLINICAL_BERT = "emilyalsentzer/Bio_ClinicalBERT"


##################################################################################
###                          process entity iris                               ###
##################################################################################


def abbr_iri(ent_iri: str):
    """Return the abbreviated iri of an entity given the base iri of its ontology
    e.g., onto_iri#fragment => onto_prefix:fragment
    """
    sep = "#"  # separators are either "#" or "/""
    if sep == "#" and not "#" in ent_iri:
        sep = "/"
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
    """Unfold the abbreviated iri of an entity given the base iri of its ontology
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


def lab_product(src_ent_labs: List[str], tgt_ent_labs: List[str]) -> Tuple[List, List]:
    """Compute Catersian Product of source and target entity labels,
    and return in the form of two lists
    """
    # zip(*) is the inverse of zip()
    src_out, tgt_out = zip(*product(src_ent_labs, tgt_ent_labs))
    return list(src_out), list(tgt_out)


def ents_labs_from_props(
    ents: Iterable[ThingClass], ents2idx: Dict, lab_props: List[str] = ["label"]
):
    """Extract unique and cleaned labels (for a group of entities) given the input annotational properties;
    entities are represented by their numbers according to {ents2idx}
    """
    ents_labels = defaultdict(list)
    num_labels = 0
    for ent in ents:
        ent_labs = ent_labs_from_props(ent, lab_props)
        ents_labels[ents2idx[abbr_iri(ent.iri)]] = ent_labs
        num_labels += len(ent_labs)
    return ents_labels, num_labels


def ent_labs_from_props(ent: ThingClass, lab_props: List[str] = ["label"]):
    """Extract unique and cleaned labels (for an entity) given the input annotational properties
    """
    ent_labels = list(chain(*[prep_labs(ent, lp) for lp in lab_props]))
    return uniqify(ent_labels)


def prep_labs(ent: ThingClass, lab_prop: str) -> List[str]:
    """Preprocess the texts of a class given by a particular property including
    underscores removal and lower-casing.

    Args:
        ent : class entity
        lab_prop (str): name of the property linked to a label e.g., "label"

    Returns:
        list: cleaned labels of the input entity
    """
    try:
        raw_labels = getattr(ent, lab_prop)
        assert isinstance(raw_labels, IndividualValueList)
        cleaned_labels = [lab.lower().replace("_", " ") for lab in raw_labels]
        return cleaned_labels
    except:
        # when input label cannot be retrieved (annotation property not defined)
        # we return an empty label list
        return []


##################################################################################
###                 naive disjointness of sibling classes                      ###
##################################################################################


def sib_labs(
    ents: Iterable[ThingClass], lab_props: List[str] = ["label"]
) -> List[List[List[str]]]:
    """Return all the sibling label groups with size > 1 and no duplicates
    """
    sib_lables = []
    for ent in ents:
        ch_labels = child_labs(ent, lab_props)
        if ch_labels and len(ch_labels) > 1:
            sib_lables.append(ch_labels)
    sib_lables = uniqify(sib_lables)
    # change type to 3-d lists
    return list(map(lambda dis_group: [list(syn_group) for syn_group in dis_group], sib_lables))


def child_labs(ent: ThingClass, lab_props: List[str] = ["label"]) -> Tuple[Tuple[str]]:
    """Return labels of child entity classes, ensuring that no label groups are duplicated 
    """
    return tuple(
        set(
            tuple(ent_labs_from_props(child, lab_props))
            for child in ent.subclasses()
            if isinstance(child, ThingClass)
        )
    )


##################################################################################
###                 inverted index-based candidate selection                   ###
##################################################################################


def idf_select(ent_toks: List[str], inv_idx: Dict, pool_size: int = 200) -> List[str]:
    """Given tokenized labels associated to an entity, select a set of entities 
    from the values of an inverted index according to `idf` scores; `idf` score of 
    a token T is lower when T is shared by more entities. 
    
    We use `idf` instead of  `tf` because labels have different lengths 
    ==> tf is not a fair measure.
    """
    cand_pool = defaultdict(lambda: 0)
    # D := number of "documents", i.e., total number of entities in the values of the inverted index
    D = len(set(chain.from_iterable(inv_idx.values())))
    for tok in ent_toks:
        # each token is associated with some classes
        potential_cands = inv_idx.setdefault(tok, [])
        if not potential_cands:
            continue
        # We use idf instead of tf because the text for each class is of different length, tf is not a fair measure
        # inverse document frequency: with more classes to have the current token tk, the score decreases
        idf = math.log10(D / len(potential_cands))
        for ent_id in potential_cands:
            # each candidate class is scored by sum(idf)
            cand_pool[ent_id] += idf
    cand_pool = list(sorted(cand_pool.items(), key=lambda item: item[1], reverse=True))
    # select the first K ranked
    return cand_pool[:pool_size]
