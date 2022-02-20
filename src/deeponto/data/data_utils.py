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
"""Providing useful utility functions regarding data"""

from owlready2 import get_ontology, ThingClass
import urllib.request

from deeponto.utils import create_path, detect_path
from deeponto.onto.text.text_utils import abbr_iri
from deeponto.onto.mapping import *


mondo_url = "http://purl.obolibrary.org/obo/mondo/mondo-with-equivalents.owl"

##################################################################################
###                                data download                               ###
##################################################################################


def onto_name_from_url(url: str):
    return url.split("/")[-1]


def download_onto(url: str, saved_path: str):
    """Download ontology from url
    """
    # by default the name is the last part of the url string
    onto_name = onto_name_from_url(url)
    onto_path = f"{saved_path}/{onto_name}"
    if detect_path(onto_path):
        print(f"Ontology: {onto_name} has been downloaded, skip the download process ...")
    else:
        create_path(saved_path)
        urllib.request.urlretrieve(url, f"{onto_path}")
    return onto_name


def download_mondo(saved_path: str):
    """Download mondo ontology
    """
    return download_onto(mondo_url, saved_path)


##################################################################################
###                           mapping extraction                               ###
##################################################################################


def extract_mappings(url: str, saved_path: str, map_prop: str, n_kept: int = 100, rel: str = "=", sep: str = "/"):
    """Extract mappings from mondo ontology
    """
    # load the mondo using owlready2 (we do not use the Ontology class of
    # this package because we don't need any processing at this stage)
    onto_name = onto_name_from_url(url)
    owl = get_ontology(f"{saved_path}/{onto_name}").load()
    mappings = OntoMappings(flag=map_prop, n_best=n_kept, rel=rel)
    # mondo_classes = [c for c in owl.classes() if c.name.startswith('MONDO_')]
    for cl in owl.classes():
        src_iri = abbr_iri(cl.iri, sep=sep)
        for match in getattr(cl, map_prop):
            if type(match) == str:  # some concepts are not updated in MONDO as a class
                tgt_iri = abbr_iri(match, sep=sep)
            elif type(match) == ThingClass:  # concepts that are independent MONDO classes
                tgt_iri = abbr_iri(match.iri, sep=sep)
            else:
                # for owl:equivalentTo, there might be logically connected classes (e.g., A AND B);
                # they should be ignored
                continue
            mappings.add(EntityMapping(src_iri, tgt_iri, rel, 1.0))
    print(f"# Mappings ({map_prop}): {len(mappings)}")
    return mappings


def extract_mondo_mappings(saved_path: str, map_prop: str, n_kept: int = 100, rel: str = "="):
    return extract_mappings(mondo_url, saved_path, map_prop, n_kept, rel, sep="/")


##################################################################################
###                         split mappings pairwise                            ###
##################################################################################

def pairwise_mappings(integrated_mappings: OntoMappings):
    """For an integrated ontology such as MONDO and UMLS, we extract pairwise
    (ontology) mappings from its mappings, where there is an `intermediate concept`
    that points to concepts of various individual ontologies.
    """
    # TODO: no need to do this yet because MONDO team provides mappings