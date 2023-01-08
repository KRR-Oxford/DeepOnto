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
"""Utilities for OAEI"""
import xml.etree.ElementTree as ET


def read_oaei_mappings(rdf_file: str):
    """
    Args:
        rdf_file: path to mappings in rdf format
        src_onto: source ontology iri abbreviation, e.g. fma
        tgt_onto: target ontology iri abbreviation, e.g. nci
    Returns:
        mappings(=;>,<), mappings(?)
    """
    xml_root = ET.parse(rdf_file).getroot()
    ref_mappings = []  # where relation is "="
    ignored_mappings = []  # where relation is "?"

    for elem in xml_root.iter():
        # every Cell contains a mapping of en1 -rel(some value)-> en2
        if "Cell" in elem.tag:
            en1, en2, rel, measure = None, None, None, None
            for sub_elem in elem:
                if "entity1" in sub_elem.tag:
                    en1 = list(sub_elem.attrib.values())[0]
                elif "entity2" in sub_elem.tag:
                    en2 = list(sub_elem.attrib.values())[0]
                elif "relation" in sub_elem.tag:
                    rel = sub_elem.text
                elif "measure" in sub_elem.tag:
                    measure = sub_elem.text
            row = (en1, en2, measure)
            # =: equivalent; > superset of; < subset of.
            if rel == "=" or rel == ">" or rel == "<":
                # rel.replace("&gt;", ">").replace("&lt;", "<")
                ref_mappings.append(row)
            elif rel == "?":
                ignored_mappings.append(row)
            else:
                print("Unknown Relation Warning: ", rel)

    print('#Maps ("="):', len(ref_mappings))
    print('#Maps ("?"):', len(ignored_mappings))

    return ref_mappings, ignored_mappings
