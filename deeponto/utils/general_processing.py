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


from __future__ import annotations

from typing import List, Set, Tuple, Optional
import logging
import datetime
import time
import itertools
import random
import xml.etree.ElementTree as ET

def uniqify(ls):
    """Return a list of unique elements and preserve ordering.
    """
    non_empty_ls = list(filter(lambda x: x != "", ls))
    return list(dict.fromkeys(non_empty_ls))


def sort_dict_by_values(dic: dict, desc: bool = True, k: Optional[int] = None):
    """Return a sorted dict by values with first k reserved if provided.
    """
    sorted_items = list(sorted(dic.items(), key=lambda item: item[1], reverse=desc))
    return dict(sorted_items[:k])


def banner_message(message: str, sym="^"):
    """Print a banner message surrounded by special symbols.
    """
    print()
    message = message.upper()
    banner_len = len(message) + 4
    message = " " * ((banner_len - len(message)) // 2) + message
    message = message + " " * (banner_len - len(message))
    print(message)
    print(sym * banner_len)
    print()


# subclass of logging.Formatter
class RuntimeFormatter(logging.Formatter):
    """Auxiliary class for runtime formatting in the logger.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = time.time()

    def formatTime(self, record, datefmt=None):
        """Record relative runtime in hr:min:sec format。
        """
        duration = datetime.datetime.utcfromtimestamp(record.created - self.start_time)
        elapsed = duration.strftime("%H:%M:%S")
        return "{}".format(elapsed)


def create_logger(model_name: str, saved_path: str):
    """Create logger for both console info and saved info。
    """
    logger = logging.getLogger(model_name)
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(f"{saved_path}/{model_name}.log")
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter = RuntimeFormatter(
        "[Time: %(asctime)s] - [PID: %(process)d] - [Model: %(name)s] \n%(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def read_oaei_mappings(rdf_file: str):
    """To read mapping files in the OAEI rdf format.
    
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
