# Copyright 2023 Yuan He. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from typing import List

from deeponto.onto import Ontology
from deeponto.align.mapping import ReferenceMapping, EntityMapping

def get_ignored_class_index(onto: Ontology):
    """Get an index for filtering classes that are marked as not used in alignment.
    
    This is indicated by the special class annotation `use_in_alignment` with the following IRI: 
        http://oaei.ontologymatching.org/bio-ml/ann/use_in_alignment
    """
    ignored_class_index = defaultdict(lambda: False)
    for class_iri, class_obj in onto.owl_classes.items():
        use_in_alignment = onto.get_owl_object_annotations(class_obj, "http://oaei.ontologymatching.org/bio-ml/ann/use_in_alignment")
        if use_in_alignment and str(use_in_alignment[0]).lower() == "false":
            ignored_class_index[class_iri] = True
    return ignored_class_index

def filter_mappings(mappings: List[EntityMapping], ignored_class_index: dict):
    """Filter prediction mappings that involve classes to be ignored.
    """
    results = []
    for m in mappings:
        if ignored_class_index[m.head] or ignored_class_index[m.tail]:
            continue
        results.append(m)
    return results
