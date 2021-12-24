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

from collections import defaultdict

# stored ontology iris and their abbreviations
namespaces = defaultdict(str)

# largebio
namespaces["http://bioontology.org/projects/ontologies/fma/fmaOwlDlComponent_2_0#"] = "fma:"
namespaces["http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#"] = "nci:"
namespaces["http://www.ihtsdo.org/snomed#"] = "snomed:"

# phenotype
namespaces["http://purl.obolibrary.org/obo/"] = "obo:"
namespaces["http://www.orpha.net/ORDO/"] = "ordo:"

# most recent version of SNOMED (accessed on May 2021)
namespaces["http://snomed.info/id/"] = "snomed:"

inv_namespaces = {v: k for k, v in namespaces.items()}
