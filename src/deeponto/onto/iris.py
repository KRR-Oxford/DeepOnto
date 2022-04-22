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
namespaces["http://bioontology.org/projects/ontologies/fma/fmaOwlDlComponent_2_0#"] = "fma_largebio:"
namespaces["http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#"] = "ncit_largebio:"
namespaces["http://www.ihtsdo.org/snomed#"] = "snomed_largebio:"

# UMLS: new largebio
namespaces["http://purl.org/sig/ont/fma/"] = "fma:"
# nci has obo: prefix so skip it
namespaces["http://snomed.info/id/"] = "snomed:"

# Mondo
namespaces["http://linkedlifedata.com/resource/umls/id/"] = "umls:"
namespaces["http://identifiers.org/hgnc/"] = "hgnc:"  # HGNC symbol: https://registry.identifiers.org/registry/hgnc.symbol
namespaces["http://identifiers.org/mesh/"] = "mesh:" 
namespaces["http://identifiers.org/snomedct/"] = "snomedct:"
namespaces["http://purl.obolibrary.org/obo/"] = "obo:"
namespaces["http://www.orpha.net/ORDO/"] = "ordo:"
namespaces["http://www.ebi.ac.uk/efo/"] = "efo:"
namespaces["http://omim.org/entry/"] = "omim:"
namespaces["http://www.omim.org/phenotypicSeries/"] = "omimps:"  # follow the naming convention as in MONDO
namespaces["http://identifiers.org/meddra/"] = "meddra:"
namespaces["http://identifiers.org/medgen/"] = "medgen:"

inv_namespaces = {v: k for k, v in namespaces.items()}
