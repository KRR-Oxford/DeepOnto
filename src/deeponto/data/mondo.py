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

from deeponto.utils import download_onto


##################################################################################
###                                data download                               ###
##################################################################################

mondo_url = "http://purl.obolibrary.org/obo/mondo/mondo-with-equivalents.owl"


def download_mondo(saved_path: str):
    """Download mondo ontology
    """
    return download_onto(mondo_url, saved_path)
