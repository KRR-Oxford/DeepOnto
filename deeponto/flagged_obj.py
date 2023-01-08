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
"""Super class for objects that need to switch src2tgt and tgt2src flags"""

from __future__ import annotations
from typing import TYPE_CHECKING
from itertools import cycle
from deeponto.onto import Ontology

# to avoid circular imports
if TYPE_CHECKING:
    from deeponto.onto import Ontology

class FlaggedObj:
    def __init__(self, src_onto: Ontology, tgt_onto: Ontology):
        self.src_onto = src_onto
        self.tgt_onto = tgt_onto
        self.flag_set = cycle(["src2tgt", "tgt2src"])
        self.flag = next(self.flag_set)

    def renew(self):
        """Renew alignment direction to src2tgt
        """
        while self.flag != "src2tgt":
            self.switch()

    def switch(self):
        """Switch alignment direction
        """
        self.src_onto, self.tgt_onto = self.tgt_onto, self.src_onto
        self.flag = next(self.flag_set)
