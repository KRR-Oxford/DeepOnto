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
"""Class for edit similarity OM system"""

from typing import Optional

from deeponto.onto.text import Tokenizer
from deeponto.onto import Ontology
from . import StringMatch


class EditSimilarity(StringMatch):
    def __init__(
        self,
        src_onto: Ontology,
        tgt_onto: Ontology,
        tokenizer: Tokenizer,
        cand_pool_size: Optional[int] = 200,
        n_best: Optional[int] = 10,
        saved_path: str = "",
    ):
        super().__init__(
            src_onto,
            tgt_onto,
            tokenizer,
            cand_pool_size,
            n_best,
            saved_path,
            use_edit_dist=True,
        )
