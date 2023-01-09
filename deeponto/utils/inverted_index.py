# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Class for constructing an inverted index."""

from collections import defaultdict

class InvertedIndex:
    
    def __init__(self, tokenizer):
        pass
    
    def build_inv_idx(self, tokenizer, cut: int = 0) -> None:
        """Create inverted index based on the extracted labels of an ontology

        Parameters
        ----------
        tokenizer : Tokenizer
            text tokenizer, word-level or sub-word-level
        cut : int, optional
            keep tokens with length > cut , by default 0
        """
        self.inv_idx = defaultdict(list)
        for cls_iri, cls_labs in self.iri2labs.items():
            for tk in tokenizer.tokenize_all(cls_labs):
                if len(tk) > cut:
                    self.inv_idx[tk].append(cls_iri)
        self.num_entries_inv_idx = len(self.inv_idx)