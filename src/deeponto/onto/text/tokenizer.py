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
"""Class for different types of tokenizers"""

from types import MethodType
from transformers import AutoTokenizer
import spacy
from itertools import chain


class Tokenizer:
    def __init__(self, tokenize: MethodType):
        """init with the input tokenizing method
        """
        self.tokenize = tokenize
        self.type = None

    def __call__(self, texts):
        return self.tokenize(texts)
    
    def tokenize_all(self, texts_list):
        return list(chain.from_iterable(self(txt) for txt in texts_list))

    @classmethod
    def from_pretrained(cls, pretrained_path: str = "bert-base-uncased"):
        """huggingface sub-word level tokenizer (statistics-based)
        """
        tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
        inst = cls(tokenizer.tokenize)
        inst.tkz = tokenizer
        inst.type = "pre-trained"
        return inst

    @classmethod
    def from_rule_based(cls, spacy_lib_path: str = "en_core_web_sm"):
        """spacy word-level tokenizer with rule-based analysis
        """
        nlp = spacy.load(spacy_lib_path)
        inst = cls(lambda texts: [word.text for word in nlp(texts).doc])
        inst.nlp = nlp
        inst.type = "rule-based"
        return inst
