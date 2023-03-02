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

from typing import Iterable, List, Dict, Tuple, Union
import re
from collections import defaultdict
from itertools import chain
import math
from transformers import AutoTokenizer
import spacy
from spacy.lang.en import English
import xml.etree.ElementTree as ET


class TextUtils:
    """Provides text processing utilities."""

    @staticmethod
    def process_annotation_literal(annotation_literal: str, apply_lowercasing: bool = True, normalise_identifiers: bool = False):
        """Pre-process an annotation literal string.

        Args:
            annotation_literal (str): A literal string of an entity's annotation.
            apply_lowercasing (bool): A boolean that determines lowercasing or not. Defaults to `True`.
            normalise_identifiers (bool): Whether to normalise annotation text that is in the Java identifier format. Defaults to `False`.

        Returns:
            (str): the processed annotation literal string.
        """

        # replace the underscores with spaces
        annotation_literal = annotation_literal.replace("_", " ")

        # if the annotation literal is a valid identifier with first letter capitalised
        # we suspect that it could be a Java style identifier that needs to be split
        if normalise_identifiers and annotation_literal[0].isupper() and annotation_literal.isidentifier():
            annotation_literal = TextUtils.split_java_identifier(annotation_literal)

        # lowercase the annotation literal if specfied
        if apply_lowercasing:
            annotation_literal = annotation_literal.lower()

        return annotation_literal

    @staticmethod
    def split_java_identifier(java_style_identifier: str):
        r"""Split words in java's identifier style into natural language phrase.

        Examples:
            - `"SuperNaturalPower"` $\rightarrow$ `"Super Natural Power"`
            - `"APIReference"` $\rightarrow$ `"API Reference"`
            - `"Covid19"` $\rightarrow$ `"Covid 19"`
        """
        # split at every capital letter or number (numbers are treated as capital letters)
        raw_words = re.findall("([0-9A-Z][a-z]*)", java_style_identifier)
        words = []
        capitalized_word = ""
        for i, w in enumerate(raw_words):

            # the above regex pattern will split at capitals
            # so the capitalized words are split into characters
            # i.e., (len(w) == 1)
            if len(w) == 1:
                capitalized_word += w
                # edge case for the last word
                if i == len(raw_words) - 1:
                    words.append(capitalized_word)

            # if the the current w is a full word, save the previous
            # cached capitalized_word and also save current full word
            elif capitalized_word:
                words.append(capitalized_word)
                words.append(w)
                capitalized_word = ""

            # just save the current full word otherwise
            else:
                words.append(w)

        return " ".join(words)


class Tokenizer:
    """A Tokenizer class for both sub-word (pre-trained) and word (rule-based) level tokenization."""

    def __init__(self, tokenizer_type: str):
        self.type = tokenizer_type
        self._tokenizer = None  # hidden tokenizer
        self.tokenize = None  # the tokenization method

    def __call__(self, texts: Union[str, List[str]]):
        if isinstance(texts, str):
            return self.tokenize(texts)
        else:
            return list(chain.from_iterable(self.tokenize(t) for t in texts))

    @classmethod
    def from_pretrained(cls, pretrained_path: str = "bert-base-uncased"):
        """(Based on **transformers**) Load a sub-word level tokenizer from pre-trained model."""
        instance = cls("pre-trained")
        instance._tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
        instance.tokenize = instance._tokenizer.tokenize
        return instance

    @classmethod
    def from_rule_based(cls):
        """(Based on **spacy**) Load a word-level (rule-based) tokenizer."""
        spacy.prefer_gpu()
        instance = cls("rule-based")
        instance._tokenizer = English()
        instance.tokenize = lambda texts: [word.text for word in instance._tokenizer(texts).doc]
        return instance


class InvertedIndex:
    r"""Inverted index built from a text index.

    Attributes:
        tokenizer (Tokenizer): A tokenizer instance to be used.
        original_index (defaultdict): A dictionary where the values are text strings to be tokenized.
        constructed_index (defaultdict): A dictionary that acts as the inverted index of `original_index`.
    """

    def __init__(self, index: defaultdict, tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        self.original_index = index
        self.constructed_index = defaultdict(list)
        for k, v in self.original_index.items():
            # value is a list of strings
            for token in self.tokenizer(v):
                self.constructed_index[token].append(k)

    def idf_select(self, texts: Union[str, List[str]], pool_size: int = 200):
        """Given a list of tokens, select a set candidates based on the inverted document frequency (idf) scores.

        We use `idf` instead of  `tf` because labels have different lengths and thus tf is not a fair measure.
        """
        candidate_pool = defaultdict(lambda: 0)
        # D := number of "documents", i.e., number of "keys" in the original index
        D = len(self.original_index)
        for token in self.tokenizer(texts):
            # each token is associated with some classes
            potential_candidates = self.constructed_index[token]
            if not potential_candidates:
                continue
            # We use idf instead of tf because the text for each class is of different length, tf is not a fair measure
            # inverse document frequency: with more classes to have the current token tk, the score decreases
            idf = math.log10(D / len(potential_candidates))
            for candidate in potential_candidates:
                # each candidate class is scored by sum(idf)
                candidate_pool[candidate] += idf
        candidate_pool = list(sorted(candidate_pool.items(), key=lambda item: item[1], reverse=True))
        # print(f"Select {min(len(candidate_pool), pool_size)} candidates.")
        # select the first K ranked
        return candidate_pool[:pool_size]
