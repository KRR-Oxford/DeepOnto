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
"""Templates for prompt-based inference
"""

PREMISE = '{"meta":"premise"}'
HYPOTHESIS = '{"meta":"hypothesis"}'
MASK = '{"mask"}'

def inference_templates(model_name: str):
    """Get manual templates for inference tasks.
    """
    # define special tokens
    cls_token, sep_token = "[CLS]", "[SEP]"
    if model_name == "roberta":
        cls_token = "<s>"
        sep_token = "</s></s>"
    return [
        f"{PREMISE}?{sep_token}{MASK}, {HYPOTHESIS}.",  # from NLI
        f'"{PREMISE}"?{sep_token}{MASK}, "{HYPOTHESIS}".',  # from NLI
        f"{PREMISE}? {MASK}, {HYPOTHESIS}.",  # from NLI
        f'"{PREMISE}"? {MASK}, "{HYPOTHESIS}".',  # from NLI
        # f'If {PREMISE}, then {MASK}, {HYPOTHESIS}.',  # more natural
        f'{PREMISE}. Is it true that {HYPOTHESIS}? {MASK}.' # more natural
    ]


def inference_label_words(include_neutral: bool = False):
    """Manual label words for verbalizer over inference tasks.
    """
    label_words = [
        {"entailment": ["Right"], "neutral": "Maybe", "contradiction": ["Wrong"]},
        {"entailment": ["Yes"], "neutral": "Maybe", "contradiction": ["No"]},
        {"entailment": ["Right", "Yes"], "neutral": "Maybe", "contradiction": ["Wrong", "No"]},
    ]
    labels = ["entailment", "neutral", "contradiction"]
    if not include_neutral:
        for lw in label_words:
            del lw["neutral"]
        labels = ["entailment", "contradiction"]
    return labels, label_words
