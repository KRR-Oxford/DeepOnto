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
"""
BERTStatic class for handling BERT embeddings and pre-trained/fine-tuned BERT models in eval mode

"static" here means no gradient shift happened...
"""

from typing import List, Tuple
import torch
from transformers import AutoModelForSequenceClassification

from deeponto.bert import BertArguments
from . import BertStaticBase


class BertStaticForSequenceClassification(BertStaticBase):
    def __init__(
        self, bert_args: BertArguments,
    ):
        super().__init__(bert_args)
        self.softmax = torch.nn.Softmax(dim=1).to(self.device)

    def load_model(self):
        return AutoModelForSequenceClassification.from_pretrained(
            self.args.bert_checkpoint, output_hidden_states=True
        )

    def __call__(self, sent_pairs: List[Tuple[str, str]]):
        """Pipeline for (binary) sequence classification,
        returning the scores of being predicted as the positive class (index=1)
        """
        inputs = self.proc_input(sent_pairs)
        with torch.no_grad():
            return self.softmax(self.model(**inputs).logits)[:, 1]
