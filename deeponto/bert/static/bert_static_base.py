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
from typing import List

from deeponto.bert import BertArguments
from deeponto.onto.text import Tokenizer
from deeponto.utils import get_device


class BertStaticBase:
    def __init__(self, bert_args: BertArguments):

        self.args = bert_args
        self.tokenizer = Tokenizer.from_pretrained(self.args.bert_checkpoint)

        # load the pre-trained BERT model and set it to eval mode (static)
        self.model = self.load_model()
        print(f"Load a BERT model from: {self.args.bert_checkpoint}")
        self.model.eval()

        # load the model into the GPU/CPU device
        self.device = get_device(device_num=self.args.device_num)
        self.model.to(self.device)

    def load_model(self):
        """Load a pre-trained BERT model
        """
        raise NotImplementedError

    def proc_input(self, sents: List):
        """Process BERT inputs and load them into the device
        """
        return self.tokenizer.tkz(
            sents,
            padding=True,
            return_tensors="pt",
            max_length=self.args.max_length,
            truncation=True,
        ).to(self.device)
