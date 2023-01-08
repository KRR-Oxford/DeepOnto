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
Fine-tuning BERT with the classtext pair datasets extracted from ontologies
Code inspired by: https://huggingface.co/transformers/training.html
"""
from typing import List, Tuple, Any, Optional

import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSequenceClassification

from deeponto.bert import BertArguments
from . import BertTrainerBase


class BertTrainerForSequenceClassification(BertTrainerBase):
    def __init__(
        self,
        bert_args: BertArguments,
        train_data: List,
        val_data: List,
        test_data: Optional[List] = None,
    ):
        super().__init__(bert_args, train_data, val_data, test_data)

    def load_model(self):
        return AutoModelForSequenceClassification.from_pretrained(self.args.bert_checkpoint)

    def load_dataset(self, data: List[Tuple[str, str, Any]]) -> Dataset:
        """For sequence classification, we have two sentences (sent1, sent2, cls_label) as input
        """
        data_df = pd.DataFrame(data, columns=["sent1", "sent2", "labels"])
        dataset = Dataset.from_pandas(data_df)
        dataset = dataset.map(
            lambda examples: self.tokenizer.tkz(
                examples["sent1"],
                examples["sent2"],
                max_length=self.args.max_length,
                truncation=True,
            ),
            batched=True,
            batch_size=1024,
            num_proc=10,
        )
        return dataset
