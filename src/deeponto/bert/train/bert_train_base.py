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
"""Class for BERT trainer"""

from typing import List, Optional

from datasets import Dataset
from sklearn.metrics import accuracy_score
from transformers import EarlyStoppingCallback, Trainer

from deeponto.bert import BertArguments
from deeponto.onto.text import Tokenizer


class BertTrainerBase:
    def __init__(
        self,
        bert_args: BertArguments,
        train_data: List,
        val_data: List,
        test_data: Optional[List] = None,
    ):

        self.args = bert_args
        self.tokenizer = Tokenizer.from_pretrained(self.args.bert_checkpoint)

        # pre-trained BERT model
        self.model = self.load_model()
        print(f"Load a BERT model from: {self.args.bert_checkpoint}")

        # load data (max_length is used for truncation)
        self.train_data = self.load_dataset(train_data)
        self.train_size = len(self.train_data)
        self.val_data = self.load_dataset(val_data)
        self.val_size = len(self.val_data)
        self.test_data = None
        self.test_size = -1
        if test_data:
            self.test_data = self.load_dataset(test_data)
            self.test_size = len(self.test_data)
        print(
            f"Loaded # of data: [train]={self.train_size};"
            + f"[val]={self.val_size}; [test]={self.test_size};"
        )

        # intialize the transformers.Trainer
        # TODO: more arguments are expected if we use more metrics
        self.train_args = self.args.generate_training_args(training_data_size=len(self.train_data))
        # print(self.train_args)
        self.trainer = Trainer(
            model=self.model,
            args=self.train_args,
            train_dataset=self.train_data,
            eval_dataset=self.val_data,
            compute_metrics=self.compute_metrics,
            tokenizer=self.tokenizer.tkz,
        )
        # add early stopping if needed
        if self.args.early_stop:
            self.trainer.add_callback(
                EarlyStoppingCallback(early_stopping_patience=self.args.early_stop_patience)
            )

    def train(self):
        self.trainer.train(resume_from_checkpoint=self.args.resume_from_ckp)

    def load_model(self):
        """Load a pre-trained BERT model
        """
        raise NotImplementedError

    def load_dataset(self, data: List) -> Dataset:
        """Load train/val/test data into transformers.dataset
        """
        raise NotImplementedError

    @staticmethod
    def compute_metrics(pred):
        """Keep more metrics in record
        """
        # TODO: currently only accuracy is added, will expect more in the future if needed
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc}
