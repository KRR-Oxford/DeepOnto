"""
Fine-tuning BERT with the classtext pair datasets extracted from ontologies
Code inspired by: https://huggingface.co/transformers/training.html
"""
from typing import List

import pandas as pd
from datasets import Dataset
from transformers import AutoModelForSequenceClassification

from deeponto.bert import BERTArgs
from . import BERTFineTune


class BERTFineTuneSeqClassifier(BERTFineTune):
    def __init__(
        self, bert_args: BERTArgs, train_data: List, val_data: List, test_data: List,
    ):
        super().__init__(bert_args, train_data, val_data, test_data)

    def load_model(self):
        return AutoModelForSequenceClassification.from_pretrained(self.args.bert_checkpoint)

    def load_dataset(self, data: List) -> Dataset:
        """For sequence classification, we have two sentences (sent1, sent2, cls_label) as input
        """
        data_df = pd.DataFrame(data, columns=["sent1", "sent2", "labels"])
        dataset = Dataset.from_pandas(data_df)
        dataset = dataset.map(
            lambda examples: self.tokenizer(
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
