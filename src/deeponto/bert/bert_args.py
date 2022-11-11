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
"""Class for input arguments of a BERT model"""

from transformers import TrainingArguments
from typing import Optional, Union
import torch


class BertArguments:
    def __init__(
        self,
        bert_checkpoint: str,
        output_dir: str,
        num_epochs: float,
        batch_size_for_training: int,
        batch_size_for_prediction: int,
        max_length: int,
        device_num: int,
        early_stop_patience: Optional[int],  # if not specified, no early stopping is performed
        resume_from_ckp: Optional[Union[bool, str]],  # None; True; specific_checkpoint_dir
    ):

        # basic arguments
        self.bert_checkpoint = bert_checkpoint

        # training arguments
        self.output_dir = output_dir
        self.num_epochs = num_epochs
        self.batch_size_for_training = batch_size_for_training
        self.batch_size_for_prediction = batch_size_for_prediction
        self.max_length = max_length
        self.device_num = device_num
        self.early_stop_patience = early_stop_patience
        self.early_stop = True if early_stop_patience else False
        self.resume_from_ckp = resume_from_ckp

    def generate_training_args(
        self,
        training_data_size: int,
        metric_for_best_model: Optional[str] = None,
        greater_is_better: Optional[bool] = None,
    ) -> TrainingArguments:

        # regularizing the steps
        epoch_steps = training_data_size // self.batch_size_for_training  # total steps of an epoch
        if torch.cuda.device_count() > 0:
            epoch_steps = epoch_steps // torch.cuda.device_count()  # to deal with multi-gpus case
        # keep logging steps consisitent even for small batch size
        # report logging on every 0.02 epoch
        logging_steps = int(epoch_steps * 0.02)
        # eval on every 0.1 epoch
        eval_steps = 5 * logging_steps

        return TrainingArguments(
            output_dir=self.output_dir,
            # max_steps=eval_steps*4 + 1,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size_for_training,
            per_device_eval_batch_size=self.batch_size_for_training,
            warmup_ratio=0.0,
            weight_decay=0.01,
            logging_steps=logging_steps,
            logging_dir=f"{self.output_dir}/tensorboard",
            eval_steps=eval_steps,
            evaluation_strategy="steps",
            do_train=True,
            do_eval=True,
            save_steps=eval_steps,
            load_best_model_at_end=True,
            save_total_limit=1,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
        )
