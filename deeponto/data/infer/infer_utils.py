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

import os
from collections import defaultdict
from typing import Callable, Optional
from datasets import load_dataset
from openprompt.data_utils import InputExample
from openprompt.data_utils.data_processor import DataProcessor
from openprompt.utils.logging import logger
from yacs.config import CfgNode

from deeponto.utils import read_table
from deeponto import SavedObj


class InferenceProcessor(DataProcessor):
    # TODO Test needed
    def __init__(self, apply_binary: bool):
        super().__init__()
        self.labels = ["entailment", "neutral", "contradiction"]
        self.label2idx = {"entailment": 0, "neutral": 1, "contradiction": 2}
        if apply_binary:
            self.labels = ["entailment", "contradiction"]
            self.label2idx = {"entailment": 0, "contradiction": 1}

    def get_examples(self, data_dir, split):
        data_path = os.path.join(data_dir, "{}.tsv".format(split))
        examples = load_prompt_data_from_table(data_path, self.label2idx)
        return examples


def load_inference_dataset(config: CfgNode, return_class=True, test=False):
    r"""A plm loader using a global config.
    It will load the train, valid, and test set (if exists) simulatenously.

    Args:
        config (:obj:`CfgNode`): The global config from the CfgNode.
        return_class (:obj:`bool`): Whether return the data processor class
                    for future usage.

    Returns:
        :obj:`Optional[List[InputExample]]`: The train dataset.
        :obj:`Optional[List[InputExample]]`: The valid dataset.
        :obj:`Optional[List[InputExample]]`: The test dataset.
        :obj:"
    """
    dataset_config = config.dataset

    processor = InferenceProcessor(True)

    train_dataset = None
    valid_dataset = None
    if not test:
        try:
            train_dataset = processor.get_train_examples(dataset_config.path)
        except FileNotFoundError:
            logger.warning(f"Has no training dataset in {dataset_config.path}.")
        try:
            valid_dataset = processor.get_dev_examples(dataset_config.path)
        except FileNotFoundError:
            logger.warning(f"Has no validation dataset in {dataset_config.path}.")

    test_dataset = None
    try:
        test_dataset = processor.get_test_examples(dataset_config.path)
    except FileNotFoundError:
        logger.warning(f"Has no test dataset in {dataset_config.path}.")
    # checking whether donwloaded.
    if (train_dataset is None) and \
       (valid_dataset is None) and \
       (test_dataset is None):
        logger.error("Dataset is empty. Either there is no download or the path is wrong. "+ \
        "If not downloaded, please `cd datasets/` and `bash download_xxx.sh`")
        exit()
    if return_class:
        return train_dataset, valid_dataset, test_dataset, processor
    else:
        return  train_dataset, valid_dataset, test_dataset



def load_prompt_data_from_huggingface(dataset_path: str = "multi_nli", *splits: str):
    """Load a datast from huggingface datasets and transform to openprompt examples.
    """
    data_dict = load_dataset(dataset_path)
    prompt_data_dict = dict()
    splits = list(splits) if splits else data_dict.keys()
    i = 0
    for split in splits:
        x_samples = []
        for samp in data_dict[split]:
            inp = InputExample(guid=i, meta=samp, label=samp["label"])
            x_samples.append(inp)
            i += 1
        prompt_data_dict[split] = x_samples
    return prompt_data_dict


def load_prompt_data_from_table(
    tabular_data_path: str,
    label2idx: dict,
    premise_pattern: Optional[
        Callable
    ] = lambda x: x,  # extra pattern for premise, default is no change
    hypothesis_pattern: Optional[
        Callable
    ] = lambda x: x,  # extra pattern for hypothesis, default is no change
):
    """Load an inference dataset containing (premise-hypothesis) pairs
    from the json file
    """
    df = read_table(tabular_data_path)
    prompt_data = []
    stats = defaultdict(lambda: 0)
    for i, dp in df.iterrows():
        samp = {
            "premise": premise_pattern(dp["Premise"]),
            "hypothesis": hypothesis_pattern(dp["Hypothesis"]),
            "label": dp["Label"],
        }
        label_idx = label2idx[dp["Label"]]
        inp = InputExample(guid=i, meta=samp, label=label_idx)
        prompt_data.append(inp)
        stats[dp["Label"]] += 1
    print("Load inference dataset with the following statistics:")
    SavedObj.print_json(stats)
    return prompt_data
