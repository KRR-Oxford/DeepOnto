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

from collections import defaultdict
from typing import Callable, Optional, List
from datasets import load_dataset
from openprompt.data_utils import InputExample
from openprompt import PromptDataLoader
from openprompt.prompt_base import Template
from openprompt.plms.utils import TokenizerWrapper
from transformers.tokenization_utils import PreTrainedTokenizer
from yacs.config import CfgNode

from deeponto.utils import read_table, credit
from deeponto import SavedObj

# default label indexing for inference classfication
trinary_label2idx = {"entailment": 0, "neutral": 1, "contradiction": 2}
binary_label2idx = {"entailment": 0, "contradiction": 1}


@credit("openprompt", "https://github.com/thunlp/OpenPrompt/blob/main/experiments/cli.py")
def build_dataloader(
    dataset: List,
    template: Template,
    tokenizer: PreTrainedTokenizer,
    tokenizer_wrapper_class: TokenizerWrapper,
    config: CfgNode,
    split: str,
):
    dataloader = PromptDataLoader(
        dataset=dataset,
        template=template,
        tokenizer=tokenizer,
        tokenizer_wrapper_class=tokenizer_wrapper_class,
        batch_size=config[split].batch_size,
        shuffle=config[split].shuffle_data,
        teacher_forcing=config[split].teacher_forcing
        if hasattr(config[split], "teacher_forcing")
        else None,
        predict_eos_token=True if config.task == "generation" else False,
        **config.dataloader,
    )
    return dataloader


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
