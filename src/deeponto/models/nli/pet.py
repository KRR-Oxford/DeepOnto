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
"""Reproducing PET Using OpenPrompt
"""

from typing import Callable, Optional
import torch
from tqdm.std import tqdm
from openprompt.prompts import ManualTemplate
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptForClassification, PromptForGeneration
from openprompt import PromptDataLoader
from openprompt.plms import load_plm

from deeponto.utils import get_device


ROBERTA_NLI_TEMPLATES = [
    lambda cls, sep: f'{{"meta":"premise"}}?{sep}{sep}{{"mask"}}, {{"meta":"hypothesis"}}',
    lambda cls, sep: f'"{{"meta":"premise"}}"?{sep}{sep}{{"mask"}}, "{{"meta":"hypothesis"}}"',
    lambda cls, sep: f'{{"meta":"premise"}}? {{"mask"}}, {{"meta":"hypothesis"}}',
    lambda cls, sep: f'"{{"meta":"premise"}}"? {{"mask"}}, "{{"meta":"hypothesis"}}"',
]

ROBERTA_NLI_LABEL_WORDS = [
    {"entailment": ["Right"], "neutral": ["Maybe"], "contradiction": ["Wrong"]},
    {"entailment": ["Yes"], "neutral": ["Maybe"], "contradiction": ["No"]},
    {"entailment": ["Right", "Yes"], "neutral": ["Maybe"], "contradiction": ["Wrong", "No"]},
]

ROBERTA_NLI_LABELS = ["entailment", "neutral", "contradiction"]


def run_pet(
    inference_dataset: list,
    train_dataset: Optional[list] = None,
    model_name: str = "roberta",
    model_path: str = "roberta-large",
    template: Callable = ROBERTA_NLI_TEMPLATES[0],
    labels: list = ROBERTA_NLI_LABELS,
    label_words: dict = ROBERTA_NLI_LABEL_WORDS[0],
    device_num: int = 0
):
    """Run the PET pipeline for the language inference task.
    """
    plm, tokenizer, model_config, WrapperClass = load_plm(model_name, model_path)
    cl = tokenizer.cls_token
    sep = tokenizer.sep_token

    # model set-up
    prompt_template = ManualTemplate(text=template(cl, sep), tokenizer=tokenizer,)
    prompt_verbalizer = ManualVerbalizer(
        classes=labels, label_words=label_words, tokenizer=tokenizer,
    )
    prompt_model = PromptForClassification(
        template = prompt_template,
        plm = plm,
        verbalizer = prompt_verbalizer,
    )
    device = get_device(device_num)
    prompt_model.to(device)
        
    # model training
    def train(train_dataset: list):
        """Train with few"""
        pass
    
    train(train_dataset=train_dataset)
    
    # inference
    inference_data_loader = PromptDataLoader(
        dataset = inference_dataset,
        tokenizer = tokenizer,
        template = prompt_template,
        tokenizer_wrapper_class=WrapperClass,
    )
    for inp in inference_data_loader.tensor_dataset:
        inp.to(device)
    prompt_model.eval()
    preds = []
    truths = []
    correct = 0
    # making zero-shot inference using pretrained MLM with prompt
    prompt_model.eval()
    total = 0
    with torch.no_grad():
        for inference_sample in tqdm(inference_data_loader, desc='NLI Inference'):
            total += 1
            logits = prompt_model(inference_sample)
            pred_idx = torch.argmax(logits, dim = -1)
            pred = labels[pred_idx]
            ground_truth = labels[inference_sample["label"]]
            # print(f"pred: {pred}; truth: {ground_truth}")
            if pred == ground_truth:
                correct += 1
            preds.append(pred)
            truths.append(ground_truth)
    acc = correct / total
    print(f"Inference results over {total} samples: {round(acc, 5)} (ACC)")
    
    return preds, truths
