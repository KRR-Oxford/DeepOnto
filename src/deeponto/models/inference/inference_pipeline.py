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

from typing import Optional
import torch
import enlighten
from openprompt.prompts import ManualTemplate
from openprompt.prompts import ManualVerbalizer
from openprompt import PromptForClassification
from openprompt import PromptDataLoader
from openprompt.plms import load_plm

from deeponto.utils import get_device


class InferencePipeline:
    def __init__(
        self,
        model_name: str = "roberta",
        model_path: str = "roberta-large",
        template: str = "",
        labels: list = [],
        label_words: dict = {},
        device_num: int = 0,
    ):
        # load language model
        self.plm, self.tokenizer, self.model_config, self.wrapper_class = load_plm(
            model_name, model_path
        )

        # load template and verbalizer
        self.prompt_template = ManualTemplate(text=template, tokenizer=self.tokenizer)
        self.inference_classes = labels
        self.inference_label_words = label_words
        self.prompt_verbalizer = ManualVerbalizer(
            classes=self.inference_classes,
            label_words=self.inference_label_words,
            tokenizer=self.tokenizer,
        )

        # load prompt model
        self.device = get_device(device_num)
        self.prompt_model = PromptForClassification(
            template=self.prompt_template, plm=self.plm, verbalizer=self.prompt_verbalizer,
        )
        self.prompt_model.to(self.device)

    def train(self, train_dataset: list, val_dataset: Optional[list]):
        """Train with few-show examples"""
        pass

    def run(self, inference_dataset: list):
        """Run inference over test dataset
        """
        # Load the inference dataset into device
        inference_data_loader = PromptDataLoader(
            dataset=inference_dataset,
            tokenizer=self.tokenizer,
            template=self.prompt_template,
            tokenizer_wrapper_class=self.wrapper_class,
        )
        for inp in inference_data_loader.tensor_dataset:
            inp.to(self.device)
        self.prompt_model.eval()
        preds = []
        truths = []
        with torch.no_grad():
            manager = enlighten.get_manager()
            pbar = manager.counter(total=len(inference_data_loader), desc="Running Inference", unit='sample')
            for inference_sample in inference_data_loader:
                logits = self.prompt_model(inference_sample)
                pred_idx = torch.argmax(logits, dim=-1)
                pred = self.inference_classes[pred_idx]
                ground_truth = self.inference_classes[inference_sample["label"]]
                # print(f"pred: {pred}; truth: {ground_truth}")
                # if pred == ground_truth:
                #     correct += 1
                preds.append(pred)
                truths.append(ground_truth)
                pbar.update()
        results = list(zip(preds, truths))
        acc = sum([p == g for p, g in results]) / len(results)
        print(f"Inference results over {len(results)} samples: {round(acc, 5)} (ACC)")
        return results, acc
