# Copyright 2023 Jiaoyan Chen. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# @paper(
#     "Contextual Semantic Embeddings for Ontology Subsumption Prediction (World Wide Web Journal)",
# )

import os
import torch
import math
import datetime
import numpy as np
from transformers import TrainingArguments
from yacs.config import CfgNode

from deeponto.onto import Ontology
from .bert_classifier import BERTSubsumptionClassifierTrainer
from .text_semantics import SubsumptionSample

DEFAULT_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "default_config_intra.yaml")


class BERTSubsIntraPipeline:
    def __init__(self, onto: Ontology, config: CfgNode):
        self.onto = onto
        self.config = config
        self.sampler = SubsumptionSample(onto=onto, config=config)
        start_time = datetime.datetime.now()

        read_subsumptions = lambda file_name: [line.strip().split(',') for line in open(file_name).readlines()]
        test_subsumptions = read_subsumptions(config.test_subsumption_file)
        valid_subsumptions = read_subsumptions(config.valid_subsumption_file)
        train_subsumptions = read_subsumptions(config.train_subsumption_file)
        train_num = len(train_subsumptions)
        print('Positive train subsumptions: %d' % train_num)

        n = 0
        for k in self.sampler.named_classes:
            n += len(self.sampler.iri_label[k])
        print('%.1f labels per class' % (n / len(self.sampler.named_classes)))

        tr = self.sampler.generate_samples(subsumptions=train_subsumptions)
        va = self.sampler.generate_samples(subsumptions=valid_subsumptions, duplicate=False)

        end_time = datetime.datetime.now()
        print('data pre-processing costs %.1f minutes' % ((end_time - start_time).seconds / 60))

        start_time = datetime.datetime.now()
        torch.cuda.empty_cache()
        bert_trainer = BERTSubsumptionClassifierTrainer(config.fine_tune.pretrained, train_data=tr, val_data=va,
                                                        max_length=config.prompt.max_length, early_stop=config.fine_tune.early_stop)

        epoch_steps = len(bert_trainer.tra) // config.fine_tune.batch_size  # total steps of an epoch
        logging_steps = int(epoch_steps * 0.02) if int(epoch_steps * 0.02) > 0 else 5
        eval_steps = 5 * logging_steps
        training_args = TrainingArguments(
            output_dir=config.fine_tune.output_dir,
            num_train_epochs=config.fine_tune.num_epochs,
            per_device_train_batch_size=config.fine_tune.batch_size,
            per_device_eval_batch_size=config.fine_tune.batch_size,
            warmup_ratio=config.fine_tune.warm_up_ratio,
            weight_decay=0.01,
            logging_steps=logging_steps,
            logging_dir=f"{config.fine_tune.output_dir}/tb",
            eval_steps=eval_steps,
            evaluation_strategy="steps",
            do_train=True,
            do_eval=True,
            save_steps=eval_steps,
            load_best_model_at_end=True,
            save_total_limit=1,
            metric_for_best_model="accuracy",
            greater_is_better=True
        )
        if config.fine_tune.do_fine_tune and (config.prompt.prompt_type == 'traversal' or (config.prompt.prompt_type == 'path' and config.prompt.use_sub_special_token)):
            bert_trainer.add_special_tokens(['<SUB>'])

        bert_trainer.train(train_args=training_args, do_fine_tune=config.fine_tune.do_fine_tune)
        if config.fine_tune.do_fine_tune:
            bert_trainer.trainer.save_model(output_dir=os.path.join(config.fine_tune.output_dir, 'fine-tuned-checkpoint'))
            print('fine-tuning done, fine-tuned model saved')
        else:
            print('pretrained or fine-tuned model loaded.')
        end_time = datetime.datetime.now()
        print('Fine-tuning costs %.1f minutes' % ((end_time - start_time).seconds / 60))

        bert_trainer.model.eval()
        device = torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")
        bert_trainer.model.to(device)
        self.tokenize = lambda x: bert_trainer.tokenizer(x, max_length=config.prompt.max_length, truncation=True, padding=True, return_tensors="pt")
        softmax = torch.nn.Softmax(dim=1)
        self.classifier = lambda x: softmax(bert_trainer.model(**x).logits)[:, 1]

        self.evaluate(target_subsumptions=valid_subsumptions, test_type='valid')
        self.evaluate(target_subsumptions=test_subsumptions, test_type='test')
        print('\n ------------------------- done! ---------------------------\n\n\n')


    def evaluate(self, target_subsumptions, test_type='test'):
        MRR_sum, hits1_sum, hits5_sum, hits10_sum = 0, 0, 0, 0
        MRR, Hits1, Hits5, Hits10 = 0, 0, 0, 0
        size_sum, size_n = 0, 0
        device = torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")
        for k0, test in enumerate(target_subsumptions):
            subcls, gt = test[0], test[1]
            candidates = test[1:]

            candidate_subsumptions = [[subcls, c] for c in candidates]
            candidate_scores = np.zeros(len(candidate_subsumptions))

            for k1, candidate_subsumption in enumerate(candidate_subsumptions):
                samples = self.sampler.subsumptions_to_samples(subsumptions=[candidate_subsumption], sample_label=None)
                sample_size = len(samples)
                size_sum += sample_size
                size_n += 1
                scores = np.zeros(sample_size)
                batch_num = math.ceil(sample_size / self.config.evaluation.batch_size)
                for i in range(batch_num):
                    j = (i + 1) * self.config.evaluation.batch_size if (i + 1) * self.config.evaluation.batch_size <= sample_size else sample_size
                    inputs = self.tokenize(samples[i * self.config.evaluation.batch_size:j])
                    inputs.to(device)
                    with torch.no_grad():
                        batch_scores = self.classifier(inputs)
                    scores[i * self.config.evaluation.batch_size:j] = batch_scores.cpu().numpy()
                candidate_scores[k1] = np.average(scores)

            sorted_indexes = np.argsort(candidate_scores)[::-1]
            sorted_classes = [candidates[i] for i in sorted_indexes]

            rank = sorted_classes.index(gt) + 1
            MRR_sum += 1.0 / rank
            hits1_sum += 1 if gt in sorted_classes[:1] else 0
            hits5_sum += 1 if gt in sorted_classes[:5] else 0
            hits10_sum += 1 if gt in sorted_classes[:10] else 0
            num = k0 + 1
            MRR, Hits1, Hits5, Hits10 = MRR_sum / num, hits1_sum / num, hits5_sum / num, hits10_sum / num
            if num % 500 == 0:
                print('\n%d tested, MRR: %.3f, Hits@1: %.3f, Hits@5: %.3f, Hits@10: %.3f\n' % (num, MRR, Hits1, Hits5, Hits10))
        print('\n[%s], MRR: %.3f, Hits@1: %.3f, Hits@5: %.3f, Hits@10: %.3f\n' % (test_type, MRR, Hits1, Hits5, Hits10))
        print('%.2f samples per testing subsumption' % (size_sum / size_n))
