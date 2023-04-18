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
import sys
import random
import datetime
import warnings
import math
from yacs.config import CfgNode
from typing import List
import numpy as np

import torch
from transformers import TrainingArguments

from deeponto.onto import Ontology
from .bert_classifier import BERTSubsumptionClassifierTrainer
from .text_semantics import SubsumptionSampler
from .pipeline_intra import BERTSubsIntraPipeline

DEFAULT_CONFIG_FILE_INTER = os.path.join(os.path.dirname(__file__), "default_config_inter.yaml")


class BERTSubsInterPipeline:
    r"""Class for the model training and prediction/validation pipeline of inter-ontology subsumption of BERTSubs.

    Attributes:
        src_onto (Ontology): Source ontology (the sub-class side).
        tgt_onto (Ontology): Target ontology (the super-class side).
        config (CfgNode): Configuration.
        src_sampler (SubsumptionSampler): Object for sampling-related functions of the source ontology.
        tgt_sampler (SubsumptionSampler): Object for sampling-related functions of the target ontology.
    """

    def __init__(self, src_onto: Ontology, tgt_onto: Ontology, config: CfgNode):
        self.src_onto = src_onto
        self.tgt_onto = tgt_onto
        self.config = config
        self.config.label_property = self.config.src_label_property
        self.src_sampler = SubsumptionSampler(onto=self.src_onto, config=self.config)
        self.config.label_property = self.config.tgt_label_property
        self.tgt_sampler = SubsumptionSampler(onto=self.tgt_onto, config=self.config)
        start_time = datetime.datetime.now()

        read_subsumptions = lambda file_name: [line.strip().split(',') for line in open(file_name).readlines()]
        test_subsumptions = None if config.test_subsumption_file is None or config.test_subsumption_file == 'None' \
            else read_subsumptions(config.test_subsumption_file)
        valid_subsumptions = None if config.valid_subsumption_file is None or config.valid_subsumption_file == 'None' \
            else read_subsumptions(config.valid_subsumption_file)

        if config.use_ontology_subsumptions_training:
            src_subsumptions = BERTSubsIntraPipeline.extract_subsumptions_from_ontology(onto=self.src_onto,
                                                                                        subsumption_type=config.subsumption_type)
            tgt_subsumptions = BERTSubsIntraPipeline.extract_subsumptions_from_ontology(onto=self.tgt_onto,
                                                                                        subsumption_type=config.subsumption_type)
            src_subsumptions0, tgt_subsumptions0 = [], []
            if config.subsumption_type == 'named_class':
                for subs in src_subsumptions:
                    c1, c2 = subs.getSubClass(), subs.getSuperClass()
                    src_subsumptions0.append([str(c1.getIRI()), str(c2.getIRI())])
                for subs in tgt_subsumptions:
                    c1, c2 = subs.getSubClass(), subs.getSuperClass()
                    tgt_subsumptions0.append([str(c1.getIRI()), str(c2.getIRI())])
            elif config.subsumption_type == 'restriction':
                for subs in src_subsumptions:
                    c1, c2 = subs.getSubClass(), subs.getSuperClass()
                    src_subsumptions0.append([str(c1.getIRI()), str(c2)])
                for subs in tgt_subsumptions:
                    c1, c2 = subs.getSubClass(), subs.getSuperClass()
                    tgt_subsumptions0.append([str(c1.getIRI()), str(c2)])
                restrictions = BERTSubsIntraPipeline.extract_restrictions_from_ontology(onto=self.tgt_onto)
                print('restrictions in the target ontology: %d' % len(restrictions))
            else:
                warnings.warn('Unknown subsumption type %s' % config.subsumption_type)
                sys.exit(0)
            print('Positive train subsumptions from the source/target ontology: %d/%d' % (
                len(src_subsumptions0), len(tgt_subsumptions0)))

            src_tr = self.src_sampler.generate_samples(subsumptions=src_subsumptions0)
            tgt_tr = self.tgt_sampler.generate_samples(subsumptions=tgt_subsumptions0)
        else:
            src_tr, tgt_tr = [], []

        if config.train_subsumption_file is None or config.train_subsumption_file == 'None':
            tr = src_tr + tgt_tr
        else:
            train_subsumptions = read_subsumptions(config.train_subsumption_file)
            tr = self.inter_ontology_sampling(subsumptions=train_subsumptions, pos_dup=config.fine_tune.train_pos_dup,
                                              neg_dup=config.fine_tune.train_neg_dup)
            tr = tr + src_tr + tgt_tr

        if len(tr) == 0:
            warnings.warn('No training samples extracted')
            if config.fine_tune.do_fine_tune:
                sys.exit(0)

        end_time = datetime.datetime.now()
        print('data pre-processing costs %.1f minutes' % ((end_time - start_time).seconds / 60))

        start_time = datetime.datetime.now()
        torch.cuda.empty_cache()
        bert_trainer = BERTSubsumptionClassifierTrainer(config.fine_tune.pretrained, train_data=tr,
                                                        val_data=tr[0:int(len(tr) / 5)],
                                                        max_length=config.prompt.max_length,
                                                        early_stop=config.fine_tune.early_stop)

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
        if config.fine_tune.do_fine_tune and (config.prompt.prompt_type == 'traversal' or (
                config.prompt.prompt_type == 'path' and config.prompt.use_sub_special_token)):
            bert_trainer.add_special_tokens(['<SUB>'])

        bert_trainer.train(train_args=training_args, do_fine_tune=config.fine_tune.do_fine_tune)
        if config.fine_tune.do_fine_tune:
            bert_trainer.trainer.save_model(
                output_dir=os.path.join(config.fine_tune.output_dir, 'fine-tuned-checkpoint'))
            print('fine-tuning done, fine-tuned model saved')
        else:
            print('pretrained or fine-tuned model loaded.')
        end_time = datetime.datetime.now()
        print('Fine-tuning costs %.1f minutes' % ((end_time - start_time).seconds / 60))

        bert_trainer.model.eval()
        self.device = torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")
        bert_trainer.model.to(self.device)
        self.tokenize = lambda x: bert_trainer.tokenizer(x, max_length=config.prompt.max_length, truncation=True,
                                                         padding=True, return_tensors="pt")
        softmax = torch.nn.Softmax(dim=1)
        self.classifier = lambda x: softmax(bert_trainer.model(**x).logits)[:, 1]

        if valid_subsumptions is not None:
            self.evaluate(target_subsumptions=valid_subsumptions, test_type='valid')

        if test_subsumptions is not None:
            if config.test_type == 'evaluation':
                self.evaluate(target_subsumptions=test_subsumptions, test_type='test')
            elif config.test_type == 'prediction':
                self.predict(target_subsumptions=test_subsumptions)
            else:
                warnings.warn("Unknown test_type: %s" % config.test_type)
        print('\n ------------------------- done! ---------------------------\n\n\n')

    def inter_ontology_sampling(self, subsumptions: List[List], pos_dup: int = 1, neg_dup: int = 1):
        r"""Transform inter-ontology subsumptions to two-string samples
        Args:
            subsumptions (List[List]): A list of subsumptions; each subsumption is composed of two IRIs.
            pos_dup (int): Positive sample duplication.
            neg_dup (int): Negative sample duplication.
        """
        pos_samples = list()
        for subs in subsumptions:
            sub_strs = self.src_sampler.subclass_to_strings(subcls=subs[0])
            sup_strs = self.tgt_sampler.supclass_to_strings(supcls=subs[1],
                                                            subsumption_type=self.config.subsumption_type)
            for sub_str in sub_strs:
                for sup_str in sup_strs:
                    pos_samples.append([sub_str, sup_str, 1])
        pos_samples = pos_dup * pos_samples

        neg_subsumptions = list()
        for subs in subsumptions:
            for _ in range(neg_dup):
                neg_c = self.tgt_sampler.get_negative_sample(subclass_iri=subs[1],
                                                             subsumption_type=self.config.subsumption_type)
                neg_subsumptions.append([subs[0], neg_c])

        neg_samples = list()
        for subs in neg_subsumptions:
            sub_strs = self.src_sampler.subclass_to_strings(subcls=subs[0])
            sup_strs = self.tgt_sampler.supclass_to_strings(supcls=subs[1],
                                                            subsumption_type=self.config.subsumption_type)
            for sub_str in sub_strs:
                for sup_str in sup_strs:
                    neg_samples.append([sub_str, sup_str, 0])

        if len(neg_samples) < len(pos_samples):
            neg_samples = neg_samples + [random.choice(neg_samples) for _ in range(len(pos_samples) - len(neg_samples))]
        if len(neg_samples) > len(pos_samples):
            pos_samples = pos_samples + [random.choice(pos_samples) for _ in range(len(neg_samples) - len(pos_samples))]
        print('training mappings, pos_samples: %d, neg_samples: %d' % (len(pos_samples), len(neg_samples)))
        all_samples = [s for s in pos_samples + neg_samples if s[0] != '' and s[1] != '']
        return all_samples

    def inter_ontology_subsumption_to_sample(self, subsumption: List):
        r"""Transform an inter ontology subsumption into a sample (a two-string list).
        
        Args:
            subsumption (List): a subsumption composed of two IRIs.
        """
        subcls, supcls = subsumption[0], subsumption[1]
        substrs = self.src_sampler.subclass_to_strings(subcls=subcls)
        supstrs = self.tgt_sampler.supclass_to_strings(supcls=supcls, subsumption_type='named_class')
        samples = list()
        for substr in substrs:
            for supstr in supstrs:
                samples.append([substr, supstr])
        return samples

    def score(self, samples):
        r"""Score the samples with the classifier.
        
        Args:
            samples (List[List]): Each item is a list with two strings (input).
        """
        sample_size = len(samples)
        scores = np.zeros(sample_size)
        batch_num = math.ceil(sample_size / self.config.evaluation.batch_size)
        for i in range(batch_num):
            j = (i + 1) * self.config.evaluation.batch_size \
                if (i + 1) * self.config.evaluation.batch_size <= sample_size else sample_size
            inputs = self.tokenize(samples[i * self.config.evaluation.batch_size:j])
            inputs.to(self.device)
            with torch.no_grad():
                batch_scores = self.classifier(inputs)
            scores[i * self.config.evaluation.batch_size:j] = batch_scores.cpu().numpy()
        return scores

    def evaluate(self, target_subsumptions: List[List], test_type: str = 'test'):
        r"""Test and calculate the metrics according to a given list of subsumptions.
        
        Args:
            target_subsumptions (List[List]): A list of subsumptions, each of which of is a two-component list `(subclass_iri, super_class_iri_or_str)`.
            test_type (str): `"test"` or `"valid"`.
        """
        MRR_sum, hits1_sum, hits5_sum, hits10_sum = 0, 0, 0, 0
        MRR, Hits1, Hits5, Hits10 = 0, 0, 0, 0
        size_sum, size_n = 0, 0
        for k0, test in enumerate(target_subsumptions):
            subcls, gt = test[0], test[1]
            candidates = test[1:]
            candidate_subsumptions = [[subcls, c] for c in candidates]
            candidate_scores = np.zeros(len(candidate_subsumptions))
            for k1, candidate_subsumption in enumerate(candidate_subsumptions):
                samples = self.inter_ontology_subsumption_to_sample(subsumption=candidate_subsumption)
                size_sum += len(samples)
                size_n += 1
                scores = self.score(samples=samples)
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
                print('\n%d tested, MRR: %.3f, Hits@1: %.3f, Hits@5: %.3f, Hits@10: %.3f\n' % (
                    num, MRR, Hits1, Hits5, Hits10))
        print('\n[%s], MRR: %.3f, Hits@1: %.3f, Hits@5: %.3f, Hits@10: %.3f\n' % (test_type, MRR, Hits1, Hits5, Hits10))
        print('%.2f samples per testing subsumption' % (size_sum / size_n))

    def predict(self, target_subsumptions: List[List]):
        r"""Predict a score for each given subsumption. 
        
        The scores will be saved in `test_subsumption_scores.csv`.
        
        Args:
            target_subsumptions (List[List]): Each item is a list with the first element as the sub-class,
                                              and the remaining elements as n candidate super-classes.
        """
        out_lines = []
        for test in target_subsumptions:
            subcls, candidates = test[0], test[1:]
            candidate_subsumptions = [[subcls, c] for c in candidates]
            candidate_scores = []

            for candidate_subsumption in candidate_subsumptions:
                samples = self.inter_ontology_subsumption_to_sample(subsumption=candidate_subsumption)
                scores = self.score(samples=samples)
                candidate_scores.append(np.average(scores))
            out_lines.append(','.join([str(i) for i in candidate_scores]))

        out_file = 'test_subsumption_scores.csv'
        with open(out_file, 'w') as f:
            for line in out_lines:
                f.write('%s\n' % line)
        print('Predicted subsumption scores are saved to %s' % out_file)
