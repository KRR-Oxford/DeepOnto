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
import warnings
import random

import torch
import math
import datetime
import numpy as np
from typing import List
from transformers import TrainingArguments
from yacs.config import CfgNode

from deeponto.onto import Ontology
from .bert_classifier import BERTSubsumptionClassifierTrainer
from .text_semantics import SubsumptionSampler

DEFAULT_CONFIG_FILE_INTRA = os.path.join(os.path.dirname(__file__), "default_config_intra.yaml")


class BERTSubsIntraPipeline:
    r"""Class for the intra-ontology subsumption prediction setting of BERTSubs.

    Attributes:
        onto (Ontology): The target ontology.
        config (CfgNode): The configuration for BERTSubs.
        sampler (SubsumptionSample): The subsumption sampler for BERTSubs.
    """

    def __init__(self, onto: Ontology, config: CfgNode):
        self.onto = onto
        self.config = config
        self.sampler = SubsumptionSampler(onto=onto, config=config)
        start_time = datetime.datetime.now()

        n = 0
        for k in self.sampler.named_classes:
            n += len(self.sampler.iri_label[k])
        print(
            "%d named classes, %.1f labels per class"
            % (len(self.sampler.named_classes), n / len(self.sampler.named_classes))
        )

        read_subsumptions = lambda file_name: [line.strip().split(",") for line in open(file_name).readlines()]
        test_subsumptions = (
            None
            if config.test_subsumption_file is None or config.test_subsumption_file == "None"
            else read_subsumptions(config.test_subsumption_file)
        )

        # The train/valid subsumptions are not given. They will be extracted from the given ontology:
        if config.train_subsumption_file is None or config.train_subsumption_file == "None":
            subsumptions0 = self.extract_subsumptions_from_ontology(
                onto=onto, subsumption_type=config.subsumption_type
            )
            random.shuffle(subsumptions0)
            valid_size = int(len(subsumptions0) * config.valid.valid_ratio)
            train_subsumptions0, valid_subsumptions0 = subsumptions0[valid_size:], subsumptions0[0:valid_size]
            train_subsumptions, valid_subsumptions = [], []
            if config.subsumption_type == "named_class":
                for subs in train_subsumptions0:
                    c1, c2 = subs.getSubClass(), subs.getSuperClass()
                    train_subsumptions.append([str(c1.getIRI()), str(c2.getIRI())])

                size_sum = 0
                for subs in valid_subsumptions0:
                    c1, c2 = subs.getSubClass(), subs.getSuperClass()
                    neg_candidates = BERTSubsIntraPipeline.get_test_neg_candidates_named_class(
                        subclass=c1, gt=c2, max_neg_size=config.valid.max_neg_size, onto=onto
                    )
                    size = len(neg_candidates)
                    size_sum += size
                    if size > 0:
                        item = [str(c1.getIRI()), str(c2.getIRI())] + [str(c.getIRI()) for c in neg_candidates]
                        valid_subsumptions.append(item)
                print("\t average neg candidate size in validation: %.2f" % (size_sum / len(valid_subsumptions)))

            elif config.subsumption_type == "restriction":
                for subs in train_subsumptions0:
                    c1, c2 = subs.getSubClass(), subs.getSuperClass()
                    train_subsumptions.append([str(c1.getIRI()), str(c2)])

                restrictions = BERTSubsIntraPipeline.extract_restrictions_from_ontology(onto=onto)
                print("restrictions: %d" % len(restrictions))
                size_sum = 0
                for subs in valid_subsumptions0:
                    c1, c2 = subs.getSubClass(), subs.getSuperClass()
                    c2_neg = BERTSubsIntraPipeline.get_test_neg_candidates_restriction(
                        subcls=c1, max_neg_size=config.valid.max_neg_size, restrictions=restrictions, onto=onto
                    )
                    size_sum += len(c2_neg)
                    item = [str(c1.getIRI()), str(c2)] + [str(r) for r in c2_neg]
                    valid_subsumptions.append(item)
                print("valid candidate negative avg. size: %.1f" % (size_sum / len(valid_subsumptions)))
            else:
                warnings.warn("Unknown subsumption type %s" % config.subsumption_type)
                sys.exit(0)

        # The train/valid subsumptions are given:
        else:
            train_subsumptions = read_subsumptions(config.train_subsumption_file)
            valid_subsumptions = read_subsumptions(config.valid_subsumption_file)

        print("Positive train/valid subsumptions: %d/%d" % (len(train_subsumptions), len(valid_subsumptions)))
        tr = self.sampler.generate_samples(subsumptions=train_subsumptions)
        va = self.sampler.generate_samples(subsumptions=valid_subsumptions, duplicate=False)

        end_time = datetime.datetime.now()
        print("data pre-processing costs %.1f minutes" % ((end_time - start_time).seconds / 60))

        start_time = datetime.datetime.now()
        torch.cuda.empty_cache()
        bert_trainer = BERTSubsumptionClassifierTrainer(
            config.fine_tune.pretrained,
            train_data=tr,
            val_data=va,
            max_length=config.prompt.max_length,
            early_stop=config.fine_tune.early_stop,
        )

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
            greater_is_better=True,
        )
        if config.fine_tune.do_fine_tune and (
            config.prompt.prompt_type == "traversal"
            or (config.prompt.prompt_type == "path" and config.prompt.use_sub_special_token)
        ):
            bert_trainer.add_special_tokens(["<SUB>"])

        bert_trainer.train(train_args=training_args, do_fine_tune=config.fine_tune.do_fine_tune)
        if config.fine_tune.do_fine_tune:
            bert_trainer.trainer.save_model(
                output_dir=os.path.join(config.fine_tune.output_dir, "fine-tuned-checkpoint")
            )
            print("fine-tuning done, fine-tuned model saved")
        else:
            print("pretrained or fine-tuned model loaded.")
        end_time = datetime.datetime.now()
        print("Fine-tuning costs %.1f minutes" % ((end_time - start_time).seconds / 60))

        bert_trainer.model.eval()
        self.device = torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")
        bert_trainer.model.to(self.device)
        self.tokenize = lambda x: bert_trainer.tokenizer(
            x, max_length=config.prompt.max_length, truncation=True, padding=True, return_tensors="pt"
        )
        softmax = torch.nn.Softmax(dim=1)
        self.classifier = lambda x: softmax(bert_trainer.model(**x).logits)[:, 1]

        self.evaluate(target_subsumptions=valid_subsumptions, test_type="valid")
        if test_subsumptions is not None:
            if config.test_type == "evaluation":
                self.evaluate(target_subsumptions=test_subsumptions, test_type="test")
            elif config.test_type == "prediction":
                self.predict(target_subsumptions=test_subsumptions)
            else:
                warnings.warn("Unknown test_type: %s" % config.test_type)
        print("\n ------------------------- done! ---------------------------\n\n\n")

    def score(self, samples: List[List]):
        r"""The scoring function based on the fine-tuned BERT classifier.

        Args:
            samples (List[Tuple]): A list of input sentence pairs to be scored.
        """
        sample_size = len(samples)
        scores = np.zeros(sample_size)
        batch_num = math.ceil(sample_size / self.config.evaluation.batch_size)
        for i in range(batch_num):
            j = (
                (i + 1) * self.config.evaluation.batch_size
                if (i + 1) * self.config.evaluation.batch_size <= sample_size
                else sample_size
            )
            inputs = self.tokenize(samples[i * self.config.evaluation.batch_size : j])
            inputs.to(self.device)
            with torch.no_grad():
                batch_scores = self.classifier(inputs)
            scores[i * self.config.evaluation.batch_size : j] = batch_scores.cpu().numpy()
        return scores

    def evaluate(self, target_subsumptions: List[List], test_type: str = "test"):
        r"""Test and calculate the metrics for a given list of subsumption pairs.

        Args:
            target_subsumptions (List[Tuple]): A list of subsumption pairs.
            test_type (str): `test` for testing or `valid` for validation.
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
                samples = self.sampler.subsumptions_to_samples(subsumptions=[candidate_subsumption], sample_label=None)
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
                print(
                    "\n%d tested, MRR: %.3f, Hits@1: %.3f, Hits@5: %.3f, Hits@10: %.3f\n"
                    % (num, MRR, Hits1, Hits5, Hits10)
                )
        print(
            "\n[%s], MRR: %.3f, Hits@1: %.3f, Hits@5: %.3f, Hits@10: %.3f\n" % (test_type, MRR, Hits1, Hits5, Hits10)
        )
        print("%.2f samples per testing subsumption" % (size_sum / size_n))

    def predict(self, target_subsumptions: List[List]):
        r"""Predict a score for each given subsumption in the list.

        The scores will be saved in `test_subsumption_scores.csv`.

        Args:
            target_subsumptions (List[List]): Each item is a list where the first element is a fixed ontology class $C$,
                and the remaining elements are potential (candidate) super-classes of $C$.
        """
        out_lines = []
        for test in target_subsumptions:
            subcls, candidates = test[0], test[1:]
            candidate_subsumptions = [[subcls, c] for c in candidates]
            candidate_scores = []

            for candidate_subsumption in candidate_subsumptions:
                samples = self.sampler.subsumptions_to_samples(subsumptions=[candidate_subsumption], sample_label=None)
                scores = self.score(samples=samples)
                candidate_scores.append(np.average(scores))

            out_lines.append(",".join([str(i) for i in candidate_scores]))

        out_file = "test_subsumption_scores.csv"
        with open(out_file, "w") as f:
            for line in out_lines:
                f.write("%s\n" % line)
        print("Predicted subsumption scores are saved to %s" % out_file)

    @staticmethod
    def extract_subsumptions_from_ontology(onto: Ontology, subsumption_type: str):
        r"""Extract target subsumptions from a given ontology.

        Args:
            onto (Ontology): The target ontology.
            subsumption_type (str): the type of subsumptions, options are `"named_class"` or `"restriction"`.

        """
        all_subsumptions = onto.get_subsumption_axioms(entity_type="Classes")
        subsumptions = []
        if subsumption_type == "restriction":
            for subs in all_subsumptions:
                if (
                    not onto.check_deprecated(owl_object=subs.getSubClass())
                    and not onto.check_named_entity(owl_object=subs.getSuperClass())
                    and SubsumptionSampler.is_basic_existential_restriction(
                        complex_class_str=str(subs.getSuperClass())
                    )
                ):
                    subsumptions.append(subs)
        elif subsumption_type == "named_class":
            for subs in all_subsumptions:
                c1, c2 = subs.getSubClass(), subs.getSuperClass()
                if (
                    onto.check_named_entity(owl_object=c1)
                    and not onto.check_deprecated(owl_object=c1)
                    and onto.check_named_entity(owl_object=c2)
                    and not onto.check_deprecated(owl_object=c2)
                ):
                    subsumptions.append(subs)
        else:
            warnings.warn("\nUnknown subsumption type: %s\n" % subsumption_type)
        return subsumptions

    @staticmethod
    def extract_restrictions_from_ontology(onto: Ontology):
        r"""Extract basic existential restriction from an ontology.

        Args:
            onto (Ontology): The target ontology.
        Returns:
            restrictions (List): a list of existential restrictions.
        """
        restrictions = []
        for complexC in onto.get_asserted_complex_classes():
            if SubsumptionSampler.is_basic_existential_restriction(complex_class_str=str(complexC)):
                restrictions.append(complexC)
        return restrictions

    @staticmethod
    def get_test_neg_candidates_restriction(subcls, max_neg_size, restrictions, onto):
        """Get a list of negative candidate class restrictions for testing."""
        neg_restrictions = list()
        n = max_neg_size * 2 if max_neg_size * 2 <= len(restrictions) else len(restrictions)
        for r in random.sample(restrictions, n):
            if not onto.reasoner.check_subsumption(sub_entity=subcls, super_entity=r):
                neg_restrictions.append(r)
                if len(neg_restrictions) >= max_neg_size:
                    break
        return neg_restrictions

    @staticmethod
    def get_test_neg_candidates_named_class(subclass, gt, max_neg_size, onto, max_depth=3, max_width=8):
        """Get a list of negative candidate named classes for testing."""
        all_nebs, seeds = set(), [gt]
        depth = 1
        while depth <= max_depth:
            new_seeds = set()
            for seed in seeds:
                nebs = set()
                for nc_iri in onto.reasoner.get_inferred_sub_entities(
                    seed, direct=True
                ) + onto.reasoner.get_inferred_super_entities(seed, direct=True):
                    nc = onto.owl_classes[nc_iri]
                    if onto.check_named_entity(owl_object=nc) and not onto.check_deprecated(owl_object=nc):
                        nebs.add(nc)
                new_seeds = new_seeds.union(nebs)
                all_nebs = all_nebs.union(nebs)
            depth += 1
            seeds = random.sample(new_seeds, max_width) if len(new_seeds) > max_width else new_seeds
        all_nebs = (
            all_nebs
            - {onto.owl_classes[iri] for iri in onto.reasoner.get_inferred_super_entities(subclass, direct=False)}
            - {subclass}
        )
        if len(all_nebs) > max_neg_size:
            return random.sample(all_nebs, max_neg_size)
        else:
            return list(all_nebs)
