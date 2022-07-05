# Copyright 2022 Jiaoyan Chen (KRR-Oxford). All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import click
import json
import datetime
import random
import os
import sys
import pickle
import torch
import math
import numpy as np
from transformers import TrainingArguments

sys.path.append("./src")

from deeponto.onto import OntologySubs
from deeponto.bert.tune import BERTTrainerSubs


# Generate negative subsumptions for training subsumptions
# Generate samples
def sample(onto, config, subsumptions, pos_dup=1, neg_dup=1):
    neg_subsumptions = list()
    for subs in subsumptions:
        c1 = subs[0]
        for _ in range(neg_dup):
            neg_c = onto.get_negative_sample(subclass_str=c1, subsumption_type=config['task']['subsumption_type'])
            neg_subsumptions.append([c1, neg_c])
    pos_samples = onto.subsumptions_to_samples(subsumptions=subsumptions, config=config, sample_label=1,
                                               subsumption_type=config['task']['subsumption_type'])
    pos_samples = pos_dup * pos_samples
    neg_samples = onto.subsumptions_to_samples(subsumptions=neg_subsumptions, config=config, sample_label=0,
                                               subsumption_type=config['task']['subsumption_type'])
    if len(neg_samples) < len(pos_samples):
        neg_samples = neg_samples + [random.choice(neg_samples) for _ in range(len(pos_samples) - len(neg_samples))]
    if len(neg_samples) > len(pos_samples):
        pos_samples = pos_samples + [random.choice(pos_samples) for _ in range(len(neg_samples) - len(pos_samples))]
    print('pos_samples: %d, neg_samples: %d' % (len(pos_samples), len(neg_samples)))
    all_samples = [s for s in pos_samples + neg_samples if s[0] != '' and s[1] != '']
    random.shuffle(all_samples)
    return all_samples


def evaluate(onto, config, batch_size, tokenize, classifier, device, target_subsumptions, test_type='test'):
    MRR_sum, hits1_sum, hits5_sum, hits10_sum = 0, 0, 0, 0
    MRR, Hits1, Hits5, Hits10 = 0, 0, 0, 0
    size_sum, size_n = 0, 0
    for k0, test in enumerate(target_subsumptions):
        subcls, gt = test[0], test[1]
        candidates = test[1:]

        candidate_subsumptions = [[subcls, c] for c in candidates]
        candidate_scores = np.zeros(len(candidate_subsumptions))

        for k1, candidate_subsumption in enumerate(candidate_subsumptions):
            samples = onto.subsumptions_to_samples(subsumptions=[candidate_subsumption], config=config,
                                                   sample_label=None,
                                                   subsumption_type=config['task']['subsumption_type'])
            sample_size = len(samples)
            size_sum += sample_size
            size_n += 1
            scores = np.zeros(sample_size)
            batch_num = math.ceil(sample_size / batch_size)
            for i in range(batch_num):
                j = (i + 1) * batch_size if (i + 1) * batch_size <= sample_size else sample_size
                inputs = tokenize(samples[i * batch_size:j])
                inputs.to(device)
                with torch.no_grad():
                    batch_scores = classifier(inputs)
                scores[i * batch_size:j] = batch_scores.cpu().numpy()
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


@click.command()
@click.option("-c", "--config_file", type=click.Path(exists=True), default="./config/intra_subs_foodon.json")
@click.option("-t", "--prompt_type", type=str, default="isolated", help="isolated, traversal, path")
@click.option("-h", "--prompt_hop", type=int, default=1, help="1,2")
@click.option("-m", "--prompt_max_subsumptions", type=int, default=4)
@click.option("-l", "--max_length", type=int, default=128)
@click.option("-o", "--output_dir", type=str, default="fine-tuned-bert")
def intra_subs(
        config_file: str,
        prompt_type: str,
        prompt_hop: int,
        prompt_max_subsumptions: int,
        max_length: int,
        output_dir: str
):
    config = json.load(open(config_file))

    # overwrite the configurations by program arguments
    config['task']['prompt_type'] = prompt_type
    config['task']['prompt_hop'] = prompt_hop
    config['task']['prompt_max_subsumptions'] = prompt_max_subsumptions
    config['task']['max_length'] = max_length
    config['fine-tune']['output_dir'] = output_dir

    start_time = datetime.datetime.now()

    read_subsumptions = lambda file_name: [line.strip().split(',') for line in open(file_name).readlines()]
    test_subsumptions = read_subsumptions(config['task']['test_subsumption_file'])
    valid_subsumptions = read_subsumptions(config['task']['valid_subsumption_file'])
    train_subsumptions = read_subsumptions(config['task']['train_subsumption_file'])
    train_num = len(train_subsumptions)
    print('Positive train subsumptions: %d' % len(train_subsumptions))
    onto = OntologySubs(onto_file=config['task']['onto_file'], label_property=config['task']['label_property'])
    print('%d named classes' % len(onto.named_classes))
    n = 0
    for k in onto.iri_label:
        n += len(onto.iri_label[k])
    print('%.1f labels per class' % (n / len(onto.named_classes)))
    onto.set_masked_subsumptions(subsumptions_to_mask=test_subsumptions + valid_subsumptions)

    if config['task']['use_pickle'] and os.path.exists(config['task']['tr_pickle']) \
            and os.path.exists(config['task']['va_pickle']):
        tr = pickle.load(open(config['task']['tr_pickle'], 'rb'))
        va = pickle.load(open(config['task']['va_pickle'], 'rb'))
    else:
        tr = sample(onto, config, subsumptions=train_subsumptions, pos_dup=config['fine-tune']['train_pos_dup'],
                    neg_dup=config['fine-tune']['train_neg_dup'])
        va = sample(onto, config, subsumptions=valid_subsumptions)
        pickle.dump(tr, open(config['task']['tr_pickle'], 'wb'))
        pickle.dump(va, open(config['task']['va_pickle'], 'wb'))

    end_time = datetime.datetime.now()
    print('data pre-processing costs %.1f minutes' % ((end_time - start_time).seconds / 60))

    start_time = datetime.datetime.now()
    torch.cuda.empty_cache()
    bert_trainer = BERTTrainerSubs(config['fine-tune']['pretrained'], train_data=tr, val_data=va,
                                   max_length=config['task']['max_length'],
                                   early_stop=config['fine-tune']['early_stop'])

    batch_size = config['task']["batch_size"]
    epoch_steps = len(bert_trainer.tra) // batch_size  # total steps of an epoch
    logging_steps = int(epoch_steps * 0.02) if int(epoch_steps * 0.02) > 0 else 5
    eval_steps = 5 * logging_steps
    training_args = TrainingArguments(
        output_dir=config['fine-tune']['output_dir'],
        num_train_epochs=config['fine-tune']['num_epochs'],
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_ratio=config['fine-tune']['warm_up_ratio'],
        weight_decay=0.01,
        logging_steps=logging_steps,
        logging_dir=f"{config['fine-tune']['output_dir']}/tb",
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

    if config['fine-tune']['do-fine-tune'] and config['task']['use_sub_special_token']:
        bert_trainer.add_special_tokens(['<SUB>'])

    bert_trainer.train(train_args=training_args, do_fine_tune=config['fine-tune']['do-fine-tune'])
    if config['fine-tune']['do-fine-tune']:
        bert_trainer.trainer.save_model(
            output_dir=os.path.join(config['fine-tune']['output_dir'], 'fine-tuned-checkpoint'))
        print('fine-tuning done, fine-tuned model saved')
    else:
        print('pretrained or fine-tuned model loaded.')
    end_time = datetime.datetime.now()
    print('Fine-tuning costs %.1f minutes' % ((end_time - start_time).seconds / 60))

    bert_trainer.model.eval()
    device = torch.device(f"cuda") if torch.cuda.is_available() else torch.device("cpu")
    bert_trainer.model.to(device)
    tokenize = lambda x: bert_trainer.tokenizer(x, max_length=config['task']['max_length'], truncation=True,
                                                padding=True,
                                                return_tensors="pt")
    softmax = torch.nn.Softmax(dim=1)
    classifier = lambda x: softmax(bert_trainer.model(**x).logits)[:, 1]

    evaluate(onto, config, batch_size, tokenize, classifier, device, target_subsumptions=valid_subsumptions,
             test_type='valid')
    evaluate(onto, config, batch_size, tokenize, classifier, device, target_subsumptions=test_subsumptions,
             test_type='test')
    print('\n ------------------------- config: ---------------------------\n')
    print(json.dumps(config, indent=4, sort_keys=True))
    print('\n ------------------------- done! ---------------------------\n\n\n')


if __name__ == "__main__":
    intra_subs()
