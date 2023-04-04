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

import random
import argparse
import sys
import warnings

from yacs.config import CfgNode
from deeponto.onto import Ontology
from deeponto.subs.bertsubs import SubsumptionSample
from deeponto.subs.bertsubs import BERTSubsIntraPipeline, DEFAULT_CONFIG_FILE
from deeponto.utils import FileUtils

'''
    partition the declared subsumptions into train, valid (--valid_ratio) and test (--test_ratio)
    when subsumption_type == named_class:
        a test sample is composed of two named classes: a subclass, a superclass (GT), 
        and at most --test_max_neg_size false superclasses are extracted from the GT's neighbourhood
    when subsumption_type == restriction:
        a sample is composed of a named class (subclass), an existential restriction (superclass GT), 
        and at most --test_max_neg_size false restrictions are randomly extracted from all existential restrictions 
        (this is different from the evaluation setting in our WWW J paper).
'''

parser = argparse.ArgumentParser()
parser.add_argument('--onto_file', type=str, default='/home/jiaoyan/bertsubs_data/foodon-merged.0.4.8.owl')
parser.add_argument('--valid_ratio', type=float, default=0.05)
parser.add_argument('--test_ratio', type=float, default=0.1)
parser.add_argument('--test_max_neg_size', type=int, default=40)
parser.add_argument('--max_depth', type=int, default=3)
parser.add_argument('--max_width', type=int, default=8)
parser.add_argument('--subsumption_type', type=str, default='restriction', help='restriction or named_class')
parser.add_argument('--train_file', type=str, default='./train_subsumptions_r.csv')
parser.add_argument('--valid_file', type=str, default='./valid_subsumptions_r.csv')
parser.add_argument('--test_file', type=str, default='./test_subsumptions_r.csv')
parser.add_argument('--evaluate_onto_file', type=str, default='./foodon.owl')
FLAGS, unparsed = parser.parse_known_args()

print('\n---- Evaluation data processing starts ----\n')
onto = Ontology(owl_path=FLAGS.onto_file)
all_subsumptions = onto.get_subsumption_axioms(entity_type='Classes')

subsumptions = []
if FLAGS.subsumption_type == 'restriction':
    for subs in all_subsumptions:
        if not onto.check_deprecated(owl_object=subs.getSubClass()) and \
                not onto.check_named_entity(owl_object=subs.getSuperClass()) and \
                SubsumptionSample.is_basic_existential_restriction(complex_class_str=str(subs.getSuperClass())):
            subsumptions.append(subs)
    restrictions = []
    for complexC in onto.get_asserted_complex_classes():
        if SubsumptionSample.is_basic_existential_restriction(complex_class_str=str(complexC)):
            restrictions.append(complexC)
elif FLAGS.subsumption_type == 'named_class':
    for subs in all_subsumptions:
        if not onto.check_deprecated(owl_object=subs.getSubClass()) and onto.check_named_entity(
                owl_object=subs.getSuperClass()) and not onto.check_deprecated(owl_object=subs.getSuperClass()):
            subsumptions.append(subs)
else:
    warnings.warn('Unknown super_class_type %s' % FLAGS.super_class_type)
    print("Unknown super_class_type")
    sys.exit(0)

valid_size = int(len(subsumptions) * FLAGS.valid_ratio)
test_size = int(len(subsumptions) * FLAGS.test_ratio)
valid_subsumptions = subsumptions[0:valid_size]
test_subsumptions = subsumptions[valid_size:(valid_size + test_size)]
train_subsumptions = subsumptions[(valid_size + test_size):]
print('train subsumptions: %d' % len(train_subsumptions))
print('valid subsumptions: %d' % len(valid_subsumptions))
print('test subsumptions: %d' % len(test_subsumptions))


def get_test_neg_candidates(subclass, gt):
    all_nebs, seeds = set(), [gt]
    depth = 1
    while depth <= FLAGS.max_depth:
        new_seeds = set()
        for seed in seeds:
            nebs = set()
            for nc_iri in onto.reasoner.get_inferred_sub_entities(seed, direct=True) + onto.reasoner.get_inferred_super_entities(seed, direct=True):
                nc = onto.owl_classes[nc_iri]
                if onto.check_named_entity(owl_object=nc) and not onto.check_deprecated(owl_object=nc):
                    nebs.add(nc)
            new_seeds = new_seeds.union(nebs)
            all_nebs = all_nebs.union(nebs)
        depth += 1
        seeds = random.sample(new_seeds, FLAGS.max_width) if len(new_seeds) > FLAGS.max_width else new_seeds
    all_nebs = all_nebs - {onto.owl_classes[iri] for iri in onto.reasoner.get_inferred_super_entities(subclass, direct=False)} - {subclass}
    if len(all_nebs) > FLAGS.test_max_neg_size:
        return random.sample(all_nebs, FLAGS.test_max_neg_size)
    else:
        return list(all_nebs)


def context_candidate(output_file, target_subs):
    with open(output_file, 'w') as ff:
        size_sum = 0
        size_num = dict()
        m = 0
        for subs0 in target_subs:
            subcls, supcls = subs0.getSubClass(), subs0.getSuperClass()
            neg_candidates = get_test_neg_candidates(subclass=subcls, gt=supcls)
            size = len(neg_candidates)
            size_sum += size
            size_num[size] = size_num[size] + 1 if size in size_num else 1
            if size > 0:
                s = ','.join([str(c.getIRI()) for c in neg_candidates])
                ff.write('%s,%s,%s\n' % (str(subcls.getIRI()), str(supcls.getIRI()), s))
                m += 1
        print('\t The distribution of negative candidate size:')
        for size in range(FLAGS.test_max_neg_size + 1):
            if size in size_num:
                print('\t size: %d, num: %d' % (size, size_num[size]))
            else:
                print('\t size: %d, num: 0' % size)
        print('\t %d subsumptions saved; average neg candidate size: %.2f' % (m, size_sum / m))


def get_test_neg_candidates_restriction(subcls):
    neg_restrictions = list()
    n = FLAGS.test_max_neg_size * 2 if FLAGS.test_max_neg_size * 2 <= len(restrictions) else len(restrictions)
    for r in random.sample(restrictions, n):
        if not onto.reasoner.check_subsumption(sub_entity=subcls, super_entity=r):
            neg_restrictions.append(r)
            if len(neg_restrictions) >= FLAGS.test_max_neg_size:
                break
    return neg_restrictions


if FLAGS.subsumption_type == 'restriction':
    with open(FLAGS.train_file, 'w') as f:
        for subs in train_subsumptions:
            c1, c2 = subs.getSubClass(), subs.getSuperClass()
            f.write('%s,%s\n' % (str(c1.getIRI()), str(c2)))
    with open(FLAGS.valid_file, 'w') as f:
        sizes = 0
        for subs in valid_subsumptions:
            c1, c2 = subs.getSubClass(), subs.getSuperClass()
            c2_neg = get_test_neg_candidates_restriction(subcls=c1)
            sizes += len(c2_neg)
            strs = [str(r) for r in c2_neg]
            f.write('%s,%s,%s\n' % (str(c1.getIRI()), str(c2), ','.join(strs)))
        print('valid candidate negative avg. size: %.1f' % (sizes / len(valid_subsumptions)))
    with open(FLAGS.test_file, 'w') as f:
        sizes = 0
        for subs in test_subsumptions:
            c1, c2 = subs.getSubClass(), subs.getSuperClass()
            c2_neg = get_test_neg_candidates_restriction(subcls=c1)
            sizes += len(c2_neg)
            strs = [str(r) for r in c2_neg]
            f.write('%s,%s,%s\n' % (str(c1.getIRI()), str(c2), ','.join(strs)))
        print('test candidate negative avg. size: %.1f' % (sizes / len(test_subsumptions)))

else:
    with open(FLAGS.train_file, 'w') as f:
        for subs in train_subsumptions:
            c1, c2 = subs.getSubClass(), subs.getSuperClass()
            f.write('%s,%s\n' % (str(c1.getIRI()), str(c2.getIRI())))

    print('\n---- context candidates for validation subsumptions ----')
    context_candidate(output_file=FLAGS.valid_file, target_subs=valid_subsumptions)

    print('\n---- context candidates for test subsumptions ----')
    context_candidate(output_file=FLAGS.test_file, target_subs=test_subsumptions)

for subs in valid_subsumptions + test_subsumptions:
    onto.remove_axiom(owl_axiom=subs)
onto.save_onto(save_path=FLAGS.evaluate_onto_file)
print('\n---- Evaluation data processing done ----\n')

print('\n---- Evaluation starts ----\n')
config = CfgNode(FileUtils.load_file(DEFAULT_CONFIG_FILE))
config.subsumption_type = FLAGS.subsumption_type
config.train_subsumption_file = FLAGS.train_file
config.valid_subsumption_file = FLAGS.valid_file
config.test_subsumption_file = FLAGS.test_file
config.onto_file = FLAGS.evaluate_onto_file
onto2 = Ontology(owl_path=FLAGS.evaluate_onto_file)
pipeline = BERTSubsIntraPipeline(onto=onto2, config=config)
print('\n---- Evaluation done ----\n')