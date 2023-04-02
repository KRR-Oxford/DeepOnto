import random
import argparse
from yacs.config import CfgNode
from deeponto.onto import Ontology
from deeponto.subs.bertsubs import SubsumptionSample
from deeponto.subs.bertsubs import BERTSubsIntraPipeline, DEFAULT_CONFIG_FILE

'''
    partition the declared subsumptions into train (80%), valid (5%) and test (15%)
    when restriction == False:
        a test sample is composed of two named classes: a subclass, a superclass (GT), 
        and at most 50 false superclasses extracted from the GT's neighbourhood
    when restriction == True:
        a sample is composed of a named class (subclass), an existential restriction (superclass GT), 
        and at most 50 false restrictions extracted with relevant classes and relations.
'''

parser = argparse.ArgumentParser()
parser.add_argument('--onto_file', type=str, default='foodon-merged.0.4.8.owl')
parser.add_argument('--valid_ratio', type=float, default=0.05)
parser.add_argument('--test_ratio', type=float, default=0.15)
parser.add_argument('--test_max_neg_size', type=int, default=40)
parser.add_argument('--max_depth', type=int, default=3)
parser.add_argument('--max_width', type=int, default=8)
parser.add_argument('--restriction', type=bool, default=True)
parser.add_argument('--train_file', type=str, default='./tmp/train_subsumptions_r.csv')
parser.add_argument('--valid_file', type=str, default='./tmp/valid_subsumptions_r.csv')
parser.add_argument('--test_file', type=str, default='./tmp/test_subsumptions_r.csv')
parser.add_argument('--evaluate_onto_file', type=str, default='./tmp/foodon-merged.0.4.8.owl')
FLAGS, unparsed = parser.parse_known_args()

onto = Ontology(owl_path=FLAGS.onto_file)
named_classes = SubsumptionSample.extract_named_classes(onto=onto)
print('%d named classes' % len(named_classes))

all_subsumptions = onto.get_subsumption_axioms(entity_type='Classes')

subsumptions = []
if FLAGS.restriction:
    for subs in all_subsumptions:
        if not onto.check_deprecated(owl_object=subs[0]) and not onto.check_named_entity(owl_object=subs[1]):
            subsumptions.append(subs)
    restrictions = []
    for complexC in onto.get_asserted_complex_classes():
        if SubsumptionSample.is_basic_existential_restriction(complex_class_str=str(complexC)):
            restrictions.append(complexC)
else:
    for subs in all_subsumptions:
        if not onto.check_deprecated(owl_object=subs[0]) and onto.check_named_entity(owl_object=subs[1]) and not onto.check_deprecated(owl_object=subs[1]):
            subsumptions.append(subs)

valid_size = int(len(subsumptions) * FLAGS.valid_ratio)
test_size = int(len(subsumptions) * FLAGS.test_ratio)
valid_subsumptions = subsumptions[0:valid_size]
test_subsumptions = subsumptions[valid_size:(valid_size + test_size)]
train_subsumptions = subsumptions[(valid_size + test_size):]
print('train subsumptions: %d' % len(train_subsumptions))
print('valid subsumptions: %d' % len(valid_subsumptions))
print('test subsumptions: %d' % len(test_subsumptions))


def get_one_hop_neighbours(c):
    nebs_classes = set()
    for nc in onto.reasoner.get_inferred_sub_entities_of(c, direct=True) + onto.reasoner.get_inferred_super_entities_of(c, direct=True):
        if onto.check_named_entity(owl_object=nc) and not onto.check_deprecated(owl_object=nc):
            nebs_classes.add(nc)
    return nebs_classes


def get_test_neg_candidates(subclass, gt):
    all_nebs, seeds = set(), [gt]
    depth = 1
    while depth <= FLAGS.max_depth:
        new_seeds = set()
        for seed in seeds:
            nebs = get_one_hop_neighbours(c=seed)
            new_seeds = new_seeds.union(nebs)
            all_nebs = all_nebs.union(nebs)
        depth += 1
        seeds = random.sample(new_seeds, FLAGS.max_width) if len(new_seeds) > FLAGS.max_width else new_seeds
    all_nebs = all_nebs - subclass.ancestors() - {subclass}
    if len(all_nebs) > FLAGS.test_max_neg_size:
        return random.sample(all_nebs, FLAGS.test_max_neg_size)
    else:
        return list(all_nebs)


def context_candidate(output_file, target_subs):
    with open(output_file, 'w') as ff:
        size_sum = 0
        size_num = dict()
        m = 0
        for subcls, supcls in target_subs:
            neg_candidates = get_test_neg_candidates(subclass=subcls, gt=supcls)
            size = len(neg_candidates)
            size_sum += size
            size_num[size] = size_num[size] + 1 if size in size_num else 1
            if size > 0:
                s = ','.join([c.getIRI() for c in neg_candidates])
                ff.write('%s,%s,%s\n' % (subcls.getIRI(), supcls.getIRI(), s))
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
    for r in restrictions:
        if not onto.reasoner.check_subsumption(sub_entity=subcls, super_entity=r):
            neg_restrictions.append(r)
    if len(neg_restrictions) > FLAGS.test_max_neg_size:
        neg_restrictions = random.sample(neg_restrictions, FLAGS.test_max_neg_size)
    return neg_restrictions


if FLAGS.restriction:
    with open(FLAGS.train_file, 'w') as f:
        for c1, c2 in train_subsumptions:
            f.write('%s,%s\n' % (c1.getIRI(), str(c2)))
    with open(FLAGS.valid_file, 'w') as f:
        sizes = 0
        for c1, c2 in valid_subsumptions:
            c2_neg = get_test_neg_candidates_restriction(subcls=c1)
            sizes += len(c2_neg)
            strs = [str(r) for r in c2_neg]
            f.write('%s,%s,%s\n' % (c1.getIRI(), str(c2), ','.join(strs)))
        print('valid candidate negative avg. size: %.1f' % (sizes / len(valid_subsumptions)))
    with open(FLAGS.test_file, 'w') as f:
        sizes = 0
        for c1, c2 in test_subsumptions:
            c2_neg = get_test_neg_candidates_restriction(subcls=c1)
            sizes += len(c2_neg)
            strs = [str(r) for r in c2_neg]
            f.write('%s,%s,%s\n' % (c1.getIRI(), str(c2), ','.join(strs)))
        print('test candidate negative avg. size: %.1f' % (sizes / len(test_subsumptions)))

else:
    with open(FLAGS.train_file, 'w') as f:
        for c1, c2 in train_subsumptions:
            f.write('%s,%s\n' % (c1.getIRI(), c2.getIRI()))

    print('\n---- context candidates for validation subsumptions ----')
    context_candidate(output_file=FLAGS.valid_file, target_subs=valid_subsumptions)

    print('\n---- context candidates for test subsumptions ----')
    context_candidate(output_file=FLAGS.test_file, target_subs=test_subsumptions)

# TODO: delete valid and test subsumption axioms, and save a new ontology

config = CfgNode(DEFAULT_CONFIG_FILE)
config.train_subsumption_file = FLAGS.train_file
config.valid_subsumption_file = FLAGS.valid_file
config.test_subsumption_file = FLAGS.test_file
config.onto_file = FLAGS.evaluate_onto_file
onto2 = Ontology(owl_path=FLAGS.evaluate_onto_file)
pipeline = BERTSubsIntraPipeline(onto=onto2, config=config)
