from owlready2 import *

onto1_file = '/mnt/datashare/MONDO/preprocessed/omim.owl'
onto2_file = '/mnt/datashare/MONDO/preprocessed/ordo.owl'
mapping_file = '/mnt/datashare/MONDO/refs/omim2ordo/unsupervised/test.tsv'

prefixes = {'ordo': 'http://www.orpha.net/ORDO/',
            'omim': 'http://omim.org/entry/',
            'omimps': 'http://www.omim.org/phenotypicSeries/'}
prefixes_r = {'http://www.orpha.net/ORDO/': 'ordo',
              'http://omim.org/entry/': 'omim',
              'http://www.omim.org/phenotypicSeries/': 'omimps'}

subs, subs_r = list(), list()
subs_file = '/home/jiaoyan/MONDO/subs.tsv'
subs_r_file = '/home/jiaoyan/MONDO/subs_r.tsv'
onto2_new_file = '/home/jiaoyan/MONDO/ordo_new.owl'


def named_parent_classes(cls):
    classes = set()
    for c in cls.is_a:
        if type(c) == ThingClass and not c == owl.Thing:
            classes.add(c)
    return classes


def named_child_classes(cls):
    classes = set()
    for c in cls.subclasses():
        if type(c) == ThingClass:
            classes.add(c)
    return classes


def iri_format(cls):
    for k in prefixes_r:
        if cls.iri.startswith(k):
            return cls.iri.replace(k, prefixes_r[k]+':')


mappings = list()
with open(mapping_file) as f:
    for line in f.readlines()[1:]:
        item = line.strip().split('\t')
        mappings.append(item[0:2])

onto2 = get_ontology(onto2_file).load()
for c1, c2 in mappings:
    tmp2 = c2.split(':')
    c2_ = IRIS[prefixes[tmp2[0]] + tmp2[1]]
    c2_subclasses = named_child_classes(cls=c2_)
    c2_parents = named_parent_classes(cls=c2_)

    # case 1: c2 has named class parent (not OWL Thing) but no child
    if len(c2_subclasses) == 0 and len(c2_parents) > 0:
        for c2_parent in c2_parents:
            subs.append([c1, iri_format(cls=c2_parent)])
        destroy_entity(c2_)

    # case 2: c2 has named class child but no parent (its parent is OWL Thing)
    if len(c2_subclasses) > 0 and len(c2_parents) == 0:
        for c2_subclass in c2_subclasses:
            subs_r.append([c1, iri_format(cls=c2_subclass)])
        destroy_entity(c2_)

    # case 3: c2 has both named class child and parent
    if len(c2_subclasses) > 0 and len(c2_parents) > 0:
        for c2_parent in c2_parents:
            subs.append([c1, iri_format(cls=c2_parent)])
        for c2_subclass in c2_subclasses:
            subs_r.append([c1, iri_format(cls=c2_subclass)])
            for c2_parent in c2_parents:
                c2_subclass.is_a.append(c2_parent)
        destroy_entity(c2_)

with open(subs_file, 'w') as f:
    f.write('SrcEntity\tTgtEntity\n')
    for src, tgt in subs:
        f.write('%s\t%s\n' % (src, tgt))

with open(subs_r_file, 'w') as f:
    f.write('SrcEntity\tTgtEntity\n')
    for src, tgt in subs_r:
        f.write('%s\t%s\n' % (src, tgt))

print('subs: %d, subs_r: %d' % (len(subs), len(subs_r)))

onto2.save(file=onto2_new_file, format="rdfxml")
print('new target ontology saved')
