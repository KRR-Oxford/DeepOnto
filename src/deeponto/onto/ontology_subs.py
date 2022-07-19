"""
This file is copied from the bert_subsumption experiment repository (the original file ontology.py)
"""

import sys
from owlready2 import class_construct, get_ontology, default_world, IRIS, owl
from owlready2.entity import ThingClass
from owlready2.prop import ObjectPropertyClass
import random
import warnings
import re


class OntologySubs:
    def __init__(self, onto_file, label_property):
        self.onto_file = onto_file
        self.onto = get_ontology(onto_file).load()
        self.label_property = label_property
        self.graph = default_world.as_rdflib_graph()
        self.iri_label = dict()
        for p in self.label_property:
            q = 'SELECT ?s ?o WHERE {?s <%s> ?o.}' % p
            for res in list(self.graph.query(q)):
                iri = res[0].n3()[1:-1]
                lit = res[1]
                if hasattr(lit, 'language') and (lit.language is None or lit.language == 'en' or
                                                 lit.language == 'en-us' or lit.language == 'en-gb'):
                    if iri in self.iri_label:
                        if lit.value not in self.iri_label[iri]:
                            self.iri_label[iri].append(lit.value)
                    else:
                        self.iri_label[iri] = [lit.value]
        self.named_classes = self.get_named_classes()
        self.className_label = dict()
        for c_iri in self.iri_label:
            self.className_label[str(IRIS[c_iri])] = self.iri_label[c_iri]

        self.name_restrictions = dict()
        self.restrictions = set()
        self.restrictionsName_label = dict()
        for c in self.named_classes:
            for parent in c.is_a:
                if self.is_target_some_restriction(c=parent):
                    self.name_restrictions[str(parent)] = parent
                    self.restrictions.add(parent)
                    if str(parent) not in self.restrictionsName_label:
                        self.restrictionsName_label[str(parent)] = self.restriction_to_label(restriction=parent)

    '''
        Generate descriptions towards a restriction
        involving properties of FoodOn:
        {'composed primarily of', 'derives from', 'develops from', 'has consumer', 'has country of origin',
            'has defining ingredient', 'has food substance analog', 'has ingredient', 'has input', 'has member',
            'has output', 'has part', 'has participant', 'has quality', 'has substance added', 'in taxon',
            'input of', 'is about', 'member of', 'output of', 'part of', 'produced by', 'surrounded by'}
    '''

    def restriction_to_label(self, restriction):
        p_label = restriction.property.label[0]
        if not (p_label.endswith('of') or p_label.endswith('from') or p_label.endswith('by')
                or p_label.endswith('about')):
            p_label = p_label + ' of'

        descriptions = list()
        if ' | ' in str(restriction.value):
            desc = 'something %s ' % p_label
            names = list()
            for item in str(restriction.value).split(' | '):
                names.append(self.className_label[item][0])
            desc = desc + ' ' + ' or '.join(names)
            descriptions.append(desc)

        elif ' & ' in str(restriction.value):

            # this is a special case with Not in FoodOn
            if str(restriction.value) == 'obo.FOODON_00002275 & obo.FOODON_00003202 & Not(obo.FOODON_00001138)':
                desc = 'something %s %s and %s and not %s' % (p_label, self.className_label['obo.FOODON_00002275'][0],
                                                              self.className_label['obo.FOODON_00003202'][0],
                                                              self.className_label['obo.FOODON_00001138'][0])
                descriptions.append(desc)

            else:
                desc = 'something %s ' % p_label
                names = list()
                for item in str(restriction.value).split(' & '):
                    names.append(self.className_label[item][0])
                desc = desc + ' ' + ' and '.join(names)
                descriptions.append(desc)

        else:
            for lab in self.className_label[str(restriction.value)]:
                desc = 'something %s %s' % (p_label, lab)
                descriptions.append(desc)
        return descriptions

    @staticmethod
    def is_target_some_restriction(c):
        # if type(c) == class_construct.Restriction and c.type == 24 and type(c.property) == prop.ObjectPropertyClass \
        #        and (type(c.value) == ThingClass or type(c.value) == Or or type(c.value) == And) \
        #        and ('some' not in str(c.value) and 'only' not in str(c.value) and
        #             'min' not in str(c.value) and 'exactly' not in str(c.value) and
        #             'max' not in str(c.value)):
        if type(c) == class_construct.Restriction and c.type == 24 and type(c.property) == ObjectPropertyClass \
                and type(c.value) == ThingClass and ('some' not in str(c.value) and 'only' not in str(c.value) and
                                                     'min' not in str(c.value) and 'exactly' not in str(
                    c.value) and 'max' not in str(c.value)):
            return True
        else:
            return False

    '''
        Get named classes that are not deprecated (excluding OWL Thing)
    '''

    def get_named_classes(self):
        named_classes = list()
        for c in self.onto.classes():
            if True not in c.deprecated and not c == owl.Thing:
                named_classes.append(c)
        return named_classes

    '''
        Given a named subclass, get a negative class for a negative subsumption
        The current implementation does not consider masked subsumptions
    '''

    def get_negative_sample(self, subclass_str, subsumption_type='named class'):
        subclass = IRIS[subclass_str]
        if subsumption_type == 'named class':
            neg_c = random.sample(set(self.named_classes) - subclass.ancestors(), 1)[0]
            return neg_c.iri
        else:
            restrictions_ancestors = self.get_all_restriction_ancestors(cls=subclass)
            neg_c = random.sample(self.restrictions - restrictions_ancestors, 1)[0]
            return str(neg_c)

    @staticmethod
    def get_all_restriction_ancestors(cls):
        ancs = set()
        for c in cls.is_a:
            if type(c) == class_construct.Restriction:
                ancs.add(c)
        for ancestor in cls.ancestors():
            for c in ancestor.is_a:
                if type(c) == class_construct.Restriction:
                    ancs.add(c)
        return ancs

    '''
        Get declared subsumptions whose parents are named classes (excluding OWL Thing)
    '''

    def get_declared_named_class_subsumption(self):
        declared_subsumptions = list()
        for c in self.named_classes:
            for parent in c.is_a:
                if self.is_normal_named_class(parent):
                    declared_subsumptions.append([c, parent])
        return declared_subsumptions

    '''
        Get subsumptions whose parents are "some" restrictions that have no restrictions inside
    '''

    def get_declared_restriction_subsumption(self):
        declared_subsumptions = list()
        for c in self.named_classes:
            for parent in c.is_a:
                if self.is_target_some_restriction(c=parent):
                    declared_subsumptions.append([c, parent])
        return declared_subsumptions

    @staticmethod
    def is_normal_named_class(c):
        if type(c) == ThingClass and c is not owl.Thing and True not in c.deprecated:
            return True
        else:
            return False

    '''
        Get one-hop neighbours of a given class
    '''

    @staticmethod
    def get_one_hop_neighbours(c):
        nebs_classes = set()
        for sub in c.subclasses():
            if type(sub) == ThingClass and True not in sub.deprecated:
                nebs_classes.add(sub)
        for sup in c.is_a:
            if type(sup) == ThingClass and True not in sup.deprecated:
                nebs_classes.add(sup)
        return nebs_classes

    '''
        Given a class, get its traversal-based subsumptions
        If the class is a subclass of a target axiom, get subsumptions from downside
        If the class is a supclass of a target axiom, get subsumptions from upside
    '''

    @staticmethod
    def traversal_subsumptions(cls, hop=1, direction='subclass', max_subsumptions=5):
        subsumptions = list()
        seeds = [cls]
        d = 1
        no_duplicate = True
        while d <= hop:
            new_seeds = list()
            for s in seeds:
                if direction == 'subclass':
                    tmp = list(s.subclasses())
                    if len(tmp) > 1:
                        no_duplicate = False
                    random.shuffle(tmp)
                    for c in tmp:
                        if type(c) == ThingClass and c is not owl.Thing and True not in c.deprecated:
                            subsumptions.append([c, s])
                            if c not in new_seeds:
                                new_seeds.append(c)
                elif direction == 'supclass':
                    tmp = s.is_a
                    if len(tmp) > 1:
                        no_duplicate = False
                    random.shuffle(tmp)
                    for c in tmp:
                        if type(c) == ThingClass and c is not owl.Thing and True not in c.deprecated:
                            subsumptions.append([s, c])
                            if c not in new_seeds:
                                new_seeds.append(c)
                else:
                    warnings.warn('Unknown direction: %s' % direction)
            if len(subsumptions) >= max_subsumptions:
                subsumptions = random.sample(subsumptions, max_subsumptions)
                break
            else:
                seeds = new_seeds
                random.shuffle(seeds)
                d += 1
        return subsumptions, no_duplicate

    '''
        Given a class, get its path subsumptions
        If the class is a subclass of a target axiom, get subsumptions from downside
        If the class is a supclass of a target axiom, get subsumptions from upside
    '''

    @staticmethod
    def path_subsumptions(cls, hop=1, direction='subclass'):
        subsumptions = list()
        seed = cls
        d = 1
        no_duplicate = True
        while d <= hop:
            if direction == 'subclass':
                tmp = list(seed.subclasses())
                if len(tmp) > 1:
                    no_duplicate = False
                end = True
                if len(tmp) > 0:
                    random.shuffle(tmp)
                    for c in tmp:
                        if type(c) == ThingClass and c is not owl.Thing and True not in c.deprecated:
                            subsumptions.append([c, seed])
                            seed = c
                            end = False
                            break
                if end:
                    break
            elif direction == 'supclass':
                tmp = seed.is_a
                if len(tmp) > 1:
                    no_duplicate = False
                end = True
                if len(tmp) > 0:
                    random.shuffle(tmp)
                    for c in tmp:
                        if type(c) == ThingClass and c is not owl.Thing and True not in c.deprecated:
                            subsumptions.append([seed, c])
                            seed = c
                            end = False
                            break
                if end:
                    break
            else:
                warnings.warn('Unknown direction: %s' % direction)

            d += 1
        return subsumptions, no_duplicate

    '''
        Transform a subsumption to string with <SUB> and classes' labels
    '''

    def subsumption_to_str(self, subsum):
        subc, supc = subsum[0], subsum[1]
        subs = self.iri_label[subc.iri][0] if subc.iri in self.iri_label else ''
        sups = self.iri_label[supc.iri][0] if supc.iri in self.iri_label else ''
        return '%s <SUB> %s' % (subs, sups)

    def set_masked_subsumptions(self, subsumptions_to_mask):
        for subsumptions in subsumptions_to_mask:
            subc, supc = IRIS[subsumptions[0]], IRIS[subsumptions[1]]

            # in case the sup class is an restriction
            if supc is None and subsumptions[1] in self.name_restrictions:
                supc = self.name_restrictions[subsumptions[1]]

            if supc in subc.is_a:
                subc.is_a.remove(supc)

    def subclass_to_string(self, subcls, config):
        substrs = self.iri_label[subcls] if subcls in self.iri_label else ['']

        if config['task']['use_one_label']:
            substrs = substrs[0:1]

        if config['task']['prompt_type'] == 'isolated':
            return substrs

        elif config['task']['prompt_type'] == 'traversal':
            subs_list_strs = set()
            for _ in range(config['task']['context_dup']):
                context_sub, no_duplicate = self.traversal_subsumptions(cls=IRIS[subcls],
                                                                        hop=config['task']['prompt_hop'],
                                                                        direction='subclass',
                                                                        max_subsumptions=config['task'][
                                                                            'prompt_max_subsumptions'])
                subs_list = [self.subsumption_to_str(subsum) for subsum in context_sub]
                subs_list_str = ' <SEP> '.join(subs_list)
                subs_list_strs.add(subs_list_str)
                if no_duplicate:
                    break

            strs = list()
            for subs_list_str in subs_list_strs:
                for substr in substrs:
                    s1 = substr + ' <SEP> ' + subs_list_str
                    strs.append(s1)
            return strs

        elif config['task']['prompt_type'] == 'path':
            sep_token = '<SUB>' if config['task']['use_sub_special_token'] else '<SEP>'

            s1_set = set()
            for _ in range(config['task']['context_dup']):
                context_sub, no_duplicate = self.path_subsumptions(cls=IRIS[subcls],
                                                                   hop=config['task']['prompt_hop'],
                                                                   direction='subclass')
                if len(context_sub) > 0:
                    s1 = ''
                    for i in range(len(context_sub)):
                        subsum = context_sub[len(context_sub) - i - 1]
                        subc = subsum[0]
                        s1 += '%s %s ' % (self.iri_label[subc.iri][0] if subc.iri in self.iri_label else '', sep_token)
                    for substr in substrs:
                        s1_set.add(s1 + substr)
                else:
                    for substr in substrs:
                        s1_set.add('%s %s' % (sep_token, substr))
                if no_duplicate:
                    break

            return list(s1_set)

    def supclass_to_samples(self, supcls, config, subsumption_type='named class'):

        if subsumption_type == 'named class':
            supstrs = self.iri_label[supcls] if supcls in self.iri_label else ['']
        else:
            if supcls in self.restrictionsName_label:
                supstrs = self.restrictionsName_label[supcls]
            else:
                print('Warning: %s has no descriptions' % supcls)
                supstrs = ['']

        if config['task']['use_one_label']:
            if subsumption_type == 'named class':
                supstrs = supstrs[0:1]

        if config['task']['prompt_type'] == 'isolated':
            return supstrs

        elif config['task']['prompt_type'] == 'traversal':
            if subsumption_type == 'named class':
                sups_list_strs = set()
                for _ in range(config['task']['context_dup']):
                    context_sup, no_duplicate = self.traversal_subsumptions(cls=IRIS[supcls],
                                                                            hop=config['task']['prompt_hop'],
                                                                            direction='supclass',
                                                                            max_subsumptions=config['task'][
                                                                                'prompt_max_subsumptions'])
                    sups_list = [self.subsumption_to_str(subsum) for subsum in context_sup]
                    sups_list_str = ' <SEP> '.join(sups_list)
                    sups_list_strs.add(sups_list_str)
                    if no_duplicate:
                        break

            else:
                sups_list_strs = set(supstrs)

            strs = list()
            for sups_list_str in sups_list_strs:
                for supstr in supstrs:
                    s2 = supstr + ' <SEP> ' + sups_list_str
                    strs.append(s2)
            return strs


        elif config['task']['prompt_type'] == 'path':
            sep_token = '<SUB>' if config['task']['use_sub_special_token'] else '<SEP>'

            if subsumption_type == 'named class':
                s2_set = set()
                for _ in range(config['task']['context_dup']):
                    context_sup, no_duplicate = self.path_subsumptions(cls=IRIS[supcls],
                                                                       hop=config['task']['prompt_hop'],
                                                                       direction='supclass')
                    if len(context_sup) > 0:
                        s2 = ''
                        for subsum in context_sup:
                            supc = subsum[1]
                            s2 += ' %s %s' % (sep_token,
                                              self.iri_label[supc.iri][0] if supc.iri in self.iri_label else '')
                        for supstr in supstrs:
                            s2_set.add(supstr + s2)
                    else:
                        for supstr in supstrs:
                            s2_set.add('%s %s' % (supstr, sep_token))

                    if no_duplicate:
                        break
            else:
                s2_set = set(supstrs)

            return list(s2_set)

        else:
            print('unknown context type %s' % config['task']['prompt_type'])
            sys.exit(0)

    '''
        Extract string pairs for subsumption axioms
        prompt type: isolated, traversal, random walk
        sample label: None, or label number (e.g., 1 and 0 for binary classification)
    '''

    def subsumptions_to_samples(self, subsumptions, config, sample_label, subsumption_type='named class'):
        local_samples = list()
        for subs in subsumptions:
            subcls, supcls = subs[0], subs[1]
            substrs = self.iri_label[subcls] if subcls in self.iri_label else ['']

            if subsumption_type == 'named class':
                supstrs = self.iri_label[supcls] if supcls in self.iri_label else ['']
            else:
                if supcls in self.restrictionsName_label:
                    supstrs = self.restrictionsName_label[supcls]
                else:
                    print('Warning: %s has no descriptions' % supcls)
                    supstrs = ['']

            if config['task']['use_one_label']:
                substrs = substrs[0:1]
                supstrs = supstrs[0:1]

            if config['task']['prompt_type'] == 'isolated':
                for substr in substrs:
                    for supstr in supstrs:
                        local_samples.append([substr, supstr])

            elif config['task']['prompt_type'] == 'traversal':
                subs_list_strs = set()
                for _ in range(config['task']['context_dup']):
                    context_sub, no_duplicate = self.traversal_subsumptions(cls=IRIS[subcls],
                                                                            hop=config['task']['prompt_hop'],
                                                                            direction='subclass',
                                                                            max_subsumptions=config['task'][
                                                                                'prompt_max_subsumptions'])
                    subs_list = [self.subsumption_to_str(subsum) for subsum in context_sub]
                    subs_list_str = ' <SEP> '.join(subs_list)
                    subs_list_strs.add(subs_list_str)
                    if no_duplicate:
                        break

                if subsumption_type == 'named class':
                    sups_list_strs = set()
                    for _ in range(config['task']['context_dup']):
                        context_sup, no_duplicate = self.traversal_subsumptions(cls=IRIS[supcls],
                                                                                hop=config['task']['prompt_hop'],
                                                                                direction='supclass',
                                                                                max_subsumptions=config['task'][
                                                                                    'prompt_max_subsumptions'])
                        sups_list = [self.subsumption_to_str(subsum) for subsum in context_sup]
                        sups_list_str = ' <SEP> '.join(sups_list)
                        sups_list_strs.add(sups_list_str)
                        if no_duplicate:
                            break
                else:
                    sups_list_strs = set(supstrs)

                for subs_list_str in subs_list_strs:
                    for substr in substrs:
                        s1 = substr + ' <SEP> ' + subs_list_str
                        for sups_list_str in sups_list_strs:
                            for supstr in supstrs:
                                s2 = supstr + ' <SEP> ' + sups_list_str
                                local_samples.append([s1, s2])

            elif config['task']['prompt_type'] == 'path':
                sep_token = '<SUB>' if config['task']['use_sub_special_token'] else '<SEP>'

                s1_set = set()
                for _ in range(config['task']['context_dup']):
                    context_sub, no_duplicate = self.path_subsumptions(cls=IRIS[subcls],
                                                                       hop=config['task']['prompt_hop'],
                                                                       direction='subclass')
                    if len(context_sub) > 0:
                        s1 = ''
                        for i in range(len(context_sub)):
                            subsum = context_sub[len(context_sub) - i - 1]
                            subc = subsum[0]
                            s1 += '%s %s ' % (self.iri_label[subc.iri][0] if subc.iri in self.iri_label else '',
                                              sep_token)
                        for substr in substrs:
                            s1_set.add(s1 + substr)
                    else:
                        for substr in substrs:
                            s1_set.add('%s %s' % (sep_token, substr))

                    if no_duplicate:
                        break

                if subsumption_type == 'named class':
                    s2_set = set()
                    for _ in range(config['task']['context_dup']):
                        context_sup, no_duplicate = self.path_subsumptions(cls=IRIS[supcls],
                                                                           hop=config['task']['prompt_hop'],
                                                                           direction='supclass')
                        if len(context_sup) > 0:
                            s2 = ''
                            for subsum in context_sup:
                                supc = subsum[1]
                                s2 += ' %s %s' % (sep_token,
                                                  self.iri_label[supc.iri][0] if supc.iri in self.iri_label else '')
                            for supstr in supstrs:
                                s2_set.add(supstr + s2)
                        else:
                            for supstr in supstrs:
                                s2_set.add('%s %s' % (supstr, sep_token))

                        if no_duplicate:
                            break
                else:
                    s2_set = set(supstrs)

                for s1 in s1_set:
                    for s2 in s2_set:
                        local_samples.append([s1, s2])

            else:
                print('unknown context type %s' % config['task']['prompt_type'])
                sys.exit(0)

        if sample_label is not None:
            for i in range(len(local_samples)):
                local_samples[i].append(sample_label)

        return local_samples


    def complement_iri_name_to_label(self, prefix):
        for c in self.named_classes:
            if c.iri not in self.iri_label:
                iri_name = c.iri.replace(prefix, '')
                """parse the URI name (camel cases)"""
                uri_name = iri_name.replace('_', ' ').replace('-', ' ').replace('.', ' '). \
                    replace('/', ' ').replace('"', ' ').replace("'", ' ')
                words = []
                for item in uri_name.split():
                    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', item)
                    for m in matches:
                        word = m.group(0)
                        words.append(word.lower())
                s = ' '.join(words)
                self.iri_label[c.iri] = [s]


