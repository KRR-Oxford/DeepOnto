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

import random
import sys
import re
import warnings
from typing import List, Union

from deeponto.onto import Ontology
from deeponto.onto import OntologyVerbaliser
from yacs.config import CfgNode


class SubsumptionSampler:
    r"""Class for sampling functions for training the subsumption prediction model.

    Attributes:
        onto (Ontology): The target ontology.
        config (CfgNode): The loaded configuration.
        named_classes (Set[str]): IRIs of named classes that are not deprecated.
        iri_label (Dict[str, List]): key -- class iris from `named_classes`, value -- a list of labels.
        restrictionObjects (Set[OWLClassExpression]): Basic existential restrictions that appear in the ontology.
        restrictions (set[str]): Strings of basic existential restrictions corresponding to `restrictionObjects`.
        restriction_label (Dict[str:List]): key -- existential restriction string, value -- a list of existential restriction labels.
        verb (OntologyVerbaliser): object for verbalisation.
    """

    def __init__(self, onto: Ontology, config: CfgNode):
        self.onto = onto
        self.config = config
        self.named_classes = self.extract_named_classes(onto=onto)
        self.iri_label = dict()
        for iri in self.named_classes:
            self.iri_label[iri] = []
            for p in config.label_property:
                strings = onto.get_owl_object_annotations(
                    owl_object=onto.get_owl_object_from_iri(iri),
                    annotation_property_iri=p,
                    annotation_language_tag=None,
                    apply_lowercasing=False,
                    normalise_identifiers=False,
                )
                for s in strings:
                    if s not in self.iri_label[iri]:
                        self.iri_label[iri].append(s)

        self.restrictionObjects = set()
        self.restrictions = set()
        self.restriction_label = dict()
        self.verb = OntologyVerbaliser(onto=onto)
        for complexC in onto.get_asserted_complex_classes():
            s = str(complexC)
            self.restriction_label[s] = []
            if self.is_basic_existential_restriction(complex_class_str=s):
                self.restrictionObjects.add(complexC)
                self.restrictions.add(s)
                self.restriction_label[s].append(self.verb.verbalise_class_expression(complexC).verbal)

    @staticmethod
    def is_basic_existential_restriction(complex_class_str: str):
        """Determine if a complex class expression is a basic existential restriction."""
        IRI = "<https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)>"
        p = rf"ObjectSomeValuesFrom\({IRI}\s{IRI}\)"
        if re.match(p, complex_class_str):
            return True
        else:
            return False

    @staticmethod
    def extract_named_classes(onto: Ontology):
        named_classes = set()
        for iri in onto.owl_classes:
            if not onto.check_deprecated(owl_object=onto.owl_classes[iri]):
                named_classes.add(iri)
        return named_classes

    def generate_samples(self, subsumptions: List[List], duplicate: bool = True):
        r"""Generate text samples from subsumptions.

        Args:
            subsumptions (List[List]): A list of subsumptions, each of which of is a two-component list `(sub_class_iri, super_class_iri_or_str)`.
            duplicate (bool): `True` -- duplicate the positive and negative samples, `False` -- do not duplicate.

        Returns:
            (List[List]): A list of samples, each element is a triple
                in the form of `(sub_class_string, super_class_string, label_index)`.
        """
        if duplicate:
            pos_dup, neg_dup = self.config.fine_tune.train_pos_dup, self.config.fine_tune.train_neg_dup
        else:
            pos_dup, neg_dup = 1, 1
        neg_subsumptions = list()
        for subs in subsumptions:
            c1 = subs[0]
            for _ in range(neg_dup):
                neg_c = self.get_negative_sample(subclass_iri=c1, subsumption_type=self.config.subsumption_type)
                if neg_c is not None:
                    neg_subsumptions.append([c1, neg_c])
        pos_samples = self.subsumptions_to_samples(subsumptions=subsumptions, sample_label=1)
        pos_samples = pos_dup * pos_samples
        neg_samples = self.subsumptions_to_samples(subsumptions=neg_subsumptions, sample_label=0)
        if len(neg_samples) < len(pos_samples):
            neg_samples = neg_samples + [
                random.choice(neg_samples) for _ in range(len(pos_samples) - len(neg_samples))
            ]
        if len(neg_samples) > len(pos_samples):
            pos_samples = pos_samples + [
                random.choice(pos_samples) for _ in range(len(neg_samples) - len(pos_samples))
            ]
        print("pos_samples: %d, neg_samples: %d" % (len(pos_samples), len(neg_samples)))
        all_samples = [s for s in pos_samples + neg_samples if s[0] != "" and s[1] != ""]
        random.shuffle(all_samples)
        return all_samples

    def subsumptions_to_samples(self, subsumptions: List[List], sample_label: Union[int, None]):
        r"""Transform subsumptions into samples of strings.

        Args:
            subsumptions (List[List]): The given subsumptions.
            sample_label (Union[int, None]): `1` (positive), `0` (negative), `None` (no label).

        Returns:
            (List[List]): A list of samples, each element is a triple
                in the form of `(sub_class_string, super_class_string, label_index)`.

        """
        local_samples = list()
        for subs in subsumptions:
            subcls, supcls = subs[0], subs[1]
            substrs = self.iri_label[subcls] if subcls in self.iri_label and len(self.iri_label[subcls]) > 0 else [""]

            if self.config.subsumption_type == "named_class":
                supstrs = self.iri_label[supcls] if supcls in self.iri_label and len(self.iri_label[supcls]) else [""]
            else:
                if supcls in self.restriction_label and len(self.restriction_label[supcls]) > 0:
                    supstrs = self.restriction_label[supcls]
                else:
                    supstrs = [self.verb.verbalise_class_expression(supcls).verbal]

            if self.config.use_one_label:
                substrs, supstrs = substrs[0:1], supstrs[0:1]

            if self.config.prompt.prompt_type == "isolated":
                for substr in substrs:
                    for supstr in supstrs:
                        local_samples.append([substr, supstr])

            elif self.config.prompt.prompt_type == "traversal":
                subs_list_strs = set()
                for _ in range(self.config.prompt.context_dup):
                    context_sub, no_duplicate = self.traversal_subsumptions(
                        cls=subcls,
                        hop=self.config.prompt.prompt_hop,
                        direction="subclass",
                        max_subsumptions=self.config.prompt.prompt_max_subsumptions,
                    )
                    subs_list = [self.named_subsumption_to_str(subsum) for subsum in context_sub]
                    subs_list_str = " <SEP> ".join(subs_list)
                    subs_list_strs.add(subs_list_str)
                    if no_duplicate:
                        break

                if self.config.subsumption_type == "named_class":
                    sups_list_strs = set()
                    for _ in range(self.config.prompt.context_dup):
                        context_sup, no_duplicate = self.traversal_subsumptions(
                            cls=supcls,
                            hop=self.config.prompt.prompt_hop,
                            direction="supclass",
                            max_subsumptions=self.config.prompt.prompt_max_subsumptions,
                        )
                        sups_list = [self.named_subsumption_to_str(subsum) for subsum in context_sup]
                        sups_list_str = " <SEP> ".join(sups_list)
                        sups_list_strs.add(sups_list_str)
                        if no_duplicate:
                            break
                else:
                    sups_list_strs = set(supstrs)

                for subs_list_str in subs_list_strs:
                    for substr in substrs:
                        s1 = substr + " <SEP> " + subs_list_str
                        for sups_list_str in sups_list_strs:
                            for supstr in supstrs:
                                s2 = supstr + " <SEP> " + sups_list_str
                                local_samples.append([s1, s2])

            elif self.config.prompt.prompt_type == "path":
                sep_token = "<SUB>" if self.config.prompt.use_sub_special_token else "<SEP>"

                s1_set = set()
                for _ in range(self.config.prompt.context_dup):
                    context_sub, no_duplicate = self.path_subsumptions(
                        cls=subcls, hop=self.config.prompt.prompt_hop, direction="subclass"
                    )
                    if len(context_sub) > 0:
                        s1 = ""
                        for i in range(len(context_sub)):
                            subsum = context_sub[len(context_sub) - i - 1]
                            subc = subsum[0]
                            s1 += "%s %s " % (
                                self.iri_label[subc][0]
                                if subc in self.iri_label and len(self.iri_label[subc]) > 0
                                else "",
                                sep_token,
                            )
                        for substr in substrs:
                            s1_set.add(s1 + substr)
                    else:
                        for substr in substrs:
                            s1_set.add("%s %s" % (sep_token, substr))

                    if no_duplicate:
                        break

                if self.config.subsumption_type == "named_class":
                    s2_set = set()
                    for _ in range(self.config.prompt.context_dup):
                        context_sup, no_duplicate = self.path_subsumptions(
                            cls=supcls, hop=self.config.prompt.prompt_hop, direction="supclass"
                        )
                        if len(context_sup) > 0:
                            s2 = ""
                            for subsum in context_sup:
                                supc = subsum[1]
                                s2 += " %s %s" % (
                                    sep_token,
                                    self.iri_label[supc][0]
                                    if supc in self.iri_label and len(self.iri_label[supc]) > 0
                                    else "",
                                )
                            for supstr in supstrs:
                                s2_set.add(supstr + s2)
                        else:
                            for supstr in supstrs:
                                s2_set.add("%s %s" % (supstr, sep_token))

                        if no_duplicate:
                            break
                else:
                    s2_set = set(supstrs)

                for s1 in s1_set:
                    for s2 in s2_set:
                        local_samples.append([s1, s2])

            else:
                print(f"unknown context type {self.config.prompt.prompt_type}")
                sys.exit(0)

        if sample_label is not None:
            for i in range(len(local_samples)):
                local_samples[i].append(sample_label)

        return local_samples

    def get_negative_sample(self, subclass_iri: str, subsumption_type: str = "named_class"):
        r"""Given a named subclass, get a negative class for a negative subsumption.

        Args:
            subclass_iri (str): IRI of a given sub-class.
            subsumption_type (str): `named_class` or `restriction`.
        """
        subclass = self.onto.get_owl_object_from_iri(iri=subclass_iri)
        if subsumption_type == "named_class":
            ancestors = set(self.onto.reasoner.get_inferred_super_entities(subclass, direct=False))
            neg_c = random.sample(self.named_classes - ancestors, 1)[0]
            return neg_c
        else:
            for neg_c in random.sample(self.restrictionObjects, 5):
                if not self.onto.reasoner.check_subsumption(sub_entity=subclass, super_entity=neg_c):
                    return str(neg_c)
            return None

    def named_subsumption_to_str(self, subsum: List):
        r"""Transform a named subsumption into string with `<SUB>` and classes' labels.

        Args:
            subsum (List[Tuple]): A list of subsumption pairs in the form of `(sub_class_iri, super_class_iri)`.
        """
        subc, supc = subsum[0], subsum[1]
        subs = self.iri_label[subc][0] if subc in self.iri_label and len(self.iri_label[subc]) > 0 else ""
        sups = self.iri_label[supc][0] if supc in self.iri_label and len(self.iri_label[supc]) > 0 else ""
        return "%s <SUB> %s" % (subs, sups)

    def subclass_to_strings(self, subcls):
        r"""Transform a sub-class into strings (with the path or traversal context template).

        Args:
            subcls (str): IRI of the sub-class.
        """
        substrs = self.iri_label[subcls] if subcls in self.iri_label and len(self.iri_label[subcls]) > 0 else [""]

        if self.config.use_one_label:
            substrs = substrs[0:1]

        if self.config.prompt.prompt_type == "isolated":
            return substrs

        elif self.config.prompt.prompt_type == "traversal":
            subs_list_strs = set()
            for _ in range(self.config.prompt.context_dup):
                context_sub, no_duplicate = self.traversal_subsumptions(
                    cls=subcls,
                    hop=self.config.prompt.prompt_hop,
                    direction="subclass",
                    max_subsumptions=self.config.prompt.prompt_max_subsumptions,
                )
                subs_list = [self.named_subsumption_to_str(subsum) for subsum in context_sub]
                subs_list_str = " <SEP> ".join(subs_list)
                subs_list_strs.add(subs_list_str)
                if no_duplicate:
                    break

            strs = list()
            for subs_list_str in subs_list_strs:
                for substr in substrs:
                    s1 = substr + " <SEP> " + subs_list_str
                    strs.append(s1)
            return strs

        elif self.config.prompt.prompt_type == "path":
            sep_token = "<SUB>" if self.config.prompt.use_sub_special_token else "<SEP>"

            s1_set = set()
            for _ in range(self.config.prompt.context_dup):
                context_sub, no_duplicate = self.path_subsumptions(
                    cls=subcls, hop=self.config.prompt.prompt_hop, direction="subclass"
                )
                if len(context_sub) > 0:
                    s1 = ""
                    for i in range(len(context_sub)):
                        subsum = context_sub[len(context_sub) - i - 1]
                        subc = subsum[0]
                        s1 += "%s %s " % (
                            self.iri_label[subc][0]
                            if subc in self.iri_label and len(self.iri_label[subc]) > 0
                            else "",
                            sep_token,
                        )
                    for substr in substrs:
                        s1_set.add(s1 + substr)
                else:
                    for substr in substrs:
                        s1_set.add("%s %s" % (sep_token, substr))
                if no_duplicate:
                    break

            return list(s1_set)

    def supclass_to_strings(self, supcls: str, subsumption_type: str = "named_class"):
        r"""Transform a super-class into strings (with the path or traversal context template if the subsumption type is `"named_class"`).

        Args:
            supcls (str): IRI of the super-class.
            subsumption_type (str): The type of the subsumption.
        """

        if subsumption_type == "named_class":
            supstrs = self.iri_label[supcls] if supcls in self.iri_label and len(self.iri_label[supcls]) else [""]
        else:
            if supcls in self.restriction_label and len(self.restriction_label[supcls]) > 0:
                supstrs = self.restriction_label[supcls]
            else:
                warnings.warn("Warning: %s has no descriptions" % supcls)
                supstrs = [""]

        if self.config.use_one_label:
            if subsumption_type == "named_class":
                supstrs = supstrs[0:1]

        if self.config.prompt.prompt_type == "isolated":
            return supstrs

        elif self.config.prompt.prompt_type == "traversal":
            if subsumption_type == "named_class":
                sups_list_strs = set()
                for _ in range(self.config.prompt.context_dup):
                    context_sup, no_duplicate = self.traversal_subsumptions(
                        cls=supcls,
                        hop=self.config.prompt.prompt_hop,
                        direction="supclass",
                        max_subsumptions=self.config.prompt.prompt_max_subsumptions,
                    )
                    sups_list = [self.named_subsumption_to_str(subsum) for subsum in context_sup]
                    sups_list_str = " <SEP> ".join(sups_list)
                    sups_list_strs.add(sups_list_str)
                    if no_duplicate:
                        break

            else:
                sups_list_strs = set(supstrs)

            strs = list()
            for sups_list_str in sups_list_strs:
                for supstr in supstrs:
                    s2 = supstr + " <SEP> " + sups_list_str
                    strs.append(s2)
            return strs

        elif self.config.prompt.prompt_type == "path":
            sep_token = "<SUB>" if self.config.prompt.use_sub_special_token else "<SEP>"

            if subsumption_type == "named_class":
                s2_set = set()
                for _ in range(self.config.prompt.context_dup):
                    context_sup, no_duplicate = self.path_subsumptions(
                        cls=supcls, hop=self.config.prompt.prompt_hop, direction="supclass"
                    )
                    if len(context_sup) > 0:
                        s2 = ""
                        for subsum in context_sup:
                            supc = subsum[1]
                            s2 += " %s %s" % (
                                sep_token,
                                self.iri_label[supc][0]
                                if supc in self.iri_label and len(self.iri_label[supc]) > 0
                                else "",
                            )
                        for supstr in supstrs:
                            s2_set.add(supstr + s2)
                    else:
                        for supstr in supstrs:
                            s2_set.add("%s %s" % (supstr, sep_token))

                    if no_duplicate:
                        break
            else:
                s2_set = set(supstrs)

            return list(s2_set)

        else:
            print("unknown context type %s" % self.config.prompt.prompt_type)
            sys.exit(0)

    def traversal_subsumptions(self, cls: str, hop: int = 1, direction: str = "subclass", max_subsumptions: int = 5):
        r"""Given a class, get its subsumptions by traversing the class hierarchy.

            If the class is a sub-class in the subsumption axiom, get subsumptions from downside.
            If the class is a super-class in the subsumption axiom, get subsumptions from upside.

        Args:
            cls (str): IRI of a named class.
            hop (int): The depth of the path.
            direction (str): `subclass` (downside path) or `supclass` (upside path).
            max_subsumptions (int): The maximum number of subsumptions to consider.
        """
        subsumptions = list()
        seeds = [cls]
        d = 1
        no_duplicate = True
        while d <= hop:
            new_seeds = list()
            for s in seeds:
                if direction == "subclass":
                    tmp = self.onto.reasoner.get_inferred_sub_entities(
                        self.onto.get_owl_object_from_iri(iri=s), direct=True
                    )
                    if len(tmp) > 1:
                        no_duplicate = False
                    random.shuffle(tmp)
                    for c in tmp:
                        if not self.onto.check_deprecated(owl_object=self.onto.get_owl_object_from_iri(iri=c)):
                            subsumptions.append([c, s])
                            if c not in new_seeds:
                                new_seeds.append(c)
                elif direction == "supclass":
                    tmp = self.onto.reasoner.get_inferred_super_entities(
                        self.onto.get_owl_object_from_iri(iri=s), direct=True
                    )
                    if len(tmp) > 1:
                        no_duplicate = False
                    random.shuffle(tmp)
                    for c in tmp:
                        if not self.onto.check_deprecated(owl_object=self.onto.get_owl_object_from_iri(iri=c)):
                            subsumptions.append([s, c])
                            if c not in new_seeds:
                                new_seeds.append(c)
                else:
                    warnings.warn("Unknown direction: %s" % direction)
            if len(subsumptions) >= max_subsumptions:
                subsumptions = random.sample(subsumptions, max_subsumptions)
                break
            else:
                seeds = new_seeds
                random.shuffle(seeds)
                d += 1
        return subsumptions, no_duplicate

    def path_subsumptions(self, cls: str, hop: int = 1, direction: str = "subclass"):
        r"""Given a class, get its path subsumptions.

            If the class is a sub-class in the subsumption axiom, get subsumptions from downside.
            If the class is a super-class in the subsumption axiom, get subsumptions from upside.

        Args:
            cls (str): IRI of a named class.
            hop (int): The depth of the path.
            direction (str): `subclass` (downside path) or `supclass` (upside path).
        """
        subsumptions = list()
        seed = cls
        d = 1
        no_duplicate = True
        while d <= hop:
            if direction == "subclass":
                tmp = self.onto.reasoner.get_inferred_sub_entities(
                    self.onto.get_owl_object_from_iri(iri=seed), direct=True
                )
                if len(tmp) > 1:
                    no_duplicate = False
                end = True
                if len(tmp) > 0:
                    random.shuffle(tmp)
                    for c in tmp:
                        if not self.onto.check_deprecated(owl_object=self.onto.get_owl_object_from_iri(iri=c)):
                            subsumptions.append([c, seed])
                            seed = c
                            end = False
                            break
                if end:
                    break
            elif direction == "supclass":
                tmp = self.onto.reasoner.get_inferred_super_entities(
                    self.onto.get_owl_object_from_iri(iri=seed), direct=True
                )
                if len(tmp) > 1:
                    no_duplicate = False
                end = True
                if len(tmp) > 0:
                    random.shuffle(tmp)
                    for c in tmp:
                        if not self.onto.check_deprecated(owl_object=self.onto.get_owl_object_from_iri(iri=c)):
                            subsumptions.append([seed, c])
                            seed = c
                            end = False
                            break
                if end:
                    break
            else:
                warnings.warn("Unknown direction: %s" % direction)

            d += 1
        return subsumptions, no_duplicate
