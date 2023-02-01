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

from __future__ import annotations

import networkx as nx
import itertools
import random
import os
from typing import List, Set, Tuple, Optional, Union

from deeponto.onto import Ontology
from deeponto.align.mapping import ReferenceMapping
from deeponto.utils import FileUtils, DataUtils


# @paper(
#     "BERTMap: A BERT-based Ontology Alignment System (AAAI-2022)",
#     "https://ojs.aaai.org/index.php/AAAI/article/view/20510",
# )
class AnnotationThesaurus:
    """A thesaurus class for synonyms and non-synonyms extracted from an ontology.

    Some related definitions of arguements here:

    - A `synonym_group` is a set of annotation phrases that are synonymous to each other;
    - The `transitivity` of synonyms means if A and B are synonymous and B and C are synonymous,
    then A and C are synonymous. This is achieved by a connected graph-based algorithm.
    - A `synonym_pair` is a pair synonymous annotation phrase which can be extracted from
    the cartesian product of a `synonym_group` and itself. NOTE that **reflexivity** and **symmetry**
    are preserved meaning that *(i)* every phrase A is a synonym of itself and *(ii)* if (A, B) is
    a synonym pair then (B, A) is a synonym pair, too.

    Attributes:
        onto (Ontology): An ontology to construct the annotation thesaurus from.
        annotation_index (Dict[str, Set[str]]): An index of the class annotations with `(class_iri, annotations)` pairs.
        annotation_property_iris (List[str]): A list of annotation property IRIs used to extract the annotations.
        average_number_of_annotations_per_class (int): The average number of (extracted) annotations per ontology class.
        apply_transitivity (bool): Apply synonym transitivity to merge synonym groups or not.
        synonym_groups (List[Set[str]]): The list of synonym groups extracted from the ontology according to specified annotation properties.
    """

    def __init__(self, onto: Ontology, annotation_property_iris: List[str], apply_transitivity: bool = False):
        r"""Initialise a thesaurus for ontology class annotations.

        Args:
            onto (Ontology): The input ontology to extract annotations from.
            annotation_property_iris (List[str]): Specify which annotation properties to be used.
            apply_transitivity (bool, optional): Apply synonym transitivity to merge synonym groups or not. Defaults to `False`.
        """

        self.onto = onto
        # build the annotation index to extract synonyms from `onto`
        # the input property iris may not exist in this ontology
        # the output property iris will be truncated to the existing ones
        index, iris = self.onto.build_annotation_index(
            annotation_property_iris=annotation_property_iris,
            entity_type="Classes",
            apply_lowercasing=True,
        )
        self.annotation_index = index
        self.annotation_property_iris = iris
        total_number_of_annotations = sum([len(v) for v in self.annotation_index.values()])
        self.average_number_of_annotations_per_class = total_number_of_annotations / len(self.annotation_index)

        # synonym groups
        self.apply_transitivity = apply_transitivity
        self.synonym_groups = list(self.annotation_index.values())
        if self.apply_transitivity:
            self.synonym_groups = self.merge_synonym_groups_by_transitivity(self.synonym_groups)

        # summary
        self.info = {
            type(self).__name__: {
                "ontology": self.onto.info[type(self.onto).__name__],
                "average_number_of_annotations_per_class": round(self.average_number_of_annotations_per_class, 3),
                "number_of_synonym_groups": len(self.synonym_groups),
            }
        }

    def __str__(self):
        str(self.onto)  # the info of ontology is updated upon calling its __str__ method
        return FileUtils.print_dict(self.info)

    @staticmethod
    def get_synonym_pairs(synonym_group: Set[str], remove_duplicates: bool = True):
        """Get synonym pairs from a synonym group through a cartesian product.

        Args:
            synonym_group (Set[str]): A set of annotation phrases that are synonymous to each other.

        Returns:
            (List[Tuple[str, str]]): A list of synonym pairs.
        """
        synonym_pairs = list(itertools.product(synonym_group, synonym_group))
        if remove_duplicates:
            return DataUtils.uniqify(synonym_pairs)
        else:
            return synonym_pairs

    @staticmethod
    def merge_synonym_groups_by_transitivity(synonym_groups: List[Set[str]]):
        r"""Merge synonym groups by transitivity.

        Synonym groups that share a common annotation phrase will be merged. NOTE that for
        multiple ontologies, we can merge their synonym groups by first concatenating them
        then use this function.

        !!! note

            In $\textsf{BERTMap}$ experiments we have considered this as a data augmentation approach
            but it does not bring a significant performance improvement. However, if the
            overall number of annotations is not large enough then this could be a good option.

        Args:
            synonym_groups (List[Set[str]]): A sequence of synonym groups to be merged.

        Returns:
            (List[Set[str]]): A list of merged synonym groups.
        """
        synonym_pairs = []
        for synonym_group in synonym_groups:
            # gather synonym pairs from the self-product of a synonym group
            synonym_pairs += AnnotationThesaurus.get_synonym_pairs(synonym_group, remove_duplicates=False)
        synonym_pairs = DataUtils.uniqify(synonym_pairs)
        merged_grouped_synonyms = AnnotationThesaurus.connected_labels(synonym_pairs)
        return merged_grouped_synonyms

    @staticmethod
    def connected_annotations(synonym_pairs: List[Tuple[str, str]]):
        """Build a graph for adjacency among the class annotations (labels) such that
        the **transitivity** of synonyms is ensured.

        Auxiliary function for [`merge_synonym_groups_by_transitivity`][deeponto.align.bertmap.text_semantics.AnnotationThesaurus.merge_synonym_groups_by_transitivity].

        Args:
            synonym_pairs (List[Tuple[str, str]]): List of pairs of phrases that are synonymous.

        Returns:
            (List[Set[str]]): A list of synonym groups.
        """
        graph = nx.Graph()
        graph.add_edges_from(synonym_pairs)
        # nx.draw(G, with_labels = True)
        connected = list(nx.connected_components(graph))
        return connected

    def synonym_sampling(self, num_samples: Optional[int] = None):
        r"""Sample synonym pairs from a list of synonym groups extracted from the input ontology.

        According to the $\textsf{BERTMap}$ paper, **synonyms** are defined as label pairs that belong
        to the same ontology class.

        NOTE this has been validated for getting the same results as in the original $\textsf{BERTMap}$ repository.

        Args:
            num_samples (int, optional): The (maximum) number of **unique** samples extracted. Defaults to `None`.

        Returns:
            (List[Tuple[str, str]]): A list of unique synonym pair samples.
        """
        synonym_pool = []
        for synonym_group in self.synonym_groups:
            # do not remove duplicates in the loop to save time
            synonym_pairs = self.get_synonym_pairs(synonym_group, remove_duplicates=False)
            synonym_pool += synonym_pairs
        # remove duplicates afer the loop
        synonym_pool = DataUtils.uniqify(synonym_pool)

        if (not num_samples) or (num_samples >= len(synonym_pool)):
            # print("Return all synonym pairs without downsampling.")
            return synonym_pool
        else:
            return random.sample(synonym_pool, num_samples)

    def soft_nonsynonym_sampling(self, num_samples: int, max_iter: int = 5):
        r"""Sample **soft** non-synonyms from a list of synonym groups extracted from the input ontology.

        According to the $\textsf{BERTMap}$ paper, **soft non-synonyms** are defined as label pairs
        from two *different* synonym groups that are **randomly** selected.

        Args:
            num_samples (int): The (maximum) number of **unique** samples extracted; this is
                required **unlike for synonym sampling** because the non-synonym pool is **significantly
                larger** (considering random combinations of different synonym groups).
            max_iter (int): The maximum number of iterations for conducting sampling. Defaults to `5`.

        Returns:
            (List[Tuple[str, str]]): A list of unique (soft) non-synonym pair samples.
        """
        nonsyonym_pool = []
        # randomly select disjoint synonym group pairs from all
        for _ in range(num_samples):
            left_synonym_group, right_synonym_group = tuple(random.sample(self.synonym_groups, 2))
            # randomly choose one label from a synonym group
            left_label = random.choice(list(left_synonym_group))
            right_label = random.choice(list(right_synonym_group))
            nonsyonym_pool.append((left_label, right_label))

        # DataUtils.uniqify is too slow so we should avoid operating it too often
        nonsyonym_pool = DataUtils.uniqify(nonsyonym_pool)

        while len(nonsyonym_pool) < num_samples and max_iter > 0:
            max_iter = max_iter - 1  # reduce the iteration to prevent exhausting loop
            nonsyonym_pool += self.soft_nonsynonym_sampling(num_samples - len(nonsyonym_pool), max_iter)
            nonsyonym_pool = DataUtils.uniqify(nonsyonym_pool)

        return nonsyonym_pool

    def weighted_random_choices_of_sibling_groups(self, k: int = 1):
        """Randomly (weighted) select a number of sibling class groups.

        The weights are computed according to the sizes of the sibling class groups.
        """
        weights = [len(s) for s in self.onto.sibling_class_groups]
        weights = [w / sum(weights) for w in weights]  # normalised
        return random.choices(self.onto.sibling_class_groups, weights=weights, k=k)

    def hard_nonsynonym_sampling(self, num_samples: int, max_iter: int = 5):
        r"""Sample **hard** non-synonyms from sibling classes of the input ontology.

        According to the $\textsf{BERTMap}$ paper, **hard non-synonyms** are defined as label pairs
        that belong to two **disjoint** ontology classes. For practical reason, the condition
        is eased to two **sibling** ontology classes.

        Args:
            num_samples (int): The (maximum) number of **unique** samples extracted; this is
                required **unlike for synonym sampling** because the non-synonym pool is **significantly
                larger** (considering random combinations of different synonym groups).
            max_iter (int): The maximum number of iterations for conducting sampling. Defaults to `5`.

        Returns:
            (List[Tuple[str, str]]): A list of unique (hard) non-synonym pair samples.
        """
        # intialise the sibling class groups
        self.onto.sibling_class_groups

        # flatten the disjointness groups into all pairs of hard neagtives
        nonsynonym_pool = []
        # randomly (weighted) select a number of sibling class groups with replacement
        sibling_class_groups = self.weighted_random_choices_of_sibling_groups(k=num_samples)

        for sibling_class_group in sibling_class_groups:
            # random select two sibling classes; no weights this time
            left_class_iri, right_class_iri = tuple(random.sample(sibling_class_group, 2))
            # random select a label for each of them
            left_label = random.choice(list(self.annotation_index[left_class_iri]))
            right_label = random.choice(list(self.annotation_index[right_class_iri]))
            # add the label pair to the pool
            nonsynonym_pool.append((left_label, right_label))

        # DataUtils.uniqify is too slow so we should avoid operating it too often
        nonsynonym_pool = DataUtils.uniqify(nonsynonym_pool)

        while len(nonsynonym_pool) < num_samples and max_iter > 0:
            max_iter = max_iter - 1  # reduce the iteration to prevent exhausting loop
            nonsynonym_pool += self.hard_nonsynonym_sampling(num_samples - len(nonsynonym_pool), max_iter)
            nonsynonym_pool = DataUtils.uniqify(nonsynonym_pool)

        return nonsynonym_pool


class IntraOntologyTextSemanticsCorpus:
    r"""Class for creating the intra-ontology text semantics corpus from an ontology.

    As defined in the $\textsf{BERTMap}$ paper, the **intra-ontology** text semantics corpus consists
    of synonym and non-synonym pairs extracted from the ontology class annotations.

    Attributes:
        onto (Ontology): An ontology to construct the intra-ontology text semantics corpus from.
        annotation_property_iris (List[str]): Specify which annotation properties to be used.
        soft_negative_ratio (int): The expected negative sample ratio of the soft non-synonyms to the extracted synonyms. Defaults to `2`.
        hard_negative_ratio (int): The expected negative sample ratio of the hard non-synonyms to the extracted synonyms. Defaults to `2`.
            However, hard non-synonyms are sometimes insufficient given an ontology's hierarchy, the soft ones are used to compensate
            the number in this case.
    """

    def __init__(
        self,
        onto: Ontology,
        annotation_property_iris: List[str],
        soft_negative_ratio: int = 2,
        hard_negative_ratio: int = 2,
    ):

        self.onto = onto
        # $\textsf{BERTMap}$ does not apply synonym transitivity
        self.thesaurus = AnnotationThesaurus(onto, annotation_property_iris, apply_transitivity=False)

        self.synonyms = self.thesaurus.synonym_sampling()
        # sample hard negatives first as they might not be enough
        num_hard = hard_negative_ratio * len(self.synonyms)
        self.hard_nonsynonyms = self.thesaurus.hard_nonsynonym_sampling(num_hard)
        # compensate the number of hard negatives as soft negatives are almost always available
        num_soft = (soft_negative_ratio + hard_negative_ratio) * len(self.synonyms) - len(self.hard_nonsynonyms)
        self.soft_nonsynonyms = self.thesaurus.soft_nonsynonym_sampling(num_soft)

        self.info = {
            type(self).__name__: {
                "num_synonyms": len(self.synonyms),
                "num_nonsynonyms": len(self.soft_nonsynonyms) + len(self.hard_nonsynonyms),
                "num_soft_nonsynonyms": len(self.soft_nonsynonyms),
                "num_hard_nonsynonyms": len(self.hard_nonsynonyms),
                "annotation_thesaurus": self.thesaurus.info["AnnotationThesaurus"],
            }
        }

    def __str__(self):
        return FileUtils.print_dict(self.info)

    def save(self, save_path: str):
        """Save the intra-ontology corpus (a `.json` file for label pairs
        and its summary) in the specified directory.
        """
        FileUtils.create_path(save_path)
        save_json = {
            "summary": self.info,
            "synonyms": [(pos[0], pos[1], 1) for pos in self.synonyms],
            "nonsynonyms": [(neg[0], neg[1], 0) for neg in self.soft_nonsynonyms + self.hard_nonsynonyms],
        }
        FileUtils.save_file(save_json, os.path.join(save_path, "intra-onto.corpus.json"))


class CrossOntologyTextSemanticsCorpus:
    r"""Class for creating the cross-ontology text semantics corpus from two ontologies and provided mappings between them.

    As defined in the $\textsf{BERTMap}$ paper, the **cross-ontology** text semantics corpus consists
    of synonym and non-synonym pairs extracted from the annotations/labels of class pairs
    involved in the provided cross-ontology mappigns.

    Attributes:
        class_mappings (List[ReferenceMapping]): A list of cross-ontology class mappings.
        src_onto (Ontology): The source ontology whose class IRIs are heads of the `class_mappings`.
        tgt_onto (Ontology): The target ontology whose class IRIs are tails of the `class_mappings`.
        annotation_property_iris (List[str]): A list of annotation property IRIs used to extract the annotations.
        negative_ratio (int): The expected negative sample ratio of the non-synonyms to the extracted synonyms. Defaults to `4`. NOTE
            that we do not have *hard* non-synonyms at the cross-ontology level.
    """

    def __init__(
        self,
        class_mappings: List[ReferenceMapping],
        src_onto: Ontology,
        tgt_onto: Ontology,
        annotation_property_iris: List[str],
        negative_ratio: int = 4,
    ):
        self.class_mappings = class_mappings
        self.src_onto = src_onto
        self.tgt_onto = tgt_onto
        # build the annotation thesaurus for each ontology
        self.src_thesaurus = AnnotationThesaurus(src_onto, annotation_property_iris)
        self.tgt_thesaurus = AnnotationThesaurus(tgt_onto, annotation_property_iris)
        self.negative_ratio = negative_ratio

        self.synonyms = self.synonym_sampling_from_mappings()
        num_negative = negative_ratio * len(self.synonyms)
        self.nonsynonyms = self.nonsynonym_sampling_from_mappings(num_negative)

        self.info = {
            type(self).__name__: {
                "num_synonyms": len(self.synonyms),
                "num_nonsynonyms": len(self.nonsynonyms),
                "num_mappings": len(self.class_mappings),
                "src_annotation_thesaurus": self.src_thesaurus.info["AnnotationThesaurus"],
                "tgt_annotation_thesaurus": self.tgt_thesaurus.info["AnnotationThesaurus"],
            }
        }

    def __str__(self):
        return FileUtils.print_dict(self.info)

    def save(self, save_path: str):
        """Save the cross-ontology corpus (a `.json` file for label pairs
        and its summary) in the specified directory.
        """
        FileUtils.create_path(save_path)
        save_json = {
            "summary": self.info,
            "synonyms": [(pos[0], pos[1], 1) for pos in self.synonyms],
            "nonsynonyms": [(neg[0], neg[1], 0) for neg in self.nonsynonyms],
        }
        FileUtils.save_file(save_json, os.path.join(save_path, "cross-onto.corpus.json"))

    def synonym_sampling_from_mappings(self):
        r"""Sample synonyms from cross-ontology class mappings.

        Arguements of this method are all class attributes.
        See [`CrossOntologyTextSemanticsCorpus`][deeponto.align.bertmap.text_semantics.CrossOntologyTextSemanticsCorpus].

        According to the $\textsf{BERTMap}$ paper, **cross-ontology synonyms** are defined as label pairs
        that belong to two **matched** classes. Suppose the class $C$ from the source ontology
        and the class $D$ from the target ontology are matched according to one of the `class_mappings`,
        then the cartesian product of labels of $C$ and labels of $D$ form cross-ontology synonyms.
        Note that **identity synonyms** in the form of $(a, a)$ are removed because they have been covered
        in the intra-ontology case.

        Returns:
            (List[Tuple[str, str]]): A list of unique synonym pair samples from ontology class mappings.
        """
        synonym_pool = []

        for class_mapping in self.class_mappings:
            src_class_iri, tgt_class_iri = class_mapping.to_tuple()
            src_class_annotations = self.src_thesaurus.annotation_index[src_class_iri]
            tgt_class_annotations = self.tgt_thesaurus.annotation_index[tgt_class_iri]
            synonym_pairs = list(itertools.product(src_class_annotations, tgt_class_annotations))
            # remove the identity synonyms as the have been covered in the intra-ontology case
            synonym_pairs = [(l, r) for l, r in synonym_pairs if l != r]
            backward_synonym_pairs = [(r, l) for l, r in synonym_pairs]
            synonym_pool += synonym_pairs + backward_synonym_pairs

        synonym_pool = DataUtils.uniqify(synonym_pool)
        return synonym_pool

    def nonsynonym_sampling_from_mappings(self, num_samples: int, max_iter: int = 5):
        r"""Sample non-synonyms from cross-ontology class mappings.

        Arguements of this method are all class attributes.
        See [`CrossOntologyTextSemanticsCorpus`][deeponto.align.bertmap.text_semantics.CrossOntologyTextSemanticsCorpus].

        According to the $\textsf{BERTMap}$ paper, **cross-ontology non-synonyms** are defined as label pairs
        that belong to two **unmatched** classes. Assume that the provided class mappings are self-contained
        in the sense that they are complete for the classes involved in them, then we can randomly
        sample two cross-ontology classes that are not matched according to the mappings and take
        their labels as nonsynonyms. In practice, it is quite unlikely to obtain false negatives since
        the number of incorrect mappings is much larger than the number of correct ones.

        Returns:
            (List[Tuple[str, str]]): A list of unique nonsynonym pair samples from ontology class mappings.
        """
        nonsynonym_pool = []

        # form cross-ontology synonym groups
        cross_onto_synonym_group_pair = []
        for class_mapping in self.class_mappings:
            src_class_iri, tgt_class_iri = class_mapping.to_tuple()
            src_class_annotations = self.src_thesaurus.annotation_index[src_class_iri]
            tgt_class_annotations = self.tgt_thesaurus.annotation_index[tgt_class_iri]
            # let each matched class pair's annotations form a synonym group_pair
            cross_onto_synonym_group_pair.append((src_class_annotations, tgt_class_annotations))

        # randomly select disjoint synonym group pairs from all
        for _ in range(num_samples):
            left_class_pair, right_class_pair = tuple(random.sample(cross_onto_synonym_group_pair, 2))
            # randomly choose one label from a synonym group
            left_label = random.choice(list(left_class_pair[0]))  # choosing the src side by [0]
            right_label = random.choice(list(right_class_pair[1]))  # choosing the tgt side by [1]
            nonsynonym_pool.append((left_label, right_label))

        # DataUtils.uniqify is too slow so we should avoid operating it too often
        nonsynonym_pool = DataUtils.uniqify(nonsynonym_pool)
        while len(nonsynonym_pool) < num_samples and max_iter > 0:
            max_iter = max_iter - 1  # reduce the iteration to prevent exhausting loop
            nonsynonym_pool += self.nonsynonym_sampling_from_mappings(num_samples - len(nonsynonym_pool), max_iter)
            nonsynonym_pool = DataUtils.uniqify(nonsynonym_pool)
        return nonsynonym_pool


class TextSemanticsCorpora:
    r"""Class for creating the collection text semantics corpora.

    As defined in the $\textsf{BERTMap}$ paper, the collection of text semantics corpora contains
    **at least two intra-ontology sub-corpora** from the source and target ontologies, respectively.
    If some class mappings are provided, then a **cross-ontology sub-corpus** will be created.
    If some additional auxiliary ontologies are provided, the intra-ontology corpora created from them
    will serve as the **auxiliary sub-corpora**.

    Attributes:
        src_onto (Ontology): The source ontology to be matched or aligned.
        tgt_onto (Ontology): The target ontology to be matched or aligned.
        annotation_property_iris (List[str]): A list of annotation property IRIs used to extract the annotations.
        class_mappings (List[ReferenceMapping], optional): A list of cross-ontology class mappings between the
            source and the target ontologies. Defaults to `None`.
        auxiliary_ontos (List[Ontology], optional): A list of auxiliary ontologies for augmenting more synonym/non-synonym samples. Defaults to `None`.
    """

    def __init__(
        self,
        src_onto: Ontology,
        tgt_onto: Ontology,
        annotation_property_iris: List[str],
        class_mappings: Optional[List[ReferenceMapping]] = None,
        auxiliary_ontos: Optional[List[Ontology]] = None,
    ):
        self.synonyms = []
        self.nonsynonyms = []

        # build intra-ontology corpora
        # negative sample ratios are by default
        self.intra_src_onto_corpus = IntraOntologyTextSemanticsCorpus(src_onto, annotation_property_iris)
        self.add_samples_from_sub_corpus(self.intra_src_onto_corpus)
        self.intra_tgt_onto_corpus = IntraOntologyTextSemanticsCorpus(tgt_onto, annotation_property_iris)
        self.add_samples_from_sub_corpus(self.intra_tgt_onto_corpus)

        # build cross-ontolgoy corpora
        self.class_mappings = class_mappings
        self.cross_onto_corpus = None
        if self.class_mappings:
            self.cross_onto_corpus = CrossOntologyTextSemanticsCorpus(
                class_mappings, src_onto, tgt_onto, annotation_property_iris
            )
            self.add_samples_from_sub_corpus(self.cross_onto_corpus)

        # build auxiliary ontology corpora (same as intra-ontology)
        self.auxiliary_ontos = auxiliary_ontos
        self.auxiliary_onto_corpora = []
        if self.auxiliary_ontos:
            for auxiliary_onto in self.auxiliary_ontos:
                self.auxiliary_onto_corpora.append(
                    IntraOntologyTextSemanticsCorpus(auxiliary_onto, annotation_property_iris)
                )
        for auxiliary_onto_corpus in self.auxiliary_onto_corpora:
            self.add_samples_from_sub_corpus(auxiliary_onto_corpus)

        # DataUtils.uniqify the samples
        self.synonyms = DataUtils.uniqify(self.synonyms)
        self.nonsynonyms = DataUtils.uniqify(self.nonsynonyms)
        # remove invalid nonsynonyms
        self.nonsynonyms = list(set(self.nonsynonyms) - set(self.synonyms))

        # summary
        self.info = {
            type(self).__name__: {
                "num_synonyms": len(self.synonyms),
                "num_nonsynonyms": len(self.nonsynonyms),
                "intra_src_onto_corpus": self.intra_src_onto_corpus.info["IntraOntologyTextSemanticsCorpus"],
                "intra_tgt_onto_corpus": self.intra_tgt_onto_corpus.info["IntraOntologyTextSemanticsCorpus"],
                "cross_onto_corpus": self.cross_onto_corpus.info["CrossOntologyTextSemanticsCorpus"] if self.cross_onto_corpus else None,
                "auxiliary_onto_corpora": [a.info["IntraOntologyTextSemanticsCorpus"] for a in self.auxiliary_onto_corpora],
            }
        }

    def __str__(self):
        return FileUtils.print_dict(self.info)

    def save(self, save_path: str):
        """Save the overall text semantics corpora (a `.json` file for label pairs
        and its summary) in the specified directory.
        """
        FileUtils.create_path(save_path)
        save_json = {
            "summary": self.info,
            "synonyms": [(pos[0], pos[1], 1) for pos in self.synonyms],
            "nonsynonyms": [(neg[0], neg[1], 0) for neg in self.nonsynonyms],
        }
        FileUtils.save_file(save_json, os.path.join(save_path, "text-semantics.corpora.json"))

    def add_samples_from_sub_corpus(
        self, sub_corpus: Union[IntraOntologyTextSemanticsCorpus, CrossOntologyTextSemanticsCorpus]
    ):
        """Add synonyms and non-synonyms from each sub-corpus to the overall collection."""
        self.synonyms += sub_corpus.synonyms
        if isinstance(sub_corpus, IntraOntologyTextSemanticsCorpus):
            self.nonsynonyms += sub_corpus.soft_nonsynonyms + sub_corpus.hard_nonsynonyms
        else:
            self.nonsynonyms += sub_corpus.nonsynonyms
