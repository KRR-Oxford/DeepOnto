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

from typing import Optional, List, TYPE_CHECKING
import pprintpp
from collections import defaultdict
import pandas as pd
import random

from deeponto.onto import Ontology
from deeponto.utils import FileUtils, DataUtils, Tokenizer

if TYPE_CHECKING:
    from org.semanticweb.owlapi.model import OWLObject  # type: ignore

DEFAULT_REL = "<?rel>"
DUP_STRATEGIES = ["average", "kept_new", "kept_old"]
DEFAULT_DUP_STRATEGY = DUP_STRATEGIES[0]


##################################################################################
###                         basic mapping structure                            ###
##################################################################################


class EntityMapping:
    r"""A datastructure for entity mapping.

    Such entities should be named and have an IRI.

    Attributes:
        src_entity_iri (str): The IRI of the source entity, usually its IRI if available.
        tgt_entity_iri (str): The IRI of the target entity, usually its IRI if available.
        relation (str, optional): A symbol that represents what semantic relation this mapping stands for. Defaults to `<?rel>` which means unspecified.
            Suggested inputs are `"<EquivalentTo>"` and `"<SubsumedBy>"`.
        score (float, optional): The score that indicates the confidence of this mapping. Defaults to `0.0`.
    """

    def __init__(self, src_entity_iri: str, tgt_entity_iri: str, relation: str = DEFAULT_REL, score: float = 0.0):
        """Intialise an entity mapping.

        Args:
            src_entity_iri (str): The IRI of the source entity, usually its IRI if available.
            tgt_entity_iri (str): The IRI of the target entity, usually its IRI if available.
            relation (str, optional): A symbol that represents what semantic relation this mapping stands for. Defaults to `<?rel>` which means unspecified.
                Suggested inputs are `"<EquivalentTo>"` and `"<SubsumedBy>"`.
            score (float, optional): The score that indicates the confidence of this mapping. Defaults to `0.0`.
        """
        self.head = src_entity_iri
        self.tail = tgt_entity_iri
        self.relation = relation
        self.score = score

    @classmethod
    def from_owl_objects(
        cls, src_entity: OWLObject, tgt_entity: OWLObject, relation: str = DEFAULT_REL, score: float = 0.0
    ):
        """Create an entity mapping from two `OWLObject` entities which have an IRI.

        Args:
            src_entity (OWLObject): The source entity in `OWLObject`.
            tgt_entity (OWLObject): The target entity in `OWLObject`.
            relation (str, optional): A symbol that represents what semantic relation this mapping stands for. Defaults to `<?rel>` which means unspecified.
                Suggested inputs are `"<EquivalentTo>"` and `"<SubsumedBy>"`.
            score (float, optional): The score that indicates the confidence of this mapping. Defaults to `0.0`.
        Returns:
            (EntityMapping): The entity mapping created from the source and target entities.
        """
        return cls(str(src_entity.getIRI()), str(tgt_entity.getIRI()), relation, score)

    def to_tuple(self, with_score: bool = False):
        """Transform an entity mapping (`self`) to a tuple representation

        Note that `relation` is discarded and `score` is optionally preserved).
        """
        if with_score:
            return (self.head, self.tail, self.score)
        else:
            return (self.head, self.tail)

    @staticmethod
    def as_tuples(entity_mappings: List[EntityMapping], with_score: bool = False):
        """Transform a list of entity mappings to their tuple representations.

        Note that `relation` is discarded and `score` is optionally preserved).
        """
        return [m.to_tuple(with_score=with_score) for m in entity_mappings]

    @staticmethod
    def sort_entity_mappings_by_score(entity_mappings: List[EntityMapping], k: Optional[int] = None):
        r"""Sort the entity mappings in a list by their scores in descending order.

        Args:
            entity_mappings (List[EntityMapping]): A list entity mappings to sort.
            k (int, optional): The number of top $k$ scored entities preserved if specified. Defaults to `None` which
                means to return **all** entity mappings.

        Returns:
            (List[EntityMapping]): A list of sorted entity mappings.
        """
        return list(sorted(entity_mappings, key=lambda x: x.score, reverse=True))[:k]

    @staticmethod
    def read_table_mappings(
        table_of_mappings_file: str,
        threshold: Optional[float] = 0.0,
        relation: str = DEFAULT_REL,
        is_reference: bool = False,
    ) -> List[EntityMapping]:
        r"""Read entity mappings from `.csv` or `.tsv` files.
        
        !!! note "Mapping Table Format"
        
            The columns of the mapping table must have the headings: `"SrcEntity"`, `"TgtEntity"`, and `"Score"`.

        Args:
            table_of_mappings_file (str): The path to the table (`.csv` or `.tsv`) of mappings.
            threshold (Optional[float], optional): Mappings with scores less than `threshold` will not be loaded. Defaults to 0.0.
            relation (str, optional): A symbol that represents what semantic relation this mapping stands for. Defaults to `<?rel>` which means unspecified.
                Suggested inputs are `"<EquivalentTo>"` and `"<SubsumedBy>"`.
            is_reference (bool): Whether the loaded mappings are reference mappigns; if so, `threshold` is disabled and mapping scores
                are all set to $1.0$. Defaults to `False`.

        Returns:
            (List[EntityMapping]): A list of entity mappings loaded from the table file.
        """
        df = FileUtils.read_table(table_of_mappings_file)
        entity_mappings = []
        for _, dp in df.iterrows():
            if is_reference:
                entity_mappings.append(ReferenceMapping(dp["SrcEntity"], dp["TgtEntity"], relation))
            else:
                if dp["Score"] >= threshold:
                    entity_mappings.append(EntityMapping(dp["SrcEntity"], dp["TgtEntity"], relation, dp["Score"]))
        return entity_mappings

    def __repr__(self):
        return f"EntityMapping({self.head} {self.relation} {self.tail}, {round(self.score, 6)})"


class ReferenceMapping(EntityMapping):
    r"""A datastructure for entity mapping that acts as a reference mapping.

    A reference mapppings is a ground truth entity mapping (with $score = 1.0$) and can
    have several entity mappings as candidates. These candidate mappings should have the
    same `head` (i.e., source entity) as the reference mapping.

    Attributes:
        src_entity_iri (str): The IRI of the source entity, usually its IRI if available.
        tgt_entity_iri (str): The IRI of the target entity, usually its IRI if available.
        relation (str, optional): A symbol that represents what semantic relation this mapping stands for. Defaults to `<?rel>` which means unspecified.
            Suggested inputs are `"<EquivalentTo>"` and `"<SubsumedBy>"`.
    """

    def __init__(
        self,
        src_entity_iri: str,
        tgt_entity_iri: str,
        relation: str = DEFAULT_REL,
        candidate_mappings: Optional[List[EntityMapping]] = [],
    ):
        r"""Intialise a reference mapping.

        Args:
            src_entity_iri (str): The IRI of the source entity, usually its IRI if available.
            tgt_entity_iri (str): The IRI of the target entity, usually its IRI if available.
            relation (str, optional): A symbol that represents what semantic relation this mapping stands for. Defaults to `<?rel>` which means unspecified.
                Suggested inputs are `"<EquivalentTo>"` and `"<SubsumedBy>"`.
            candidate_mappings (List[EntityMapping], optional): A list of entity mappings that are candidates for this reference mapping. Defaults to `[]`.
        """
        super().__init__(src_entity_iri, tgt_entity_iri, relation, 1.0)
        self.candidates = []
        for candidate in candidate_mappings:
            self.add_candidate(candidate)

    def __repr__(self):
        reference_mapping_str = f"ReferenceMapping({self.head} {self.relation} {self.tail}, 1.0)"
        if self.candidates:
            candidate_mapping_str = pprintpp.pformat(self.candidates)
            reference_mapping_str += f" with candidates:\n{candidate_mapping_str}"
        return reference_mapping_str

    def add_candidate(self, candidate_mapping: EntityMapping):
        """Add a candidate mapping whose relation and head entity are the
        same as the reference mapping's.
        """
        if self.relation != candidate_mapping.relation:
            raise ValueError(
                f"Expect relation of candidate mapping to be {self.relation} but got {candidate_mapping.relation}"
            )
        if self.head != candidate_mapping.head:
            raise ValueError("Candidate mapping does not have the same head entity as the anchor mapping.")
        self.candidates.append(candidate_mapping)

    @staticmethod
    def read_table_mappings(table_of_mappings_file: str, relation: str = DEFAULT_REL):
        r"""Read reference mappings from `.csv` or `.tsv` files.
        
        !!! note "Mapping Table Format"
        
            The columns of the mapping table must have the headings: `"SrcEntity"`, `"TgtEntity"`, and `"Score"`.

        Args:
            table_of_mappings_file (str): The path to the table (`.csv` or `.tsv`) of mappings.
            relation (str, optional): A symbol that represents what semantic relation this mapping stands for. Defaults to `<?rel>` which means unspecified.
                Suggested inputs are `"<EquivalentTo>"` and `"<SubsumedBy>"`.

        Returns:
            (List[ReferenceMapping]): A list of reference mappings loaded from the table file.
        """
        return EntityMapping.read_table_mappings(table_of_mappings_file, relation=relation, is_reference=True)


class SubsFromEquivMappingGenerator:
    r"""Generating subsumption mappings from gold standard equivalence mappings.

    !!! credit "paper"

        The online subsumption mapping construction algorithm is proposed in the paper:
        [Machine Learning-Friendly Biomedical Datasets for Equivalence and Subsumption Ontology Matching (ISWC 2022)](https://link.springer.com/chapter/10.1007/978-3-031-19433-7_33).

    This generator has an attribute `delete_used_equiv_tgt_class` for determining whether or not to sabotage the equivalence
    mappings used to create $\geq 1$ subsumption mappings. The reason is that, if the equivalence mapping is broken, then the
    OM tool is expected to predict subsumption mappings directly without relying on the equivalence mappings as an intermediate.

    Attributes:
        src_onto (Ontology): The source ontology.
        tgt_onto (Ontology): The target ontology.
        equiv_class_pairs (List[Tuple[str, str]]): A list of class pairs (in IRIs) that are **equivalent** according to the input
            equivalence mappings.
        subs_generation_ratio (int, optional): The maximum number of subsumption mappings generated from each equivalence
            mapping. Defaults to `None` which means there is no limit on the number of subsumption mappings.
        delete_used_equiv_tgt_class (bool): Whether to mark the target side of an equivalence mapping **used** for creating
            at least one subsumption mappings as "deleted". Defaults to `True`.
    """

    def __init__(
        self,
        src_onto: Ontology,
        tgt_onto: Ontology,
        equiv_mappings: List[ReferenceMapping],
        subs_generation_ratio: Optional[int] = None,
        delete_used_equiv_tgt_class: bool = True,
    ):
        self.src_onto = src_onto
        self.tgt_onto = tgt_onto
        self.equiv_class_pairs = [m.to_tuple() for m in equiv_mappings]
        self.subs_generation_ratio = subs_generation_ratio
        self.delete_used_equiv_tgt_class = delete_used_equiv_tgt_class

        subs_from_equivs, self.used_equiv_tgt_class_iris = self.online_construction()
        # turn into triples with scores 1.0
        self.subs_from_equivs = [(c, p, 1.0) for c, p in subs_from_equivs]

    def online_construction(self):
        r"""An **online** algorithm for constructing subsumption mappings from gold standard equivalence mappings.

        Let $t$ denote the boolean value that indicates if the target class involved in an equivalence mapping
        will be deleted. If $t$ is true, then for each equivalent class pair $(c, c')$, do the following:

        1. If $c'$ has been inolved in a subsumption mapping, skip this pair as otherwise $c'$ will need to be deleted.
        2. For each parent class of $c'$, skip it if it has been marked deleted (i.e., involved in an equivalence mapping that has been used to create a subsumption mapping).
        3. If any subsumption mapping has been created from $(c, c')$, mark $c'$ as deleted.

        Steps 1 and 2 ensure that target classes that have been **involved in a subsumption mapping** have **no conflicts** with
        target classes that have been **used to create a subsumption mapping**.

        This algorithm is *online* because the construction and deletion depend on the order of the input equivalent class pairs.
        """
        subs_class_pairs = []
        in_subs = defaultdict(lambda: False)  # in a subsumption mapping
        used_equivs = defaultdict(lambda: False)  # in a used equivalence mapping

        for src_class_iri, tgt_class_iri in self.equiv_class_pairs:

            cur_subs_pairs = []

            # NOTE (1) an equiv pair is skipped if the target side is marked constructed
            if self.delete_used_equiv_tgt_class and in_subs[tgt_class_iri]:
                continue

            # construct subsumption pairs by matching the source class and the target class's parents
            tgt_class = self.tgt_onto.get_owl_object_from_iri(tgt_class_iri)
            tgt_class_parent_iris = self.tgt_onto.reasoner.super_entities_of(tgt_class, direct=True)
            for parent_iri in tgt_class_parent_iris:
                # skip this parent if it is marked as "used"
                if self.delete_used_equiv_tgt_class and used_equivs[parent_iri]:
                    continue
                cur_subs_pairs.append((src_class_iri, parent_iri))
                # if successfully created, mark this parent as "in"
                if self.delete_used_equiv_tgt_class:
                    in_subs[parent_iri] = True

            # mark the target class as "used" because it has been used for creating a subsumption mapping
            if self.delete_used_equiv_tgt_class and cur_subs_pairs:
                used_equivs[tgt_class_iri] = True

            if self.subs_generation_ratio and len(cur_subs_pairs) > self.subs_generation_ratio:
                cur_subs_pairs = random.sample(cur_subs_pairs, self.subs_generation_ratio)
            subs_class_pairs += cur_subs_pairs

        used_equiv_tgt_class_iris = None
        if self.delete_used_equiv_tgt_class:
            used_equiv_tgt_class_iris = [iri for iri, used in used_equivs.items() if used is True]
            print(
                f"{len(used_equiv_tgt_class_iris)}/{len(self.equiv_class_pairs)} are used for creating at least one subsumption mapping."
            )

        subs_class_pairs = DataUtils.uniqify(subs_class_pairs)
        print(f"{len(subs_class_pairs)} subsumption mappings are created in the end.")

        return subs_class_pairs, used_equiv_tgt_class_iris

    def save_subs(self, save_path: str):
        """Save the constructed subsumption mappings (in tuples) to a local `.tsv` file."""
        subs_df = pd.DataFrame(self.subs_from_equivs, columns=["SrcEntity", "TgtEntity", "Score"])
        subs_df.to_csv(save_path, sep="\t", index=False)


# TODO: to be updated constantly
SAMPLING_OPTIONS = ["idf", "neighbour", "random"]

class NegativeCandidateMappingGenerator:
    r"""Generating **negative** candidate mappings for each gold standard mapping.

    Note that the source side of the golden standard mapping is fixed, i.e., candidate mappings are generated
    according to the target side.

    !!! credit "paper"

        The candidate mapping generation algorithm is proposed in the paper:
        [Machine Learning-Friendly Biomedical Datasets for Equivalence and Subsumption Ontology Matching (ISWC 2022)](https://link.springer.com/chapter/10.1007/978-3-031-19433-7_33).
    """

    def __init__(
        self,
        src_onto: Ontology,
        tgt_onto: Ontology,
        reference_class_mappings: List[ReferenceMapping],  # equivalence or subsumption
        annotation_property_iris: List[str],  # for text-based candidates
        tokenizer: Tokenizer,  # for text-based candidates
        max_hops: int = 5,  # for graph-based candidates
        for_subsumption: bool = False,  # if for subsumption, avoid adding ancestors as candidates
    ):

        self.src_onto = src_onto
        self.tgt_onto = tgt_onto
        self.reference_class_mappings = reference_class_mappings
        self.reference_class_dict = defaultdict(list)  # to prevent wrongly adding negative candidates
        for m in self.reference_class_mappings:
            src_class_iri, tgt_class_iri = m.to_tuple()
            self.reference_class_dict[src_class_iri].append(tgt_class_iri)

        # for IDF sample
        self.tgt_annotation_index, self.annotation_property_iris = self.tgt_onto.build_annotation_index(
            annotation_property_iris
        )
        self.tokenizer = tokenizer
        self.tgt_inverted_annotation_index = self.tgt_onto.build_inverted_annotation_index(
            self.tgt_annotation_index, self.tokenizer
        )

        # for neighbour sample
        self.max_hops = max_hops

        # if for subsumption, avoid adding ancestors as candidates
        self.for_subsumption = for_subsumption
        # if for subsumption, add (src_class, tgt_class_ancestor) into the reference mappings
        if self.for_subsumption:
            for m in self.reference_class_mappings:
                src_class_iri, tgt_class_iri = m.to_tuple()
                tgt_class = self.tgt_onto.get_owl_object_from_iri(tgt_class_iri)
                tgt_class_ancestors = self.tgt_onto.reasoner.super_entities_of(tgt_class)
                for tgt_ancestor_iri in tgt_class_ancestors:
                    self.reference_class_dict[src_class_iri].append(tgt_ancestor_iri)


    def mixed_sample(self, reference_class_mapping: ReferenceMapping, **strategy2nums):
        """A mixed sampling approach that combines several sampling strategies.
        
        As introduced in the Bio-ML paper, this mixed approach guarantees that the number of samples for each
        strategy is either the **maximum that can be sampled** or the required number.
        
        Specifically, at each sampling iteration, the number of candidates is **first increased by the number of 
        previously sampled candidates**, as in the worst case, all the candidates sampled at this iteration
        will be duplicated with the previous. 
        
        The random sampling is used as the amending strategy, i.e., if other sampling strategies cannot retrieve
        the specified number of samples, then use random sampling to amend the number.
        
        Args:
            reference_class_mapping (ReferenceMapping): The reference class mapping for generating the candidate mappings.
            **strategy2nums (int): The keyword arguments that specify the expected number of candidates for each
                sampling strategy.
        """

        valid_tgt_candidate_iris = []
        sample_stats = defaultdict(lambda: 0)
        i = 0
        total_num_candidates = 0
        for strategy, num_canddiates in strategy2nums.items():
            i += 1
            if strategy in SAMPLING_OPTIONS:
                sampler = getattr(self, f"{strategy}_sample")
                # for ith iteration, the worst case is when all n_cands are duplicated
                # or should be excluded from other reference targets so we generate
                # NOTE:  total_num_candidates + num_candidates + len(excluded_tgt_class_iris)
                # candidates first and prune the rest; another edge case is when sampled
                # candidates are not sufficient and we use random sample to meet n_cands
                cur_valid_tgt_candidate_iris = sampler(
                    reference_class_mapping, total_num_candidates + num_canddiates
                )
                # remove the duplicated candidates (and excluded refs) and prune the tail
                cur_valid_tgt_candidate_iris = list(
                    set(cur_valid_tgt_candidate_iris) - set(valid_tgt_candidate_iris)
                )[:num_canddiates]
                sample_stats[strategy] += len(cur_valid_tgt_candidate_iris)
                # use random samples for complementation if not enough
                while len(cur_valid_tgt_candidate_iris) < num_canddiates:
                    amend_candidate_iris = self.random_sample(
                        reference_class_mapping, num_canddiates - len(cur_valid_tgt_candidate_iris)
                    )
                    amend_candidate_iris = list(
                        set(amend_candidate_iris)
                        - set(valid_tgt_candidate_iris)
                        - set(cur_valid_tgt_candidate_iris)
                    )
                    cur_valid_tgt_candidate_iris += amend_candidate_iris
                assert len(cur_valid_tgt_candidate_iris) == num_canddiates
                # record how many random samples to amend
                if strategy != "random":
                    sample_stats["random"] += num_canddiates - sample_stats[strategy]
                valid_tgt_candidate_iris += cur_valid_tgt_candidate_iris
                total_num_candidates += num_canddiates
            else:
                raise ValueError(f"Invalid sampling trategy: {strategy}.")
        assert len(valid_tgt_candidate_iris) == total_num_candidates
        
        # TODO: add the candidate mappings into the reference mapping 
        
        return valid_tgt_candidate_iris, sample_stats

    def random_sample(self, reference_class_mapping: ReferenceMapping, num_candidates: int):
        r"""**Randomly** sample a set of target class candidates $c'_{cand}$ for a given reference mapping $(c, c')$.

        The sampled candidate classes will be combined with the source reference class $c$ to get a set of
        candidate mappings $\{(c, c'_{cand})\}$.
        
        Args:
            reference_class_mapping (ReferenceMapping): The reference class mapping for generating the candidate mappings.
            num_candidates (int): The expected number of candidate mappings to generate.
        """
        ref_src_class_iri, ref_tgt_class_iri = reference_class_mapping.to_tuple()
        all_tgt_class_iris = set(self.tgt_onto.owl_classes.keys())
        valid_tgt_class_iris = all_tgt_class_iris - set(
            self.reference_class_dict[ref_src_class_iri]
        )  # exclude gold standards
        assert not ref_tgt_class_iri in valid_tgt_class_iris
        return random.sample(valid_tgt_class_iris, num_candidates)

    def idf_sample(self, reference_class_mapping: ReferenceMapping, num_candidates: int):
        r"""Sample a set of target class candidates $c'_{cand}$ for a given reference mapping $(c, c')$ based on the $idf$ scores
        w.r.t. the inverted annotation index (sub-word level).

        Candidate classes with higher $idf$ scores will be considered first, and then combined with the source reference class $c$
        to get a set of candidate mappings $\{(c, c'_{cand})\}$.

        Args:
            reference_class_mapping (ReferenceMapping): The reference class mapping for generating the candidate mappings.
            num_candidates (int): The expected number of candidate mappings to generate.
        """
        ref_src_class_iri, ref_tgt_class_iri = reference_class_mapping.to_tuple()

        tgt_candidates = self.tgt_inverted_annotation_index.idf_select(
            self.tgt_annotation_index[ref_tgt_class_iri]
        )  # select all non-trivial candidates first
        valid_tgt_class_iris = []
        for tgt_candidate_iri, _ in tgt_candidates:
            # valid as long as it is not one of the reference target
            if tgt_candidate_iri not in self.reference_class_dict[ref_src_class_iri]:
                valid_tgt_class_iris.append(tgt_candidate_iri)
            if len(valid_tgt_class_iris) == num_candidates:
                break
        assert not ref_tgt_class_iri in valid_tgt_class_iris
        return valid_tgt_class_iris

    def neighbour_sample(self, reference_class_mapping: ReferenceMapping, num_candidates: int):
        r"""Sample a set of target class candidates $c'_{cand}$ for a given reference mapping $(c, c')$ based on the **subsumption
        hierarchy**.

        Define one-hop as one edge derived from an **asserted** subsumption axiom, i.e., to the parent class or the child class.
        Candidates classes with nearer hops will be considered first, and then combined with the source reference class $c$
        to get a set of candidate mappings $\{(c, c'_{cand})\}$.
        
        Args:
            reference_class_mapping (ReferenceMapping): The reference class mapping for generating the candidate mappings.
            num_candidates (int): The expected number of candidate mappings to generate.
        """
        ref_src_class_iri, ref_tgt_class_iri = reference_class_mapping.to_tuple()

        valid_tgt_class_iris = set()
        cur_hop = 1
        frontier = [ref_tgt_class_iri]
        # extract from the nearest neighbours until enough candidates or max hop
        while len(valid_tgt_class_iris) < num_candidates and cur_hop <= self.max_hops:

            neighbours_of_cur_hop = []
            for tgt_class_iri in frontier:
                tgt_class = self.tgt_onto.get_owl_object_from_iri(tgt_class_iri)
                parents = self.tgt_onto.reasoner.super_entities_of(tgt_class, direct=True)
                children = self.tgt_onto.reasoner.sub_entities_of(tgt_class, direct=True)
                neighbours_of_cur_hop += parents + children  # used for further hop expansion
            
            valid_neighbours_of_cur_hop = set(neighbours_of_cur_hop) - set(self.reference_class_dict[ref_src_class_iri])
            # print(valid_neighbours_of_cur_hop)

            # NOTE if by adding neighbours of current hop the require number will be met
            # we randomly pick among them
            if len(valid_neighbours_of_cur_hop) > num_candidates - len(valid_tgt_class_iris):
                valid_neighbours_of_cur_hop = random.sample(
                    valid_neighbours_of_cur_hop, num_candidates - len(valid_tgt_class_iris)
                )
            valid_tgt_class_iris.update(valid_neighbours_of_cur_hop)

            frontier = neighbours_of_cur_hop  # update the frontier with all possible neighbors
            cur_hop += 1

        assert not ref_tgt_class_iri in valid_tgt_class_iris
        return list(valid_tgt_class_iris)
