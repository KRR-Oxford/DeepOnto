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

from typing import Optional, List, Set, Tuple
from yacs.config import CfgNode
import os
from textdistance import levenshtein
from logging import Logger
import itertools
import torch
import pandas as pd
import enlighten


from deeponto.align.mapping import EntityMapping
from deeponto.onto import Ontology
from deeponto.utils import FileUtils, Tokenizer
from .bert_classifier import BERTSynonymClassifier


# @paper(
#     "BERTMap: A BERT-based Ontology Alignment System (AAAI-2022)",
#     "https://ojs.aaai.org/index.php/AAAI/article/view/20510",
# )
class MappingPredictor:
    r"""Class for the mapping prediction module of $\textsf{BERTMap}$ and $\textsf{BERTMapLt}$ models.

    Attributes:
        tokenizer (Tokenizer): The tokenizer used for constructing the inverted annotation index and candidate selection.
        src_annotation_index (dict): A dictionary that stores the `(class_iri, class_annotations)` pairs from `src_onto` according to `annotation_property_iris`.
        tgt_annotation_index (dict): A dictionary that stores the `(class_iri, class_annotations)` pairs from `tgt_onto` according to `annotation_property_iris`.
        tgt_inverted_annotation_index (InvertedIndex): The inverted index built from `tgt_annotation_index` used for target class candidate selection.
        bert_synonym_classifier (BERTSynonymClassifier, optional): The BERT synonym classifier fine-tuned on text semantics corpora.
        num_raw_candidates (int): The maximum number of selected target class candidates for a source class.
        num_best_predictions (int): The maximum number of best scored mappings presevred for a source class.
        batch_size_for_prediction (int): The batch size of class annotation pairs for computing synonym scores.
    """

    def __init__(
        self,
        output_path: str,
        tokenizer_path: str,
        src_annotation_index: dict,
        tgt_annotation_index: dict,
        bert_synonym_classifier: Optional[BERTSynonymClassifier],
        num_raw_candidates: Optional[int],
        num_best_predictions: Optional[int],
        batch_size_for_prediction: int,
        logger: Logger,
        enlighten_manager: enlighten.Manager,
        enlighten_status: enlighten.StatusBar,
    ):
        self.logger = logger
        self.enlighten_manager = enlighten_manager
        self.enlighten_status = enlighten_status

        self.tokenizer = Tokenizer.from_pretrained(tokenizer_path)

        self.logger.info("Build inverted annotation index for candidate selection.")
        self.src_annotation_index = src_annotation_index
        self.tgt_annotation_index = tgt_annotation_index
        self.tgt_inverted_annotation_index = Ontology.build_inverted_annotation_index(
            tgt_annotation_index, self.tokenizer
        )
        # the fundamental judgement for whether bertmap or bertmaplt is loaded
        self.bert_synonym_classifier = bert_synonym_classifier
        self.num_raw_candidates = num_raw_candidates
        self.num_best_predictions = num_best_predictions
        self.batch_size_for_prediction = batch_size_for_prediction
        self.output_path = output_path

        self.init_class_mapping = lambda head, tail, score: EntityMapping(head, tail, "<EquivalentTo>", score)

    def bert_mapping_score(
        self,
        src_class_annotations: Set[str],
        tgt_class_annotations: Set[str],
    ):
        r"""$\textsf{BERTMap}$'s main mapping score module which utilises the fine-tuned BERT synonym
        classifier.

        Compute the **synonym score** for each pair of src-tgt class annotations, and return
        the **average** score as the mapping score. Apply string matching before applying the
        BERT module to filter easy mappings (with scores $1.0$).
        """
        # apply string matching before applying the bert module
        prelim_score = self.edit_similarity_mapping_score(
            src_class_annotations,
            tgt_class_annotations,
            string_match_only=True,
        )
        if prelim_score == 1.0:
            return prelim_score
        # apply BERT classifier and define mapping score := Average(SynonymScores)
        class_annotation_pairs = list(itertools.product(src_class_annotations, tgt_class_annotations))
        synonym_scores = self.bert_synonym_classifier.predict(class_annotation_pairs)
        # only one element tensor is able to be extracted as a scalar by .item()
        return float(torch.mean(synonym_scores).item())

    @staticmethod
    def edit_similarity_mapping_score(
        src_class_annotations: Set[str],
        tgt_class_annotations: Set[str],
        string_match_only: bool = False,
    ):
        r"""$\textsf{BERTMap}$'s string match module and $\textsf{BERTMapLt}$'s mapping prediction function.

        Compute the **normalised edit similarity** `(1 - normalised edit distance)` for each pair
        of src-tgt class annotations, and return the **maximum** score as the mapping score.
        """
        # edge case when src and tgt classes have an exact match of annotation
        if len(src_class_annotations.intersection(tgt_class_annotations)) > 0:
            return 1.0
        # a shortcut to save time for $\textsf{BERTMap}$
        if string_match_only:
            return 0.0
        annotation_pairs = itertools.product(src_class_annotations, tgt_class_annotations)
        sim_scores = [levenshtein.normalized_similarity(src, tgt) for src, tgt in annotation_pairs]
        return max(sim_scores)

    def mapping_prediction_for_src_class(self, src_class_iri: str) -> List[EntityMapping]:
        r"""Predict $N$ best scored mappings for a source ontology class, where
        $N$ is specified in `self.num_best_predictions`.

        1. Apply the **string matching** module to compute "easy" mappings.
        2. Return the mappings if found any, or if there is no BERT synonym classifier
        as in $\textsf{BERTMapLt}$.
        3. If using the BERT synonym classifier module:

            - Generate batches for class annotation pairs. Each batch contains the combinations of the
            source class annotations and $M$ target candidate classes' annotations. $M$ is determined
            by `batch_size_for_prediction`, i.e., stop adding annotations of a target class candidate into
            the current batch if this operation will cause the size of current batch to exceed the limit.
            - Compute the synonym scores for each batch and aggregate them into mapping scores; preserve
            $N$ best scored candidates and update them in the next batch. By this dynamic process, we eventually
            get $N$ best scored mappings for a source ontology class.
        """

        src_class_annotations = self.src_annotation_index[src_class_iri]
        # previously wrongly put tokenizer again !!!
        tgt_class_candidates = self.tgt_inverted_annotation_index.idf_select(
            list(src_class_annotations), pool_size=self.num_raw_candidates
        )  # [(tgt_class_iri, idf_score)]
        best_scored_mappings = []

        # for string matching: save time if already found string-matched candidates
        def string_match():
            """Compute string-matched mappings."""
            string_matched_mappings = []
            for tgt_candidate_iri, _ in tgt_class_candidates:
                tgt_candidate_annotations = self.tgt_annotation_index[tgt_candidate_iri]
                prelim_score = self.edit_similarity_mapping_score(
                    src_class_annotations,
                    tgt_candidate_annotations,
                    string_match_only=True,
                )
                if prelim_score > 0.0:
                    # if src_class_annotations.intersection(tgt_candidate_annotations):
                    string_matched_mappings.append(
                        self.init_class_mapping(src_class_iri, tgt_candidate_iri, prelim_score)
                    )

            return string_matched_mappings

        best_scored_mappings += string_match()
        # return string-matched mappings if found or if there is no bert module (bertmaplt)
        if best_scored_mappings or not self.bert_synonym_classifier:
            self.logger.info(f"The best scored class mappings for {src_class_iri} are\n{best_scored_mappings}")
            return best_scored_mappings

        def generate_batched_annotations(batch_size: int):
            """Generate batches of class annotations for the input source class and its
            target candidates.
            """
            batches = []
            # the `nums`` parameter determines how the annotations are grouped
            current_batch = CfgNode({"annotations": [], "nums": []})
            for i, (tgt_candidate_iri, _) in enumerate(tgt_class_candidates):
                tgt_candidate_annotations = self.tgt_annotation_index[tgt_candidate_iri]
                annotation_pairs = list(itertools.product(src_class_annotations, tgt_candidate_annotations))
                current_batch.annotations += annotation_pairs
                num_annotation_pairs = len(annotation_pairs)
                current_batch.nums.append(num_annotation_pairs)
                # collect when the batch is full or for the last target class candidate
                if sum(current_batch.nums) > batch_size or i == len(tgt_class_candidates) - 1:
                    batches.append(current_batch)
                    current_batch = CfgNode({"annotations": [], "nums": []})
            return batches

        def bert_match():
            """Compute mappings with fine-tuned BERT synonym classifier."""
            bert_matched_mappings = []
            class_annotation_batches = generate_batched_annotations(self.batch_size_for_prediction)
            batch_base_candidate_idx = (
                0  # after each batch, the base index will be increased by # of covered target candidates
            )
            device = self.bert_synonym_classifier.device

            # intialize N prediction scores and N corresponding indices w.r.t `tgt_class_candidates`
            final_best_scores = torch.tensor([-1] * self.num_best_predictions).to(device)
            final_best_idxs = torch.tensor([-1] * self.num_best_predictions).to(device)

            for annotation_batch in class_annotation_batches:

                synonym_scores = self.bert_synonym_classifier.predict(annotation_batch.annotations)
                # aggregating to mappings cores
                grouped_synonym_scores = torch.split(
                    synonym_scores,
                    split_size_or_sections=annotation_batch.nums,
                )
                mapping_scores = torch.stack([torch.mean(chunk) for chunk in grouped_synonym_scores])
                assert len(mapping_scores) == len(annotation_batch.nums)

                # preserve N best scored mappings
                # scale N in case there are less than N tgt candidates in this batch
                N = min(len(mapping_scores), self.num_best_predictions)
                batch_best_scores, batch_best_idxs = torch.topk(mapping_scores, k=N)
                batch_best_idxs += batch_base_candidate_idx

                # we do the substitution for every batch to prevent from memory overflow
                final_best_scores, _idxs = torch.topk(
                    torch.cat([batch_best_scores, final_best_scores]),
                    k=self.num_best_predictions,
                )
                final_best_idxs = torch.cat([batch_best_idxs, final_best_idxs])[_idxs]

                # update the index for target candidate classes
                batch_base_candidate_idx += len(annotation_batch.nums)

            for candidate_idx, mapping_score in zip(final_best_idxs, final_best_scores):
                # ignore intial values (-1.0) for dummy mappings
                # the threshold 0.9 is for mapping extension
                if mapping_score.item() >= 0.9:
                    tgt_candidate_iri = tgt_class_candidates[candidate_idx.item()][0]
                    bert_matched_mappings.append(
                        self.init_class_mapping(
                            src_class_iri,
                            tgt_candidate_iri,
                            mapping_score.item(),
                        )
                    )

            assert len(bert_matched_mappings) <= self.num_best_predictions
            self.logger.info(f"The best scored class mappings for {src_class_iri} are\n{bert_matched_mappings}")
            return bert_matched_mappings

        return bert_match()

    def mapping_prediction(self):
        r"""Apply global matching for each class in the source ontology.

        See [`mapping_prediction_for_src_class`][deeponto.align.bertmap.mapping_prediction.MappingPredictor.mapping_prediction_for_src_class].

        If this process is accidentally stopped, it can be resumed from already saved predictions. The progress
        bar keeps track of the number of source ontology classes that have been matched.
        """
        self.logger.info("Start global matching for each class in the source ontology.")

        match_dir = os.path.join(self.output_path, "match")
        try:
            mapping_index = FileUtils.load_file(os.path.join(match_dir, "raw_mappings.json"))
            self.logger.info("Load the existing mapping prediction file.")
        except:
            mapping_index = dict()
            FileUtils.create_path(match_dir)

        progress_bar = self.enlighten_manager.counter(
            total=len(self.src_annotation_index), desc="Mapping Prediction", unit="per src class"
        )
        self.enlighten_status.update(demo="Mapping Prediction")

        for i, src_class_iri in enumerate(self.src_annotation_index.keys()):
            if src_class_iri in mapping_index.keys():
                self.logger.info(f"[Class {i}] Skip matching {src_class_iri} as already computed.")
                progress_bar.update()
                continue
            mappings = self.mapping_prediction_for_src_class(src_class_iri)
            mapping_index[src_class_iri] = [m.to_tuple(with_score=True) for m in mappings]

            if i % 100 == 0 or i == len(self.src_annotation_index) - 1:
                FileUtils.save_file(mapping_index, os.path.join(match_dir, "raw_mappings.json"))
                # also save a .tsv version
                mapping_in_tuples = list(itertools.chain.from_iterable(mapping_index.values()))
                mapping_df = pd.DataFrame(mapping_in_tuples, columns=["SrcEntity", "TgtEntity", "Score"])
                mapping_df.to_csv(os.path.join(match_dir, "raw_mappings.tsv"), sep="\t", index=False)
                self.logger.info("Save currently computed mappings to prevent undesirable loss.")

            progress_bar.update()

        self.logger.info("Finished mapping prediction for each class in the source ontology.")
        progress_bar.close()
