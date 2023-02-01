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

from typing import List, Tuple
import os
from logging import Logger
import itertools
import random
import pandas as pd
import enlighten


from deeponto.align.mapping import EntityMapping
from deeponto.onto import Ontology
from deeponto.utils import FileUtils, Tokenizer
from deeponto.utils.decorators import paper
from deeponto.align.logmap import run_logmap_repair
from .mapping_prediction import MappingPredictor


# @paper(
#     "BERTMap: A BERT-based Ontology Alignment System (AAAI-2022)",
#     "https://ojs.aaai.org/index.php/AAAI/article/view/20510",
# )
class MappingRefiner:
    r"""Class for the mapping refinement module of $\textsf{BERTMap}$.
    
    $\textsf{BERTMapLt}$ does not go through mapping refinement for its being "light".
    All the attributes of this class are supposed to be passed from `BERTMapPipeline`.

    Attributes:
        src_onto (Ontology): The source ontology to be matched.
        tgt_onto (Ontology): The target ontology to be matched.
        mapping_predictor (MappingPredictor): The mapping prediction module of BERTMap.
        mapping_extension_threshold (float): Mappings with scores $\geq$ this value will be considered in the iterative mapping extension process.
        raw_mappings (List[EntityMapping]): List of **raw class mappings** predicted in the **global matching** phase.
        mapping_score_dict (dict): A dynamic dictionary that keeps track of mappings (with scores) that have already been computed.
        mapping_filter_threshold (float): Mappings with scores $\geq$ this value will be preserved for the final mapping repairing. 
    """
    def __init__(
        self,
        output_path: str,
        src_onto: Ontology,
        tgt_onto: Ontology,
        mapping_predictor: MappingPredictor,
        mapping_extension_threshold: float,
        mapping_filtered_threshold: float,
        logger: Logger,
        enlighten_manager: enlighten.Manager,
        enlighten_status: enlighten.StatusBar
    ):
        self.output_path = output_path
        self.logger = logger
        self.enlighten_manager = enlighten_manager
        self.enlighten_status = enlighten_status
        
        self.src_onto = src_onto
        self.tgt_onto = tgt_onto

        # iterative mapping extension
        self.mapping_predictor = mapping_predictor
        self.mapping_extension_threshold = mapping_extension_threshold  # \kappa
        self.raw_mappings = EntityMapping.read_table_mappings(
            os.path.join(self.output_path, "match", "raw_mappings.tsv"),
            threshold=self.mapping_extension_threshold,
            relation="<EquivalentTo>",
        )
        # keep track of already scored mappings to prevent duplicated predictions
        self.mapping_score_dict = dict()
        for m in self.raw_mappings:
            src_class_iri, tgt_class_iri, score = m.to_tuple(with_score=True)
            self.mapping_score_dict[(src_class_iri, tgt_class_iri)] = score

        # the threshold for final filtering the extended mappings
        self.mapping_filtered_threshold = mapping_filtered_threshold  # \lambda

        # logmap mapping repair folder
        self.logmap_repair_path = os.path.join(self.output_path, "match", "logmap-repair")
        
        # paths for mapping extension and repair
        self.extended_mapping_path = os.path.join(self.output_path, "match", "extended_mappings.tsv")
        self.filtered_mapping_path = os.path.join(self.output_path, "match", "filtered_mappings.tsv")
        self.repaired_mapping_path = os.path.join(self.output_path, "match", "repaired_mappings.tsv")

    def mapping_extension(self, max_iter: int = 10):
        r"""Iterative mapping extension based on the locality principle.
        
        For each class pair $(c, c')$ (scored in the global matching phase) with score 
        $\geq \kappa$, search for plausible mappings between the parents of $c$ and $c'$,
        and between the children of $c$ and $c'$. This is an iterative process as the set 
        newly discovered mappings can act renew the frontier for searching. Terminate if
        no new mappings with score $\geq \kappa$ can be found or the limit `max_iter` has 
        been reached. Note that $\kappa$ is set to $0.9$ by default (can be altered
        in the configuration file). The mapping extension progress bar keeps track of the 
        total number of extended mappings (including the previously predicted ones).
        
        A further filtering will be performed by only preserving mappings with score $\geq \lambda$,
        in the original BERTMap paper, $\lambda$ is determined by the validation mappings, but
        in practice $\lambda$ is not a sensitive hyperparameter and validation mappings are often
        not available. Therefore, we manually set $\lambda$ to $0.9995$ by default (can be altered
        in the configuration file). The mapping filtering progress bar keeps track of the 
        total number of filtered mappings (this bar is purely for logging purpose).

        Args:
            max_iter (int, optional): The maximum number of mapping extension iterations. Defaults to `10`.
        """
        
        num_iter = 0
        self.enlighten_status.update(demo="Mapping Extension")
        extension_progress_bar = self.enlighten_manager.counter(
            desc=f"Mapping Extension [Iteration #{num_iter}]", unit="mapping"
        )
        filtering_progress_bar = self.enlighten_manager.counter(
            desc=f"Mapping Filtering", unit="mapping"
        )

        if os.path.exists(self.extended_mapping_path) and os.path.exists(self.filtered_mapping_path):
            self.logger.info(
                f"Found extended and filtered mapping files at {self.extended_mapping_path}"
                + f" and {self.filtered_mapping_path}.\nPlease check file integrity; if incomplete, "
                + "delete them and re-run the program."
            )
            
            # for animation purposes
            extension_progress_bar.desc = f"Mapping Extension"
            for _ in EntityMapping.read_table_mappings(self.extended_mapping_path):
                extension_progress_bar.update()
                
            self.enlighten_status.update(demo="Mapping Filtering")
            for _ in EntityMapping.read_table_mappings(self.filtered_mapping_path):
                filtering_progress_bar.update()
            
            extension_progress_bar.close()
            filtering_progress_bar.close()
            
            return
        # intialise the frontier, explored, final expansion sets with the raw mappings
        # NOTE be careful of address pointers
        frontier = [m.to_tuple() for m in self.raw_mappings]
        expansion = [m.to_tuple(with_score=True) for m in self.raw_mappings]
        # for animation purposes
        for _ in range(len(expansion)):
            extension_progress_bar.update()

        self.logger.info(
            f"Start mapping extension for each class pair with score >= {self.mapping_extension_threshold}."
        )
        while frontier and num_iter < max_iter:
            new_mappings = []
            for src_class_iri, tgt_class_iri in frontier:
                # one hop extension makes sure new mappings are really "new"
                cur_new_mappings = self.one_hop_extend(src_class_iri, tgt_class_iri)
                extension_progress_bar.update(len(cur_new_mappings))
                new_mappings += cur_new_mappings
            # add new mappings to the expansion set
            expansion += new_mappings
            # renew frontier with the newly discovered mappings
            frontier = [(x, y) for x, y, _ in new_mappings]

            self.logger.info(f"Add {len(new_mappings)} mappings at iteration #{num_iter}.")
            num_iter += 1
            extension_progress_bar.desc = f"Mapping Extension [Iteration #{num_iter}]"

        num_extended = len(expansion) - len(self.raw_mappings)
        self.logger.info(
            f"Finished iterative mapping extension with {num_extended} new mappings and in total {len(expansion)} extended mappings."
        )

        extended_mapping_df = pd.DataFrame(expansion, columns=["SrcEntity", "TgtEntity", "Score"])
        extended_mapping_df.to_csv(self.extended_mapping_path, sep="\t", index=False)

        self.enlighten_status.update(demo="Mapping Filtering")
        
        filtered_expansion = [
            (src, tgt, score) for src, tgt, score in expansion if score >= self.mapping_filtered_threshold
        ]
        self.logger.info(
            f"Filtered the extended mappings by a threshold of {self.mapping_filtered_threshold}."
            + f"There are {len(filtered_expansion)} mappings left for mapping repair."
        )
        
        for _ in range(len(filtered_expansion)):
            filtering_progress_bar.update()

        filtered_mapping_df = pd.DataFrame(filtered_expansion, columns=["SrcEntity", "TgtEntity", "Score"])
        filtered_mapping_df.to_csv(self.filtered_mapping_path, sep="\t", index=False)
        
        extension_progress_bar.close()
        filtering_progress_bar.close()
        return filtered_expansion

    def one_hop_extend(self, src_class_iri: str, tgt_class_iri: str, pool_size: int = 200):
        r"""Extend mappings from a scored class pair $(c, c')$ by
        searching from one-hop neighbors.

        Search for plausible mappings between the parents of $c$ and $c'$,
        and between the children of $c$ and $c'$. Mappings that are not
        already computed (recorded in `self.mapping_score_dict`) and have
        a score $\geq$ `self.mapping_extension_threshold` will be returned as
        **new** mappings.

        Args:
            src_class_iri (str): The IRI of the source ontology class $c$.
            tgt_class_iri (str): The IRI of the target ontology class $c'$.
            pool_size (int, optional): The maximum number of plausible mappings to be extended. Defaults to 200.

        Returns:
            (List[EntityMapping]): A list of one-hop extended mappings.
        """

        src_class = self.src_onto.get_owl_object_from_iri(src_class_iri)
        src_class_parent_iris = self.src_onto.reasoner.super_entities_of(src_class, direct=True)
        src_class_children_iris = self.src_onto.reasoner.sub_entities_of(src_class, direct=True)

        tgt_class = self.tgt_onto.get_owl_object_from_iri(tgt_class_iri)
        tgt_class_parent_iris = self.tgt_onto.reasoner.super_entities_of(tgt_class, direct=True)
        tgt_class_children_iris = self.tgt_onto.reasoner.sub_entities_of(tgt_class, direct=True)

        # pair up parents and children, respectively; NOTE set() might not be necessary
        parent_pairs = list(set(itertools.product(src_class_parent_iris, tgt_class_parent_iris)))
        children_pairs = list(set(itertools.product(src_class_children_iris, tgt_class_children_iris)))

        candidate_pairs = parent_pairs + children_pairs
        # downsample if the number of candidates is too large
        if len(candidate_pairs) > pool_size:
            candidate_pairs = random.sample(candidate_pairs, pool_size)

        extended_mappings = []
        for src_candidate_iri, tgt_candidate_iri in parent_pairs + children_pairs:

            # if already computed meaning that it is not a new mapping
            if (src_candidate_iri, tgt_candidate_iri) in self.mapping_score_dict:
                continue

            src_candidate_annotations = self.mapping_predictor.src_annotation_index[src_candidate_iri]
            tgt_candidate_annotations = self.mapping_predictor.tgt_annotation_index[tgt_candidate_iri]
            score = self.mapping_predictor.bert_mapping_score(src_candidate_annotations, tgt_candidate_annotations)
            # add to already scored collection
            self.mapping_score_dict[(src_candidate_iri, tgt_candidate_iri)] = score

            # skip mappings with low scores
            if score < self.mapping_extension_threshold:
                continue

            extended_mappings.append((src_candidate_iri, tgt_candidate_iri, score))

        self.logger.info(
            f"New mappings (in tuples) extended from {(src_class_iri, tgt_class_iri)} are:\n" + f"{extended_mappings}"
        )

        return extended_mappings

    def mapping_repair(self):
        """Repair the filtered mappings with LogMap's debugger.
        
        !!! note
            
            A sub-folder under `match` named `logmap-repair` contains LogMap-related intermediate files.
        """
        
        # progress bar for animation purposes
        self.enlighten_status.update(demo="Mapping Repairing")
        repair_progress_bar = self.enlighten_manager.counter(
            desc=f"Mapping Repairing", unit="mapping"
        )
        
        # skip repairing if already found the file
        if os.path.exists(self.repaired_mapping_path):
            self.logger.info(
                f"Found the repaired mapping file at {self.repaired_mapping_path}."
                + "\nPlease check file integrity; if incomplete, "
                + "delete it and re-run the program."
            )
            # update progress bar for animation purposes
            for _ in EntityMapping.read_table_mappings(self.repaired_mapping_path):
                repair_progress_bar.update()
            repair_progress_bar.close()
            return 

        # start mapping repair
        self.logger.info("Repair the filtered mappings with LogMap debugger.")
        # formatting the filtered mappings
        self.logmap_repair_formatting()
        
        # run the LogMap repair module on the extended mappings
        run_logmap_repair(
            self.src_onto.owl_path,
            self.tgt_onto.owl_path,
            os.path.join(self.logmap_repair_path, f"filtered_mappings_for_LogMap_repair.txt"),
            self.logmap_repair_path,
        )

        # create table mappings from LogMap repair outputs
        with open(os.path.join(self.logmap_repair_path, "mappings_repaired_with_LogMap.tsv"), "r") as f:
            lines = f.readlines()
        with open(os.path.join(self.output_path, "match", "repaired_mappings.tsv"), "w+") as f:
            f.write("SrcEntity\tTgtEntity\tScore\n")
            for line in lines:
                src_ent_iri, tgt_ent_iri, score = line.split("\t")
                f.write(f"{src_ent_iri}\t{tgt_ent_iri}\t{score}")
                repair_progress_bar.update()

        self.logger.info("Mapping repair finished.")
        repair_progress_bar.close()

    def logmap_repair_formatting(self):
        """Transform the filtered mapping file into the LogMap format.
        
        An auxiliary function of the mapping repair module which requires mappings
        to be formatted as LogMap's input format.
        """
        # read the filtered mapping file and convert to tuples
        filtered_mappings = EntityMapping.read_table_mappings(self.filtered_mapping_path)
        filtered_mappings_in_tuples = [m.to_tuple(with_score=True) for m in filtered_mappings]
        
        # write the mappings into logmap format
        lines = []
        for src_class_iri, tgt_class_iri, score in filtered_mappings_in_tuples:
            lines.append(f"{src_class_iri}|{tgt_class_iri}|=|{score}|CLS\n")
            
        # create a path to prevent error
        FileUtils.create_path(self.logmap_repair_path)
        formatted_file = os.path.join(self.logmap_repair_path, f"filtered_mappings_for_LogMap_repair.txt")
        with open(formatted_file, "w") as f:
            f.writelines(lines)
            
        return lines
