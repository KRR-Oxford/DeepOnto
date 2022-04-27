import os
import sys

main_dir = os.getcwd().split("DeepOnto")[0] + "DeepOnto/src"
sys.path.append(main_dir)

from deeponto.data.align import NegativeCandidateGenerator
from deeponto.onto import Ontology
from deeponto.onto.text import Tokenizer

import click


@click.command()
@click.option("-s", "--src_onto_path", type=click.Path())
@click.option("-t", "--tgt_onto_path", type=click.Path())
@click.option("-r", "--ref_mappings_path", type=click.Path())
@click.option("-n", "--null_ref_mappings_path", type=click.Path())
@click.option(
    "-tp", "--tokenizer_path", type=click.Path(), default="emilyalsentzer/Bio_ClinicalBERT"
)
@click.option("-in", "--idf_sample_num", type=int, default=50)
@click.option("-gn", "--neighbour_sample_num", type=int, default=50)
@click.option("-rn", "--random_sample_num", type=int, default=0)
@click.option("-mh", "--max_hobs", type=int, default=6)
@click.option("-aa", "--avoid_ancestors", type=bool, default=False)
@click.option("-ac", "--avoid_descendents", type=bool, default=False)
def main(
    src_onto_path: str,
    tgt_onto_path: str,
    ref_mappings_path: str,
    null_ref_mappings_path: str,
    tokenizer_path: str,
    idf_sample_num: int,
    neighbour_sample_num: int,
    random_sample_num: int,
    max_hobs: int,
    avoid_ancestors: bool,
    avoid_descendents: bool,
):
    lab_props = [
        "label",
        "hasExactSynonym",
        "exactMatch",
        "alternative_term",
        "symbol",
        "P90",
        "prefLabel",
        "altLabel",
    ]
    tokenizer = Tokenizer.from_pretrained(tokenizer_path)
    src_onto = Ontology.from_new(src_onto_path, lab_props, tokenizer)
    tgt_onto = Ontology.from_new(tgt_onto_path, lab_props, tokenizer)
    sampler = NegativeCandidateGenerator(
        src_onto,
        tgt_onto,
        ref_mappings_path,
        null_ref_mappings_path,
        rel="=",
        tokenizer=tokenizer,
        max_hops=max_hobs,
        avoid_ancestors=avoid_ancestors,
        avoid_descendents=avoid_descendents,
    )

    sampler.sample_for_all_one_side(
        neighbour=neighbour_sample_num, idf=idf_sample_num, random=random_sample_num
    )


main()
