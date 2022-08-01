import os
import sys

main_dir = os.getcwd().split("DeepOnto")[0] + "DeepOnto/src"
sys.path.append(main_dir)

from deeponto.data.align import NegativeCandidateGenerator
from deeponto.onto import Ontology
from deeponto.onto.text import Tokenizer
from deeponto.utils import set_seed, print_choices
from deeponto.utils.logging import banner_msg
from deeponto import SavedObj

import click


@click.command()
@click.option("-o", "--saved_path", type=click.Path(exists=True), default=".")
@click.option("-s", "--src_onto_path", type=click.Path())
@click.option("-t", "--tgt_onto_path", type=click.Path())
@click.option("-r", "--ref_mappings_path", type=click.Path())
@click.option("-n", "--null_ref_mappings_path", type=click.Path())
@click.option(
    "-tp", "--tokenizer_path", type=click.Path(), default="emilyalsentzer/Bio_ClinicalBERT"
)
@click.option("-in", "--idf_sample_num", type=int, default=50)
@click.option("-gn", "--neighbour_sample_num", type=int, default=50)
@click.option("-mh", "--max_hops", type=int, default=6)
@click.option("-rn", "--random_sample_num", type=int, default=0)
def main(
    saved_path: str,
    src_onto_path: str,
    tgt_onto_path: str,
    ref_mappings_path: str,
    null_ref_mappings_path: str,
    tokenizer_path: str,
    idf_sample_num: int,
    neighbour_sample_num: int,
    max_hops: int,
    random_sample_num: int,
):
    set_seed(42)

    banner_msg("Choose a Supported OM Relation")
    supported_relations = ["'=' (equivalence)", "'<' (subClassOf)", "'>' (superClassOf)"]
    print_choices(supported_relations)
    rel = supported_relations[click.prompt("Enter a number", type=int)]
    avoid_ancestors, avoid_descendents = False, False
    if rel == "'<' (subClassOf)":
        avoid_ancestors = True
    if rel == "'>' (superClassOf)":
        avoid_descendents = True

    # using BERTMap's configuration for label properties
    lab_props = SavedObj.load_json("./config/bertmap.json")["lab_props"]
    tokenizer = Tokenizer.from_pretrained(tokenizer_path)
    src_onto = Ontology.from_new(src_onto_path, lab_props, tokenizer)
    tgt_onto = Ontology.from_new(tgt_onto_path, lab_props, tokenizer)
    sampler = NegativeCandidateGenerator(
        saved_path,
        src_onto,
        tgt_onto,
        ref_mappings_path,
        null_ref_mappings_path,
        rel="=",
        tokenizer=tokenizer,
        max_hops=max_hops,
        avoid_ancestors=avoid_ancestors,
        avoid_descendents=avoid_descendents,
    )

    sampler.sample_for_all_one_side(
        neighbour=neighbour_sample_num, idf=idf_sample_num, random=random_sample_num
    )


main()
