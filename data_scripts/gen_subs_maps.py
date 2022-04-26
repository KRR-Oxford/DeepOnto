import os
import sys

main_dir = os.getcwd().split("DeepOnto")[0] + "DeepOnto/src"
sys.path.append(main_dir)

from deeponto.data.align import SubsumptionMappingGenerator
from deeponto.onto.graph.pruning import preserve_classes
from deeponto.onto import Ontology
from deeponto import SavedObj
from deeponto.onto.text.text_utils import unfold_iri, abbr_iri
import pandas as pd

import click


@click.command()
@click.option("-s", "--src_onto_path", type=click.Path())
@click.option("-t", "--tgt_onto_path", type=click.Path())
@click.option("-r", "--equiv_mappings_path", type=click.Path())
@click.option("-n", "--map_ratio", type=int, default=1)
@click.option("-ds", "--delete_equiv_src", type=bool, default=False)
@click.option("-dt", "--delete_equiv_tgt", type=bool, default=True)
def main(
    src_onto_path: str,
    tgt_onto_path: str,
    equiv_mappings_path: str,
    map_ratio: int,
    delete_equiv_src: bool,
    delete_equiv_tgt: bool,
):
    src_onto = Ontology.from_new(src_onto_path)
    tgt_onto = Ontology.from_new(tgt_onto_path)
    subs = SubsumptionMappingGenerator(
        src_onto, tgt_onto, equiv_mappings_path, map_ratio, delete_equiv_src, delete_equiv_tgt
    )
    subs.generate_all_is_a_maps()
    subs_maps = pd.DataFrame(subs.src2tgt.subs, columns=["SrcEntity", "TgtEntity"])
    subs_maps["Score"] = [1.0] * len(subs_maps)
    subs_maps.to_csv("./sub_maps.tsv", sep="\t", index=False)
    preserved_iris = dict()
    preserved_iris[src_onto.owl.name] = subs.preserved_src_iris()
    # because we are not fixing the source side
    assert not subs.deleted_equiv_src_iris()
    preserved_iris[tgt_onto.owl.name] = subs.preserved_tgt_iris()
    # print(set(subs_maps["SrcEntity"].apply(lambda x: unfold_iri(x))) - set(preserved_iris[src_onto.owl.name]))
    SavedObj.save_json(preserved_iris, "./preserved_iris.json")
    
    # Test that all subsumption classes are in the preserved classes
    assert not set(subs_maps["SrcEntity"].apply(lambda x: unfold_iri(x))) - set(
        preserved_iris[src_onto.owl.name]
    )
    assert not set(subs_maps["TgtEntity"].apply(lambda x: unfold_iri(x))) - set(
        preserved_iris[tgt_onto.owl.name]
    )
    
    # isolate and reconstruct hierarchy
    preserved_names = [abbr_iri(x) for x in preserved_iris[tgt_onto.owl.name]]
    tgt_onto = preserve_classes(tgt_onto, preserved_names, keep_hierarchy=True, apply_destroy=False)
    tgt_onto.owl.save(f"./{tgt_onto.owl.name}.isolated.owl")

main()
