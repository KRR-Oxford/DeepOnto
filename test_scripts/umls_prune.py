import os
import sys

main_dir = os.getcwd().split("DeepOnto")[0] + "DeepOnto/src"
sys.path.append(main_dir)

from deeponto.onto import Ontology
from deeponto.onto.text.text_utils import unfold_iri
from deeponto import SavedObj
from deeponto.onto.prune.pruning import preserve_classes
import pandas as pd
from deeponto.utils import detect_path

import click

fma_name = lambda id: f"fma:fma{id}"
ncit_name = lambda id: f"ncit_largebio:" + id
snomed_name = lambda id: f"snomed:{id}"
naming = {
    "fma": fma_name,
    "ncit": ncit_name,
    "snomed": snomed_name,
}

@click.command()
@click.option("-o", "--onto_path", type=click.Path(exists=True))
@click.option("-s", "--scui_path", type=click.Path(exists=True))
@click.option("-d", "--destroy", type=bool, default=False)
def main(onto_path, scui_path, destroy):
    scuis = list(pd.read_csv(scui_path, header=None)[0])
    onto = Ontology.from_new(onto_path)
    name_method = naming[onto.owl.name]
    valid_ent_names = [name_method(s) for s in scuis]
    if detect_path("./umls_iris.json"):
        saved_iris = SavedObj.load_json("./umls_iris.json")
    else:
        saved_iris = dict()
    saved_iris[onto.owl.name] = [
        unfold_iri(n) for n in valid_ent_names
    ]
    SavedObj.save_json(saved_iris, "./umls_iris.json")
    
    if destroy:
        pruned_onto = preserve_classes(onto, valid_ent_names, keep_hierarchy=True)
        pruned_onto.owl.save(f"./{onto.owl.name}.pruned.owl")
    else:
        pruned_onto = preserve_classes(onto, valid_ent_names, keep_hierarchy=True, apply_destroy=False)
        pruned_onto.owl.save(f"./{onto.owl.name}.kept.owl")
main()