import os
import sys

main_dir = os.getcwd().split("DeepOnto")[0] + "DeepOnto/src"
sys.path.append(main_dir)

from deeponto.onto import Ontology
from deeponto.onto.text.text_utils import unfold_iri
from deeponto import SavedObj
from deeponto.onto.graph.pruning import preserve_classes
import pandas as pd

from collections import defaultdict
from itertools import chain

import click

def to_dict(mondo_maps_df):
    dic = defaultdict(list)
    for _, dp in mondo_maps_df.iterrows():
        dic[dp["subject_id"]].append(dp["object_id"])
    return dic

@click.command()
@click.option("-o", "--onto_path", type=click.Path(exists=True))
@click.option("-d", "--destroy", type=bool, default=False)
def main(onto_path, destroy):

    data_dir = "/home/yuan/projects/DeepOnto/data/Mondo"
    omim_entry_maps = pd.read_csv(f"{data_dir}/mappings/mondo_exactmatch_omim.sssom.tsv", sep="\t")
    omim_ps_maps = pd.read_csv(f"{data_dir}/mappings/mondo_exactmatch_omimps.sssom.tsv", sep="\t")
    ordo_maps = pd.read_csv(f"{data_dir}/mappings/mondo_exactmatch_orphanet.sssom.tsv", sep="\t")
    doid_maps = pd.read_csv(f"{data_dir}/mappings/mondo_exactmatch_doid.sssom.tsv", sep="\t")
    ncit_maps = pd.read_csv(f"{data_dir}/mappings/mondo_exactmatch_ncit.sssom.tsv", sep="\t")

    ordo_dict = to_dict(ordo_maps)
    omim_entry_dict = to_dict(omim_entry_maps)
    omim_ps_dict = to_dict(omim_ps_maps)
    doid_dict = to_dict(doid_maps)
    ncit_dict = to_dict(ncit_maps)

    mondo_verified = lambda x: list(chain.from_iterable(x.values()))
    valid_omim_ents = ["omim:" + x.split(":")[-1] for x in mondo_verified(omim_entry_dict)]
    valid_omim_ents += ["omimps:PS" + x.split(":")[-1] for x in mondo_verified(omim_ps_dict)]
    valid_ordo_ents = ["ordo:Orphanet_" + x.split(":")[-1] for x in mondo_verified(ordo_dict)]
    valid_doid_ents = ["obo:DOID_" + x.split(":")[-1] for x in mondo_verified(doid_dict)]
    valid_ncit_ents = ["nci:" + x.split(":")[-1] for x in mondo_verified(ncit_dict)]
    
    dic = {
        "omim": valid_omim_ents,
        "ordo": valid_ordo_ents,
        "doid": valid_doid_ents,
        "ncit": valid_ncit_ents,
    }
    
    saved_dic = {}
    for k, v in dic.items():
        v = [unfold_iri(ent) for ent in v]
        saved_dic[k] = v
    SavedObj.save_json(saved_dic, "mondo_iris.json") 
    
    props = ["label", "hasExactSynonym", "exactMatch", "alternative_term", "symbol", "P90"]
    onto = Ontology.from_new(onto_path, lab_props=props)

    
    if destroy:
        pruned_onto = preserve_classes(onto, dic[onto.owl.name], keep_hierarchy=True)
        pruned_onto.owl.save(f"./{onto.owl.name}.pruned.owl")
    else:
        pruned_onto = preserve_classes(onto, dic[onto.owl.name], keep_hierarchy=True, apply_destroy=False)
        pruned_onto.owl.save(f"./{onto.owl.name}.kept.owl")

main()