import os
import sys

main_dir = os.getcwd().split("DeepOnto")[0] + "DeepOnto/src"
sys.path.append(main_dir)

from deeponto.utils.oaei_utils import *
from deeponto.onto.mapping import *

import click

@click.command()
@click.option("-r", "--result_path", required=True, type=str)
def main(result_path):
    ref, ignored = read_oaei_mappings(f"{result_path}/systemAlignment.rdf")
    om = OntoMappings(flag="src2tgt", rel="=", n_best=None)
    for s, t, m in ref:
        om.add(EntityMapping(s, t, "=", float(m)))
    om.save_instance(f"{result_path}/../src2tgt")
    
main()
