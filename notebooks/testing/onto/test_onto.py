# append the paths
import os
import sys

main_dir = os.getcwd().split("OntoPLM")[0] + "OntoPLM/src"
sys.path.append(main_dir)
import shutil
import pytest

from ontoplm.onto import Ontology
from ontoplm.onto.onto_text import *

# testing data with fma2nci.small ontology file
test_ontos = [
    main_dir + "/../data/LargeBio/ontos/fma2nci.small.owl",
    main_dir + "/../data/LargeBio/ontos/nci2fma.small.owl",
    main_dir + "/../data/LargeBio/ontos/fma2snomed.small.owl",
    main_dir + "/../data/LargeBio/ontos/snomed2fma.small.owl",
    main_dir + "/../data/LargeBio/ontos/snomed2nci.small.owl",
    main_dir + "/../data/LargeBio/ontos/nci2snomed.small.owl",
]


@pytest.mark.parametrize("onto_file", test_ontos)
def test_main(onto_file):
    tkz = Tokenizer.from_pretrained(BIOCLINICAL_BERT)
    onto = Ontology.from_new(onto_file, tokenizer=tkz)
    assert onto.num_classes == len(list(onto.owl.classes()))
    assert onto.num_classes == len(onto.class2idx)
    assert onto.num_classes == len(onto.idx2class)
    assert onto.num_classes == len(onto.class2labs)
    onto.save_instance("./test_onto_temp")
    onto2 = Ontology.from_saved("./test_onto_temp")
    assert onto.__dict__.keys() == onto2.__dict__.keys()
    assert list(onto.__dict__.values()) == list(onto2.__dict__.values())
    for k in onto.__dict__.keys():
        assert onto.__dict__[k] == onto2.__dict__[k], "should be all the same"
    shutil.rmtree("./test_onto_temp")
