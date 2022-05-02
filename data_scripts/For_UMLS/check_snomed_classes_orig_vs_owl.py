from owlready2 import *
from tqdm import tqdm

#output str content to a file
#input: filename and the content (str)
def output_to_file(file_name,str):
    with open(file_name, 'w') as f_output:
        f_output.write(str)

use_raw_onto = False # whether to get the statistics from the non-preprocessed ontologies.

SNOMEDCT_concept_file_path = '/mnt/datashare/UMLS/sct2_Concept_Full_US1000124_20210901.txt'   
SNOMEDCT_onto_file = '/mnt/datashare/UMLS-processed/ontos/snomed.owl' # the cleaned version as the original one throws some errors when loaded by owlready2

#process SNOMED CT
dict_SCUI_all_orig = {}
with open(SNOMEDCT_concept_file_path,encoding='utf-8') as f_content:
    doc = f_content.readlines()

    for ind, line in tqdm(enumerate(doc)):
        if ind>0:
            data_eles_SNOMED_CT = line.split('\t')
            SCUI_orig = data_eles_SNOMED_CT[0]
            dict_SCUI_all_orig[SCUI_orig] = 1
print('num of %s IDs:' % 'SNOMED CT orig',len(dict_SCUI_all_orig))

onto = get_ontology(SNOMEDCT_onto_file).load()
dict_SCUI_all_owl = {str(class_name)[3:] : 1 for class_name in list(onto.classes())} # remove 'id.' in str(class_name), e.g. id.9995004
print('num of %s IDs:' % 'SNOMED CT owl',len(dict_SCUI_all_owl))

list_new_in_orig = dict_SCUI_all_orig.keys() - dict_SCUI_all_owl.keys()
#print(len(list_new_in_orig))

output_to_file('SNOMEDCT orig IDs.txt','\n'.join(dict_SCUI_all_orig.keys()))
output_to_file('SNOMEDCT owl IDs.txt','\n'.join(dict_SCUI_all_owl.keys()))
output_to_file('SNOMEDCT orig IDs not in owl.txt','\n'.join(list_new_in_orig))
