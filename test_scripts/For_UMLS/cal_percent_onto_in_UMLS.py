# output the SNOMEDCT_US portion in UMLS and check the percentage in the original SNOMEDCT_US
# output the portion of NCIT/FMA in UMLS

from tqdm import tqdm
import sys
from owlready2 import *

# load ontology using owlready2: 
# input: .owl onto file path; prefix_length of the id, e.g. 3 as 'id.' in id.9999954
# output: a dict of its SCUIs
def load_onto(onto_file,prefix_length=3,onto_name='onto'):
    onto = get_ontology(onto_file).load()
    dict_SCUI_onto = {str(class_name)[prefix_length:] : 1 for class_name in list(onto.classes())} 
    print('num of %s IDs:' % onto_name,len(dict_SCUI_onto))
    return dict_SCUI_onto

#output str content to a file
#input: filename and the content (str)
def output_to_file(file_name,str):
    with open(file_name, 'w', encoding="utf-8-sig") as f_output:
        f_output.write(str)

#make sure you point out to the correct file paths before you run
UMLS_file_path = '/mnt/datashare/UMLS/MRCONSO.RRF'
#or
#UMLS_file_path1 = 'MRCONSO.RRF.aa'
#UMLS_file_path2 = 'MRCONSO.RRF.ab'

use_raw_onto = False # whether to get the statistics from the non-preprocessed ontologies.

SNOMEDCT_concept_file_path = '/mnt/datashare/UMLS/sct2_Concept_Full_US1000124_20210901.txt'   
# SNOMEDCT_onto_file = '/mnt/datashare/UMLS-processed/ontos/snomed.owl'
# NCIT_onto_file = '/mnt/datashare/UMLS-processed/ontos/nci.owl'
# FMA_onto_file = '/mnt/datashare/UMLS-processed/ontos/fma.owl'
SNOMEDCT_onto_file = '/mnt/datashare/UMLS-processed/ontos/snomed.owl' # the cleaned version as the original one throws some errors when loaded by owlready2
NCIT_onto_file = '/mnt/datashare/UMLS/raw_data/ncit_21.02d.owl'
FMA_onto_file = '/mnt/datashare/UMLS/raw_data/fma_4.14.0.owl'

list_source_abbrs = ['SNOMEDCT_US','FMA','NCI']
#list_source_abbrs = ['NCI']

for source_abbr in list_source_abbrs:
    dict_SCUI_all = {}
    if source_abbr == 'SNOMEDCT_US':
         #process SNOMED CT
        if use_raw_onto:           
            with open(SNOMEDCT_concept_file_path,encoding='utf-8') as f_content:
                doc = f_content.readlines()

                for ind, line in tqdm(enumerate(doc)):
                    if ind>0:
                        data_eles_SNOMED_CT = line.split('\t')
                        SCUI_orig = data_eles_SNOMED_CT[0]
                        dict_SCUI_all[SCUI_orig] = 1
        else:
            dict_SCUI_all = load_onto(SNOMEDCT_onto_file,prefix_length=3,onto_name=source_abbr)
            #onto = get_ontology(SNOMEDCT_onto_file).load()
            #dict_SCUI_all = {str(class_name)[3:] : 1 for class_name in list(onto.classes())} # remove 'id.' in str(class_name), e.g. id.9995004
        print('num of %s IDs:' % source_abbr,len(dict_SCUI_all))
    elif source_abbr == 'FMA':
        dict_SCUI_all = load_onto(FMA_onto_file,prefix_length=7,onto_name=source_abbr)
        #onto = get_ontology(FMA_onto_file).load()
        #dict_SCUI_all = {str(class_name)[7:] : 1 for class_name in list(onto.classes())} # remove 'xxx.fma' in str(class_name), e.g. xxx.fma9577
        #print('num of %s IDs:' % source_abbr,len(dict_SCUI_all))
    elif source_abbr == 'NCI':
        dict_SCUI_all = load_onto(NCIT_onto_file,prefix_length=12,onto_name=source_abbr)
        #onto = get_ontology(NCIT_onto_file).load()
        #print(list(onto.classes())[0])
        #dict_SCUI_all = {str(class_name)[12:] : 1 for class_name in list(onto.classes())} 
        # prev: remove 'xxx.NCIT_' in str(class_name), e.g. xxx.NCIT_C98910
        # curr: remove 'ncit_21.02d.' in str(class_name), e.g. ncit_21.02d.C1908
        #print('num of %s IDs:' % source_abbr,len(dict_SCUI_all))    
    
    #sys.exit(0)
    #process UMLS
    dict_SCUI = {} # dict of SCUI in UMLS
    #for UMLS_file_path in [UMLS_file_path1,UMLS_file_path2]:
    for UMLS_file_path in [UMLS_file_path]:
        with open(UMLS_file_path,encoding='utf-8') as f_content:
            doc = f_content.readlines()

        for line in tqdm(doc):
            data_eles = line.split('|')
            if len(data_eles) > 11:
                source = data_eles[11]
            else:
                source = ''    

            if source == source_abbr:
                SCUI = data_eles[9]
                CUI = data_eles[0]
                dict_SCUI[SCUI] = CUI # only the last CUI is recorded if there are multiple CUIs matched to the SCUI

    print('num of %s IDs in UMLS:' % source_abbr,len(dict_SCUI))

    #if source_abbr == 'SNOMEDCT_US':
    #check whether all orig SCUIs are in UMLS
    #has_ID_missing = False
    n_missing=0
    for SCUI_orig in dict_SCUI_all.keys():
        if not SCUI_orig in dict_SCUI:
            #print(SCUI_orig, 'not in UMLS')
            n_missing = n_missing + 1
            #has_ID_missing = True

    #if not has_ID_missing:
    n_SCUI_all = len(dict_SCUI_all)
    print('%.2f%%(%d/%d) %s IDs are in UMLS' % (float(n_SCUI_all-n_missing)/n_SCUI_all*100,n_SCUI_all-n_missing,n_SCUI_all,source_abbr))

    list_SCUIs = dict_SCUI.keys()
    list_CUIs = dict_SCUI.values()
    list_SCUIs_CUIs = [SCUI + ',' + CUI for SCUI,CUI in zip(list_SCUIs,list_CUIs)]
    output_to_file('%s_in_UMLS.txt' % source_abbr,'\n'.join(list_SCUIs_CUIs))

    '''
    console output:
    num of SNOMEDCT_US IDs: 356531
    num of SNOMEDCT_US IDs in UMLS: 494082
    100.00%(356531/356531) SNOMEDCT_US IDs are in UMLS
    num of FMA IDs: 104523
    num of FMA IDs in UMLS: 104704
    97.30%(101698/104523) FMA IDs are in UMLS
    num of NCI IDs: 163842
    num of NCI IDs in UMLS: 159270
    97.21%(159270/163842) NCI IDs are in UMLS
    '''