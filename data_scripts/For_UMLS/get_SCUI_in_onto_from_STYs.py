# get the list of SCUIs from an ontology based on its STY

from owlready2 import *
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="get SCUIs in the ontologies, SNOMEDCT, FMA, and NCI, from a selected set of STYs")
parser.add_argument('-s','--setting', type=str,
                    help="settings of list of STYs to select SCUIs in the ontologies", default='large')
args = parser.parse_args()

setting = args.setting #see the list below, for example, 'mini-SN-pharm-neoplas' #'mini-pharm-neoplas-body' # 'mini-disease-body' # or 'large'

if setting == 'large': #p(STY | STY of mapping in any two ontologies) >= 10%
    list_STYs_SNOMED = ['Disease or Syndrome (T047)','Pharmacologic Substance (T121)','Neoplastic Process (T191)','Body Part, Organ, or Organ Component (T023)']
    list_STYs_NCI = ['Disease or Syndrome (T047)','Pharmacologic Substance (T121)','Neoplastic Process (T191)','Body Part, Organ, or Organ Component (T023)']
    list_STYs_FMA = ['Body Part, Organ, or Organ Component (T023)']
elif setting == 'mini-disease-body': # p(STY | STY of mapping in any two ontologies) >= 10%, disease and body only 
    list_STYs_SNOMED = ['Disease or Syndrome (T047)','Body Part, Organ, or Organ Component (T023)']
    list_STYs_NCI = ['Disease or Syndrome (T047)','Body Part, Organ, or Organ Component (T023)']
    list_STYs_FMA = ['Body Part, Organ, or Organ Component (T023)']
elif setting == 'mini-pharm-neoplas-body': # p(STY | STY of mapping in any two ontologies) >= 10%, pharmacologic substance, neoplastic process, and body only 
    list_STYs_SNOMED = ['Pharmacologic Substance (T121)','Neoplastic Process (T191)','Body Part, Organ, or Organ Component (T023)']
    list_STYs_NCI = ['Pharmacologic Substance (T121)','Neoplastic Process (T191)','Body Part, Organ, or Organ Component (T023)']
    list_STYs_FMA = ['Body Part, Organ, or Organ Component (T023)']
elif setting == 'mini-body': # p(STY | STY of mapping in any two ontologies) >= 10%, body only 
    list_STYs_SNOMED = ['Body Part, Organ, or Organ Component (T023)']
    list_STYs_NCI = ['Body Part, Organ, or Organ Component (T023)']
    list_STYs_FMA = ['Body Part, Organ, or Organ Component (T023)']
elif setting == 'mini-SN': # p(STY | STY of mapping in any two ontologies) >= 10%, SNOMED-to-NCI only
    list_STYs_SNOMED = ['Disease or Syndrome (T047)','Pharmacologic Substance (T121)','Neoplastic Process (T191)']
    list_STYs_NCI = ['Disease or Syndrome (T047)','Pharmacologic Substance (T121)','Neoplastic Process (T191)']
    list_STYs_FMA = []
elif setting == 'mini-SF-NF': # p(STY | STY of mapping in any two ontologies) >= 10%, SNOMED-to-FMA and NCI-to-FMA only
    list_STYs_SNOMED = ['Body Part, Organ, or Organ Component (T023)']
    list_STYs_NCI = ['Body Part, Organ, or Organ Component (T023)']
    list_STYs_FMA = ['Body Part, Organ, or Organ Component (T023)']
elif setting == 'mini-SN-disease': # p(STY | STY of mapping in any two ontologies) >= 10%, SNOMED-to-NCI, disease only
    list_STYs_SNOMED = ['Disease or Syndrome (T047)']
    list_STYs_NCI = ['Disease or Syndrome (T047)']
    list_STYs_FMA = []
elif setting == 'mini-SN-pharm-neoplas': # p(STY | STY of mapping in any two ontologies) >= 10%, SNOMED-to-NCI, pharmacologic substance and neoplastic process only
    list_STYs_SNOMED = ['Pharmacologic Substance (T121)','Neoplastic Process (T191)']
    list_STYs_NCI = ['Pharmacologic Substance (T121)','Neoplastic Process (T191)']
    list_STYs_FMA = []
elif setting == 'mini-SN-pharm': # p(STY | STY of mapping in any two ontologies) >= 10%, SNOMED-to-NCI, pharmacologic substance only
    list_STYs_SNOMED = ['Pharmacologic Substance (T121)']
    list_STYs_NCI = ['Pharmacologic Substance (T121)']
    list_STYs_FMA = []
elif setting == 'mini-SN-neoplas': # p(STY | STY of mapping in any two ontologies) >= 10%, SNOMED-to-NCI, neoplastic process only
    list_STYs_SNOMED = ['Neoplastic Process (T191)']
    list_STYs_NCI = ['Neoplastic Process (T191)']
    list_STYs_FMA = []

SNOMEDCT_onto_file = '/mnt/datashare/UMLS-processed/ontos/snomed.owl'
NCIT_onto_file = '/mnt/datashare/UMLS/raw_data/ncit_21.02d.owl'
FMA_onto_file = '/mnt/datashare/UMLS/raw_data/fma_4.14.0.owl'
# SNOMEDCT_onto_file = '/mnt/datashare/UMLS-processed/ontos/snomed.owl'
# NCIT_onto_file = '/mnt/datashare/UMLS-processed/ontos/nci.owl'
# FMA_onto_file = '/mnt/datashare/UMLS-processed/ontos/fma.owl'

UMLS_file_path = '/mnt/datashare/UMLS/MRCONSO.RRF'
UMLS_STY_file_path = '/mnt/datashare/UMLS/MRSTY.RRF'

list_source_abbrs = ['SNOMEDCT_US','FMA','NCI']
dict_sab_list_STY_onto = {source_abbr: lst_STY_onto for source_abbr, lst_STY_onto in zip(list_source_abbrs, [list_STYs_SNOMED,list_STYs_FMA,list_STYs_NCI])}

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
    with open(file_name, 'w') as f_output:
        f_output.write(str)

def get_SCUI_from_list_STYs(dict_SCUI_to_list_STYs,list_STYs_onto):
    list_SCUIs = []
    for STY in tqdm(list_STYs_onto):
        for SCUI,list_STYs in dict_SCUI_to_list_STYs.items(): 
            if STY in list_STYs:
                list_SCUIs.append(SCUI)
    return list_SCUIs                            

# get dict of SCUI to list of distinct STYs from MRCONSO and MRSTY
# read MRSTY and save STY of CUI to dict
dict_CUI_STY = {}
with open(UMLS_STY_file_path,encoding='utf-8') as f_content:
    doc_MRSTY = f_content.readlines()

for line in tqdm(doc_MRSTY):
    data_eles = line.split('|')
    CUI = data_eles[0]
    STY = data_eles[3] + ' (' + data_eles[1] + ')'
    dict_CUI_STY[CUI] = STY

with open(UMLS_file_path,encoding='utf-8') as f_content:
    doc_MRRCONSO = f_content.readlines()

for source_abbr in list_source_abbrs:
    # get the processed ontology
    if source_abbr == 'SNOMEDCT_US':
        dict_SCUI_onto = load_onto(SNOMEDCT_onto_file,prefix_length=3,onto_name=source_abbr)
    elif source_abbr == 'FMA':
        dict_SCUI_onto = load_onto(FMA_onto_file,prefix_length=7,onto_name=source_abbr)
    elif source_abbr == 'NCI':
        dict_SCUI_onto = load_onto(NCIT_onto_file,prefix_length=12,onto_name=source_abbr)

    dict_SCUI_to_list_STYs={} # get dict of SCUI to list of distinct STYs
    for line in tqdm(doc_MRRCONSO):
        data_eles = line.split('|')
        SCUI = data_eles[9]
        # do not count the SCUI when it is not in the (processed) source ontology
        if not SCUI in dict_SCUI_onto:
            continue
        CUI = data_eles[0]
        # get all STYs for the SCUI (not just the first occurrence)
        STY = dict_CUI_STY[CUI]
        if not SCUI in dict_SCUI_to_list_STYs:
            dict_SCUI_to_list_STYs[SCUI] = [STY]
        else:
            list_STYs_from_SCUI = dict_SCUI_to_list_STYs[SCUI]
            if not STY in list_STYs_from_SCUI:
                list_STYs_from_SCUI.append(STY)
            dict_SCUI_to_list_STYs[SCUI] = list_STYs_from_SCUI

    # get the list of SCUIs from the selected STYs
    list_STYs_onto = dict_sab_list_STY_onto[source_abbr]
    list_SCUIs = get_SCUI_from_list_STYs(dict_SCUI_to_list_STYs,list_STYs_onto)

    output_to_file('%s_SCUI_to_keep_%s.csv' % (source_abbr,setting),'\n'.join(list_SCUIs))