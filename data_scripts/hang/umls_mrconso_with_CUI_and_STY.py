# get mappings based on MRCONSO in UMLS with CUI and STY 
from tqdm import tqdm
from collections import defaultdict, Counter
from owlready2 import *

MRCONSO_FILE = '/mnt/datashare/UMLS/MRCONSO.RRF'
MRSTY_FILE = '/mnt/datashare/UMLS/MRSTY.RRF'

SNOMEDCT_onto_file = '/mnt/datashare/UMLS-processed/ontos/snomed.owl' # the cleaned version as the original one throws some errors when loaded by owlready2
NCIT_onto_file = '/mnt/datashare/UMLS/raw_data/ncit_21.02d.owl'
FMA_onto_file = '/mnt/datashare/UMLS/raw_data/fma_4.14.0.owl'

'''
Header:
'CUI', 'LAT', 'TS', 'LUI', 'STT', 'SUI', 'ISPREF', 'AUI', 'SAUI',
'SCUI', 'SDUI', 'SAB', 'TTY', 'CODE', 'STR', 'SRL', 'SUPPRESS', 'CVF'
'''

# load ontology using owlready2: 
# input: .owl onto file path; prefix_length of the id, e.g. 3 as 'id.' in id.9999954
# output: a dict of its SCUIs
def load_onto(onto_file,prefix_length=3,onto_name='onto'):
    onto = get_ontology(onto_file).load()
    dict_SCUI_onto = {str(class_name)[prefix_length:] : 1 for class_name in list(onto.classes())} 
    print('num of %s IDs:' % onto_name,len(dict_SCUI_onto))
    return dict_SCUI_onto

ontos = ['SNOMEDCT_US', 'FMA', 'NCI']
# load all source ontologies
dict_SCUI_onto_SNOMED = load_onto(SNOMEDCT_onto_file,prefix_length=3,onto_name='SNOMEDCT_US')
dict_SCUI_onto_FMA = load_onto(FMA_onto_file,7,'FMA')
dict_SCUI_onto_NCIT = load_onto(NCIT_onto_file,12,'NCI')

dict_source_to_SCUI_onto = {onto: dict_SCUI for onto, dict_SCUI in zip(ontos, [dict_SCUI_onto_SNOMED,dict_SCUI_onto_FMA,dict_SCUI_onto_NCIT])}

# read MRSTY and save STY of CUI to dict
dict_CUI_STY = {}
with open(MRSTY_FILE,encoding='utf-8') as f_content:
    doc_MRSTY = f_content.readlines()

for line in tqdm(doc_MRSTY):
    data_eles = line.split('|')
    CUI = data_eles[0]
    STY = data_eles[3] + ' (' + data_eles[1] + ')'
    dict_CUI_STY[CUI] = STY

# read MRCONSO and save CUI->[[sab,scui]] dict
cui_entities = dict()
with open(MRCONSO_FILE) as f:
    for line in tqdm(f.readlines()):
        items = line.strip().split('|')
        cui, scui, sab = items[0], items[9], items[11]
        if sab in ontos:
            if cui in cui_entities:
                cui_entities[cui].append([sab, scui])
            else:
                cui_entities[cui] = [[sab, scui]]

print('CUI #: %d' % len(cui_entities.keys()))

def get_mappings_with_cui(onto1, onto2):
    mappings = list()
    dict_SCUI_onto1 = dict_source_to_SCUI_onto[onto1]
    dict_SCUI_onto2 = dict_source_to_SCUI_onto[onto2]
    for cui,entities in tqdm(cui_entities.items()):
        entities1 = set()
        entities2 = set()
        for e in entities:
            if e[0] == onto1:
                entities1.add(e[1])
            if e[0] == onto2:
                entities2.add(e[1])
        for e1 in entities1:
            if e1 in dict_SCUI_onto1: # e1 should also be in the ontology
                for e2 in entities2: 
                    if e2 in dict_SCUI_onto2: # e2 should also be in the ontology
                        mappings.append([e1, e2, cui, dict_CUI_STY[cui]])
    return mappings

FMA_SNOMED_mappings = get_mappings_with_cui(onto1='FMA', onto2='SNOMEDCT_US')
FMA_NCI_mappings = get_mappings_with_cui(onto1='FMA', onto2='NCI')
SNOMED_NCI_mappings = get_mappings_with_cui(onto1='SNOMEDCT_US', onto2='NCI')

# generate the distribution of STY from mappings (as a list of 4-element lists, e1, e2, cui, sty, see above)
def generate_dist_STY(mappings):
    dict_onto_STY_freq = defaultdict(int)
    for m in mappings:
        dict_onto_STY_freq[m[3]] += 1
    #rank
    cntr_dict_onto_STY_freq = Counter(dict_onto_STY_freq)
    list_STY_freq_precent = []
    for STY,freq in cntr_dict_onto_STY_freq.most_common():
        percent = float(freq)/len(mappings)
        list_STY_freq_precent.append(STY.replace(',','_') + ',' + str(freq) + ',' + '%.2f%%' % (percent*100))
    return list_STY_freq_precent

FMA_SNOMED_mappings_str = '\n'.join(generate_dist_STY(FMA_SNOMED_mappings))
FMA_NCI_mappings_str = '\n'.join(generate_dist_STY(FMA_NCI_mappings))
SNOMED_NCI_mappings_str = '\n'.join(generate_dist_STY(SNOMED_NCI_mappings))
# print('\n'.join(generate_dist_STY(FMA_SNOMED_mappings)))
# print('\n'.join(generate_dist_STY(FMA_NCI_mappings)))
# print('\n'.join(generate_dist_STY(SNOMED_NCI_mappings)))

#output str content to a file
#input: filename and the content (str)
def output_to_file(file_name,str):
    with open(file_name, 'w') as f_output:
        f_output.write(str)

output_to_file("SNOMEDCT_US-FMA_STY_dist.csv",FMA_SNOMED_mappings_str)
output_to_file("FMA-NCI_STY_dist.csv",FMA_NCI_mappings_str)
output_to_file("SNOMEDCT_US-NCI_STY_dist.csv",SNOMED_NCI_mappings_str)

def write_mappings(mappings, mapping_file):
    with open(mapping_file, 'w') as out_f:
        for m in mappings:
            out_f.write('%s,%s,%s,%s\n' % (m[0], m[1], m[2], m[3].replace(',','_'))) #replace the ',' in STY to '_' for better csv display


write_mappings(mappings=FMA_SNOMED_mappings, mapping_file='fma_snomed_with_STY.csv')
write_mappings(mappings=FMA_NCI_mappings, mapping_file='fma_nci_with_STY.csv')
write_mappings(mappings=SNOMED_NCI_mappings, mapping_file='snomed_nci_with_STY.csv')