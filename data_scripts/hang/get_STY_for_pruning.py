# input a source ontology and UMLS (MRCONSO.RR and MRSTY.RR)
# output: 
# (i) STY distribution of the CUIs for the SCUIs in the source ontology
# (ii) updated source ontology with pruning by class based on the STYs 
#    - to do, so far I just output all the pruned SCUIs.

from owlready2 import *
from collections import defaultdict,Counter
from tqdm import tqdm

#output str content to a file
#input: filename and the content (str)
def output_to_file(file_name,str):
    with open(file_name, 'w') as f_output:
        f_output.write(str)

#ontology files and loading
#onto_file = '/mnt/datashare/UMLS/SnomedCT_USEditionRF2_PRODUCTION_20210901T120000Z.owl'
SNOMEDCT_onto_file = '/mnt/datashare/UMLS-processed/ontos/snomed.owl'
NCIT_onto_file = '/mnt/datashare/UMLS-processed/ontos/nci.owl'
FMA_onto_file = '/mnt/datashare/UMLS-processed/ontos/fma.owl'

UMLS_file_path = '/mnt/datashare/UMLS/MRCONSO.RRF'
UMLS_STY_file_path = '/mnt/datashare/UMLS/MRSTY.RRF'

list_source_abbrs = ['SNOMEDCT_US','FMA','NCI']

# read MRSTY and save STY of CUI to dict
dict_CUI_STY = {}
with open(UMLS_STY_file_path,encoding='utf-8') as f_content:
    doc = f_content.readlines()

for line in tqdm(doc):
    data_eles = line.split('|')
    CUI = data_eles[0]
    STY = data_eles[3] + ' (' + data_eles[1] + ')'
    dict_CUI_STY[CUI] = STY

# read and loop over MRCONSO, to get:
#   (i) the distribution of STYs for the source ontology in UMLS
#   (ii) the list of SCUIs to be removed in the source ontology, based on 
#            the STY filtering rule - keeping any STYs representing over 1% of all SCUIs

with open(UMLS_file_path,encoding='utf-8') as f_content:
    doc = f_content.readlines()

for source_abbr in list_source_abbrs:
    # get the processed ontology
    if source_abbr == 'SNOMEDCT_US':
        onto = get_ontology(SNOMEDCT_onto_file).load()
        dict_SCUI_onto = {str(class_name)[3:] : 1 for class_name in list(onto.classes())} # remove 'id.' in str(class_name), e.g. id.9995004
        print('num of %s IDs:' % source_abbr,len(dict_SCUI_onto))
    elif source_abbr == 'FMA':
        onto = get_ontology(FMA_onto_file).load()
        dict_SCUI_onto = {str(class_name)[7:] : 1 for class_name in list(onto.classes())} # remove 'xxx.fma' in str(class_name), e.g. xxx.fma9577
        print('num of %s IDs:' % source_abbr,len(dict_SCUI_onto))
    elif source_abbr == 'NCI':
        onto = get_ontology(NCIT_onto_file).load()
        dict_SCUI_onto = {str(class_name)[9:] : 1 for class_name in list(onto.classes())} # remove 'xxx.NCIT_' in str(class_name), e.g. xxx.NCIT_C98910
        print('num of %s IDs:' % source_abbr,len(dict_SCUI_onto))    

    dict_onto_STY_freq = defaultdict(int) # dict to capture the freq of STY of all SCUI in the source ontology
    dict_SCUI_STY = {} # dict of SCUI to STY (the first occurrence only) in UMLS
    n_SCUI = 0
    for line in tqdm(doc):
        data_eles = line.split('|')
        if len(data_eles) > 11:
            source = data_eles[11]
        else:
            source = ''    

        if source == source_abbr:
            SCUI = data_eles[9]
            # do not count the SCUI when it is not in the (processed) source ontology
            if not SCUI in dict_SCUI_onto:
                continue
            CUI = data_eles[0]
            # only count the first SCUI occurrence to get its STY
            if SCUI in dict_SCUI_STY:
                continue
            STY = dict_CUI_STY[CUI]
            dict_SCUI_STY[SCUI] = STY
            dict_onto_STY_freq[STY] += 1
            n_SCUI += 1

    #re-rank and render freq to freq (percentage) in the dict        
    #also get the list of STY to keep based on 1% representation in all SCUIs in the (processed) ontology
    print('output STY and freq for %s' % source_abbr)
    #print(source_abbr, dict_onto_STY_freq)
    cntr_dict_onto_STY_freq = Counter(dict_onto_STY_freq)
    #print(cntr_dict_onto_STY_freq.most_common())
    list_STY_freq_precent = []
    list_STY_to_keep = []
    list_STY_to_keep_freq_prec = []
    for STY,freq in cntr_dict_onto_STY_freq.most_common():
        #print(STY + ',' + str(freq) + ',' + '%.2f%%' % (float(freq)/n_SCUI)*100)
        percent = float(freq)/n_SCUI
        list_STY_freq_precent.append(STY.replace(',','_') + ',' + str(freq) + ',' + '%.2f%%' % (percent*100))
        if percent >= 0.01:
            list_STY_to_keep.append(STY)
            list_STY_to_keep_freq_prec.append(STY.replace(',','_') + ',' + str(freq) + ',' + '%.2f%%' % (percent*100))

    # to note that only the first SCUI occurred in MRCONSO was considered for each SCUI.
    # output the STY distribution
    output_to_file('%s_STY_dist_first_SCUI_occ.csv' % source_abbr,'\n'.join(list_STY_freq_precent))
    # output the STY to be kept
    output_to_file('%s_STY_kept_first_SCUI_occ.csv' % source_abbr,'\n'.join(list_STY_to_keep_freq_prec))
    
    list_SCUI_to_keep = []
    for STY_to_keep in tqdm(list_STY_to_keep):
        for SCUI, STY in dict_SCUI_STY.items():
            if STY_to_keep == STY:
                list_SCUI_to_keep.append(SCUI)
                
    # output the SCUI pruned based on the list of STY to be kept
    output_to_file('%s_SCUI_pruned_by_STY.csv' % source_abbr,'\n'.join(list_SCUI_to_keep))