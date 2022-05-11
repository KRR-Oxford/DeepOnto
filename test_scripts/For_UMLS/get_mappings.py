# get mappings from the SCUIs of two ontologies (each as a file)
# input: SCUIs for onto1, SCUIs for onto2, MRCONSO.RR

from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="get mappings based on two lists of SCUIs, where each list corresponds to an ontology in SNOMEDCT, FMA, and NCI")
parser.add_argument('-o1','--onto1_scui_fn', type=str,
                    help="scui_filename1", default='FMA_SCUI_to_keep_large.csv')
parser.add_argument('-o2','--onto2_scui_fn', type=str,
                    help="scui_filename2", default='NCI_SCUI_to_keep_large.csv')                    
args = parser.parse_args()

onto_prefix1 = args.onto1_scui_fn.split('_SCUI_to_keep_')[0]
onto_prefix2 = args.onto2_scui_fn.split('_SCUI_to_keep_')[0]
onto_fn_postfix1 = args.onto1_scui_fn.split('_SCUI_to_keep_')[1]
onto_fn_postfix2 = args.onto2_scui_fn.split('_SCUI_to_keep_')[1]
#check that the onto_fn_postfixes (or settings) are the same 
assert onto_fn_postfix1 == onto_fn_postfix2

MRCONSO_FILE = '/mnt/datashare/UMLS/MRCONSO.RRF'
MRSTY_FILE = '/mnt/datashare/UMLS/MRSTY.RRF'

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
ontos = {'NCI', 'FMA', 'SNOMEDCT_US'}
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

def read_list_from_filename(fn):
    with open(fn,encoding='utf-8') as f_content:
        lines = f_content.readlines()
    return [line.rstrip() for line in lines] #remove the trailing \n

def get_mappings_with_cui_from_scuis(onto_prefix1,onto_prefix2,list_SCUIs_onto1,list_SCUIs_onto2):
    dict_mappings = {}
    for cui,entities in tqdm(cui_entities.items()):
        entities1 = set()
        entities2 = set()
        # get the set of entities for each onto, all equiv to the cui
        for e in entities:
            if e[0] == onto_prefix1:
                entities1.add(e[1])
            if e[0] == onto_prefix2:
                entities2.add(e[1])
        # loop over the onto mappings for the cui and see if they are in the two lists of scuis
        for e1 in entities1:
            for e2 in entities2:
                #print(e1,e2)
                if (e1 in list_SCUIs_onto1) and (e2 in list_SCUIs_onto2):
                    dict_mappings[(e1,e2, cui, dict_CUI_STY[cui])] = 1                        
    return dict_mappings

def write_mappings(dict_mappings, mapping_file):
    with open(mapping_file, 'w') as out_f:
        for m in dict_mappings.keys():
            out_f.write('%s,%s,%s,%s\n' % (m[0], m[1], m[2], m[3].replace(',','_'))) #replace the ',' in STY to '_' for better csv display

list_SCUIs_onto1 = read_list_from_filename(args.onto1_scui_fn)
list_SCUIs_onto2 = read_list_from_filename(args.onto2_scui_fn)

#print(list_SCUIs_onto1)
#print(list_SCUIs_onto2)

dict_mappings = get_mappings_with_cui_from_scuis(onto_prefix1, onto_prefix2, list_SCUIs_onto1,list_SCUIs_onto2)
print(len(dict_mappings))

mapping_fn = onto_prefix1 + '-' + onto_prefix2 + '_' + onto_fn_postfix1
write_mappings(dict_mappings=dict_mappings, mapping_file=mapping_fn)
