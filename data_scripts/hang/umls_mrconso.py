
MRCONSO_FILE = '/home/jiaoyan/UMLS/MRCONSO.RRF'

'''
Header:
'CUI', 'LAT', 'TS', 'LUI', 'STT', 'SUI', 'ISPREF', 'AUI', 'SAUI',
'SCUI', 'SDUI', 'SAB', 'TTY', 'CODE', 'STR', 'SRL', 'SUPPRESS', 'CVF'
'''

cui_entities = dict()
ontos = {'NCI', 'FMA', 'SNOMEDCT_US'}
with open(MRCONSO_FILE) as f:
    for line in f.readlines():
        items = line.strip().split('|')
        cui, scui, sab = items[0], items[9], items[11]
        if sab in ontos:
            if cui in cui_entities:
                cui_entities[cui].append([sab, scui])
            else:
                cui_entities[cui] = [[sab, scui]]

print('CUI #: %d' % len(cui_entities.keys()))


def get_mappings(onto1, onto2):
    mappings = list()
    for entities in cui_entities.values():
        entities1 = set()
        entities2 = set()
        for e in entities:
            if e[0] == onto1:
                entities1.add(e[1])
            if e[0] == onto2:
                entities2.add(e[1])
        for e1 in entities1:
            for e2 in entities2:
                mappings.append([e1, e2])
    return mappings


FMA_SNOMED_mappings = get_mappings(onto1='FMA', onto2='SNOMEDCT_US')
FMA_NCI_mappings = get_mappings(onto1='FMA', onto2='NCI')
SNOMED_NCI_mappings = get_mappings(onto1='SNOMEDCT_US', onto2='NCI')


def write_mappings(mappings, mapping_file):
    with open(mapping_file, 'w') as out_f:
        for m in mappings:
            out_f.write('%s,%s\n' % (m[0], m[1]))


write_mappings(mappings=FMA_SNOMED_mappings, mapping_file='/home/jiaoyan/UMLS/fma_snomed.csv')
write_mappings(mappings=FMA_NCI_mappings, mapping_file='/home/jiaoyan/UMLS/fma_nci.csv')
write_mappings(mappings=SNOMED_NCI_mappings, mapping_file='/home/jiaoyan/UMLS/snomed_nci.csv')
