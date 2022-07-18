from subprocess import Popen, STDOUT
from typing import List, Tuple
import itertools

onto_prefixes = ['SNOMEDCT_US','FMA','NCI']
middlefix = 'SCUI_to_keep'
#onto_postfixes = ['large','mini-body','mini-disease-body','mini-pharm-neoplas-body','mini-SF-NF','mini-SN-disease','mini-SN-pharm-neoplas','mini-SN-pharm','mini-SN-neoplas','mini-SN'] # full settings
onto_postfixes = ['large','mini-body','mini-SN-disease','mini-SN-pharm','mini-SN-neoplas'] # selected settings

run_part1_get_SCUI = True
run_part2_get_mappings = True

if run_part1_get_SCUI:
    # part 1: run get_SCUI_in_onto_from_STYs.py to get SCUIs to keep
    for onto_postfix in onto_postfixes:
        command = 'python get_SCUI_in_onto_from_STYs.py -s %s' % onto_postfix
        print('running ', command)
        p = Popen(command, shell=True, stderr=STDOUT)
        p.wait()

        if 0 != p.returncode:
            print('Command %s wrong!' % command)
            continue

if run_part2_get_mappings:    
# part 2: run get_mappings.py to get the mappings with the counts for each setting

    # get ordered pairwise combinations
    def unique_combinations(elements: List[str]) -> List[Tuple[str, str]]:
        """
        from https://codereview.stackexchange.com/a/256954/257768
        Precondition: `elements` does not contain duplicates.
        Postcondition: Returns unique combinations of length 2 from `elements`.

        >>> unique_combinations(["apple", "orange", "banana"])
        [("apple", "orange"), ("apple", "banana"), ("orange", "banana")]
        """
        return list(itertools.combinations(elements, 2))

    for onto_prefix1, onto_prefix2 in unique_combinations(onto_prefixes):
        for onto_postfix in onto_postfixes:
            onto1_scui_fn = '_'.join([onto_prefix1,middlefix,onto_postfix]) + '.csv'
            onto2_scui_fn = '_'.join([onto_prefix2,middlefix,onto_postfix]) + '.csv'       

            command = 'python get_mappings.py -o1 %s -o2 %s' % (onto1_scui_fn,onto2_scui_fn)
            print('running ', command)
            p = Popen(command, shell=True, stderr=STDOUT)
            p.wait()

            if 0 != p.returncode:
                print('Command %s wrong!' % command)
                continue
            #else:
            #    print('Command %s completed successfully!' % command)
            #Popen(command)