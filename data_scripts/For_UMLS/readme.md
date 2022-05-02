The data scripts to generate mappings from UMLS and select the SCUIs to keep based on semantic types (or STYs).

The main script is `run_scripts_prune_SCUIs_and_maps.py`, it generates the selected SCUIs based on different settings (pre-defined in `get_SCUI_in_onto_from_STYs.py`) and filter the mappings from UMLS (implemented in `get_mappings.py`).

A separate script to get all mappings, regardless of semantic types, is in `umls_mrconso_with_CUI_and_STY.py`, where it also displays the CUI and STY of each mapping (only the last STY is displayed if there are multiple STYs).

`cal_percent_onto_in_UMLS.py` and `check_snomed_classes_orig_vs_owl.py` calculates the percentage of each selected full ontologies covered by UMLS, and the number of classes ub SNOMED CT (the one processed by snomed-owl-toolkit vs. the original one), respectively.