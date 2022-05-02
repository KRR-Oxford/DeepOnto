Besides model development and implementation, DeepOnto also aims to contribute new Ontology Matching (OM) resources and comprehensive evaluation workaround. In this page, provide download links to our datasets and instructions of data usage.

## LargeBioMedData 

The `LargeBioMedData` provides five ontology pairs for both equivalence and subsumption ontology matching.

> Download link: https://doi.org/10.5281/zenodo.6510087.

> See detailed resource construction and evaluation methods in our paper: link-to-be-updated

Statistics for the equivalence matching set-ups.

| Source | Task        | Category | #Classes      | #RefMaps (equiv) | #Annot.  | AvgDepths |
|--------|-------------|----------|---------------|------------------|----------|-----------|
| Mondo  | OMIM-ORDO   | Disease  | 9,642-8838    | 3,721            | 34K-34K  | 1.44-1.63 |
| Mondo  | NCIT-DOID   | Disease  | 6,835-8,848   | 4,684            | 80K-38K  | 2.04-6.85 |
| UMLS   | SNOMED-FMA  | Body     | 24,182-64,726 | 7,256            | 39K-711K | 1.86-9.32 |
| UMLS   | SNOMED-NCIT | Pharm    | 16,045-15,250 | 5,803            | 19K-220K | 1.09-3.26 |
| UMLS   | SNOMED-NCIT | Neoplas  | 11,271-13,956 | 3,804            | 23K-182K | 1.15-1.68 |

Statistics for the subsumption matching set-ups. Note that each subsumption matching task is constructed from an equivalence matching task subject to target side class deletion.


| Source | Task        | Category | #Classes      | #RefMaps (subs)  |
|--------|-------------|----------|---------------|------------------|
| Mondo  | OMIM-ORDO   | Disease  | 9,642-8,735   | 103              | 
| Mondo  | NCIT-DOID   | Disease  | 6,835-5,113   | 3,339            | 
| UMLS   | SNOMED-FMA  | Body     | 24,182-59,567 | 5,506            | 
| UMLS   | SNOMED-NCIT | Pharm    | 16,045-12,462 | 4,225            | 
| UMLS   | SNOMED-NCIT | Neoplas  | 11,271-13,790 | 213              | 

The downloaded datasets include `Mondo.zip` and `UMLS.zip` for resources constructed from Mondo and UMLS, respectively.
Each `.zip` file has three folders: `raw_data`, `equiv_match`, and `subs_match`, corresponding to the raw source ontologies, data for equivalence matching, and data for subsumption matching, respectively. Detailed structure is presented in the following figure. Note for candidate mappings, we generate `for_eval` and `for_score` folders for evaluation and mapping scoring purposes; for users who wish to directly use the generated candidates for each reference mapping without depending on DeepOnto, they can use the file at: `.../for_eval/src2tgt.anchored.maps.json`.

<p align="center">
  <a href="https://krr-oxford.github.io/DeepOnto/">
    <img alt="deeponto" src="https://raw.githubusercontent.com/KRR-Oxford/DeepOnto/main/docs/images/largebiomeddata.svg">
  </a>
</p>



## Appendix: Abbreviations of URIs/IRIs

For readability, the following URIs/IRIs are abbreviated in the reference and candidate mappings included in the OM resources in accordance with other implementations in DeepOnto (see `src/deeponto/onto/iris.py`).

```json
{
    "http://bioontology.org/projects/ontologies/fma/fmaOwlDlComponent_2_0#": "fma_largebio:",
    "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#": "ncit_largebio:",
    "http://www.ihtsdo.org/snomed#": "snomed_largebio:",
    "http://purl.org/sig/ont/fma/": "fma:",
    "http://snomed.info/id/": "snomed:",
    "http://linkedlifedata.com/resource/umls/id/": "umls:",
    "http://identifiers.org/hgnc/": "hgnc:",
    "http://identifiers.org/mesh/": "mesh:",
    "http://identifiers.org/snomedct/": "snomedct:",
    "http://purl.obolibrary.org/obo/": "obo:",
    "http://www.orpha.net/ORDO/": "ordo:",
    "http://www.ebi.ac.uk/efo/": "efo:",
    "http://omim.org/entry/": "omim:",
    "http://www.omim.org/phenotypicSeries/": "omimps:",
    "http://identifiers.org/meddra/": "meddra:",
    "http://identifiers.org/medgen/": "medgen:"
}
```
