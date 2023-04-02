# Changelog :newspaper:

<!-- Added for new features.
Changed for changes in existing functionality.
Deprecated for soon-to-be removed features.
Removed for now removed features.
Fixed for any bug fixes.
Security in case of vulnerabilities. -->


## Unreleased

### Added

- [ ] **Add** the ontology normaliser at `deeponto.onto.normaliser`.
- [ ] **Add** the BERTSubs module at `deeponto.subs`.
- [ ] **Add** the [detailed instructions](../verbaliser) for how to use the ontology verbaliser. 
- [X] **Add** the method `remove_axiom` for removing an axiom from the ontology at `deeponto.onto.Ontology` (note that the counterpart `add_axiom` has already been available).
- [X] **Add** the method `check_named_entity` for checking if an entity is named at `deeponto.onto.Ontology`.
- [X] **Add** the method `get_subsumption_axioms` for getting subsumption axioms subject to different entity types at `deeponto.onto.Ontology`.
- [X] **Add** the method `get_asserted_complex_classes` for getting all complex classes that occur in ontology (subsumption and/or equivalence) axioms at `deeponto.onto.Ontology`.
- [X] **Add** the methods `get_asserted_parents` and `get_asserted_children` for getting asserted parent and children for a given entity at `deeponto.onto.Ontology`.
- [X] **Add** the method `check_deprecation` for checking an owl object's deprecation (annotated) at `deeponto.onto.Ontology`.

### Changed

- [X] **Move** the spacy `en_core_web_sm` download into the initialisation of `OntologyVerbaliser`.
- [X] **Change** the method of getting equivalence axioms by adding support to different entity types at `deeponto.onto.Ontology`.
- [X] **Rename** the methods of getting inferred super-entities and sub-entities at `deeponto.onto.OntologyReasoner`:
    -  `super_entities_of` $\rightarrow$ `get_inferred_super_entities` 
    -  `sub_entities_of` $\rightarrow$ `get_inferred_sub_entities`

### Fixed

- [X] **Fix** the top and bottom data property iris (from "https:" to "http:") at `deeponto.onto.Ontology`.

## Most Recent 

**v0.6.0 (2023 Mar)**

- [X] **Add** the OntoLAMA module at `deeponto.lama`.
- [X] **Add** the verb auto-correction and more precise documentation for `deeponto.onto.verbalisation`.

## Previous Versions

**v0.5.x (2023 Jan - Feb)**

- [X] **Add** the preliminary ontology verbalisation module at `deeponto.onto.verbalisation`.
- [X] **Fix** PyPI issues based on the new code layout.
- [X] **Change** code layout to the `src/` layout.
- [X] **Rebuild** the whole package based on the OWLAPI.
- [X] **Remove** owlready2 from the essential dependencies.

!!! warning

    The deprecated code (before v0.5.0) is available at the [legacy branch](https://github.com/KRR-Oxford/DeepOnto/tree/legacy).