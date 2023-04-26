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
- [ ] **Add** the ontology-to-graph builder at `deeponto.onto.graph_builder`.

### Changed

- [X] **Change** `get_owl_object_annotations` by adding `uniqify` at the end to preserve the order.

### Fixed

- [X] **Fix** BERTMap's non-synonym sampling when the class labels are not available using the try-catch block.

## v0.7.0 (2023 April)

### Added

- [X] **Add** the BERTSubs module at `deeponto.subs.bertsubs`; its inter-ontology setting is also imported at `deeponto.align.bertsubs`.

### Changed

- [X] **Move** the pruning functionality into `deeponto.onto.OntologyPruner` as a separate module.
- [X] **Amend** JVM checking before displaying the JVM memory prompt from importing `deeponto.onto.Ontology`; if started already, skip this step.
- [X] **Change** the function `get_owl_object_annotations` at `deeponto.onto.Ontology` by preserving the relative order of annotation retrieval, i.e., create `set` first and use the `.add()` function instead of casting the `list` into `set` in the end.

### Fixed

- [X] **Fix** the function `check_deprecated` at `deeponto.onto.Ontology` by adding a check for the $\texttt{owl:deprecated}$ annotation property -- if this property does not exist in the current ontology, return `False` (not deprecated).
 

## v0.6.1 (2023 April)
### Added

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

## v0.6.0 (2023 Mar)

- [X] **Add** the OntoLAMA module at `deeponto.lama`.
- [X] **Add** the verb auto-correction and more precise documentation for `deeponto.onto.verbalisation`.

## v0.5.x (2023 Jan - Feb)

- [X] **Add** the preliminary ontology verbalisation module at `deeponto.onto.verbalisation`.
- [X] **Fix** PyPI issues based on the new code layout.
- [X] **Change** code layout to the `src/` layout.
- [X] **Rebuild** the whole package based on the OWLAPI.
- [X] **Remove** owlready2 from the essential dependencies.

## Deprecated (before 2023)

The code before v0.5.0 is no longer available.