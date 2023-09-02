# Changelog :newspaper:

<!-- Added for new features.
Changed for changes in existing functionality.
Deprecated for soon-to-be removed features.
Removed for now removed features.
Fixed for any bug fixes.
Security in case of vulnerabilities. -->


## Unreleased 


### Added

- [X] **Add** `set_seed()` method at `deeponto.utils`.
### Changed 

- [X] **Change** the layout of all utility methods by making them stand-alone instead of static methods.
- [X] **Change** the `.verbalise_class_expression()` method by adding an option to keep entity IRIs without verbalising them using `.vocabs` at `deeponto.onto.OntologyVerbaliser`.
- [X] **Change** default `apply_lowercasing` value to `False` for both `.get_annotations()` and `.build_annotation_index()` methods at `deeponto.onto.Ontology`.
- [X] **Change** the method `.get_owl_object_annotations()` to `.get_annotations()` at `deeponto.onto.Ontology`.
- [X] **Change** the LogMap debugger memory options for BERTMap's mapping repair.
- [X] **Change** the default jar command timeout to 1 hour.

## v0.8.4 (2023 July)
### Added

- [X] **Add** specific check of `use_in_alignment` annotation in BERTMap for the OAEI.
- [X] **Add** OAEI utilities at `deeponto.align.oaei`.

### Changed

- [X] **Change** the `read_table_mappings` method to allow `None` for threshold.
### Fixed

- [X] **Fix** BERTMap error and add corresponding warning when an input ontology has no sibling class group, related to Issue #10.
- [X] **Fix** BERTMap error and add corresponding warning when an input ontology has some class with no label (annotation), related to Issue #10.

## v0.8.3 (2023 July)

### Changed

- [X] **Change** the mapping extension from using reasoner to direct assertions.
- [X] **Change** the name of pruning function in `deeponto.onto.OntologyPruner`.
- [X] **Change** the verbalisation function by setting quantifier words as optional (by default not adding).
- [X]  **Change** sibing retrieval from using reasoner to direct assertions.

### Fixed

- [X] **Fix** the minor bug for the `f1` and `MRR` method in `deeponto.align.evaluation.AlignmentEvaluator`.

## v0.8.0 (2023 June)

### Added

- [X] **Add** the ontology normaliser at `deeponto.onto.OntologyNormaliser`.
- [X] **Add** the ontology projector at `deeponto.onto.OntologyProjector`.

### Changed

- [X] **Change** the dependency `transformers` to `transformers[torch]`.


## v0.7.5 (2023 June)

### Changed

- [X] **Change** Java dependencies from using `lib` from mowl to direct import.
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