# Changelog :newspaper:

<!-- Added for new features.
Changed for changes in existing functionality.
Deprecated for soon-to-be removed features.
Removed for now removed features.
Fixed for any bug fixes.
Security in case of vulnerabilities. -->

## v0.9.2 (2024 Oct)

### Changed

- [X] **Refine** `deeponto.onto.taxonomy` and add more options.

## v0.9.1 (2024 Mar)

### Changed

- [X] **Move** `openprompt` to extra dependencies as its supported version of `sentencepiece` has known conflicts with many packages.
- [X] **Add** `nltk` and `tqdm` as a dependency.


## v0.8.9 (2024 Mar)

### Added

<!-- - [ ] **Add** ICON as a completion service at `deeponto.complete`. -->
- [X] **Add** empty annotation index warning for BERTMap, related to issue #18.
- [X] **Add** `check_consistency()` at `deeponto.onto.Ontology`.
- [X] **Add** a warning message for empty vocab at `deeponto.onto.OntologyVerbaliser`.

### Changed

- [X] **Remove** dependency version restrictions.
- [X] **Change** `.instances_of()` to `get_instances()` at `deeponto.onto.OntologyReasoner`.
- [X] **Change** `deeponto.subs` to `deeponto.complete`.
- [X] **Move** `deeponto.probe.ontolama` into `deeponto.complete`.

## v0.8.8 (2023 October)

### Added

- [X] **Add** object property domain/range verbalisation at `deeponto.onto.OntologyVerbaliser`.
- [X] **Add** new reasoner type `"struct"` ([Structural Reasoner](https://owlcs.github.io/owlapi/apidocs_4/org/semanticweb/owlapi/reasoner/structural/StructuralReasoner.html)) at `deeponto.onto.OntologyReasoner`.
- [X] **Add** `load_reasoner()` method at `deeponto.onto.OntologyReasoner` for convenience of changing the reasoner type and remove `reload_reasoner()` method as it is a special case of `load_reasoner()`.
- [x] **Add** `rdflib` into the dependencies for building graph-related features.
- [X] **Add** `deeponto.onto.taxonomy` for building the taxonomy over ontologies and potentially other structured data.

### Changed
- [X] **Change** printing to appropriate logging (gradually).
- [X] **Change** `read_table_mappings()` method at `deeponto.align.mapping` from using `dataframe.iterrows()` to `dataframe.itertuples()` which is much more efficient.
- [X] **Change** the default lowcasing argument of `deeponto.utils.process_annotation_literal()` to `False`.
- [X] **Change** the default logging level of `slf4j` to `warn` to prevent tons of printing at ELK (issue (#13)[https://github.com/KRR-Oxford/DeepOnto/issues/13]).

## v0.8.7 (2023 September)

### Added

- [x] **Add** the OAEI evaluation code for the main track global matching, local ranking, and the special sub-track bio-llm at `deeponto.align.oaei`.
- [x] **Add** `reasoner_type` argument at `deeponto.onto.OntologyReasoner`, now supporting `hermit` (default) and `elk`.
- [x] **Add** `get_all_axioms()` method at `deeponto.onto.Ontology`.
- [x] **Add** `get_iri()` method at `deeponto.onto.Ontology`.

- [x] **Add** new features into `deeponto.onto.OntologyVerbaliser` including:

  - `verbalise_object_property_subsumption()` for object property subsumption axioms.
  - property chain verbalisation at `verbalise_class_expression()`.
  - `verbalise_class_subsumption()` for class subsumption axioms;
  - `verbalise_class_equivalence()` for class equivalence axioms;
  - `verbalise_class_assertion()` for class assertion axioms;
  - `verbalise_relation_assertion()` for relation assertion axioms;
  - `auto-correction` option for fixing entity names.
  - `keep_iri` option for keeping entity IRIs.
  - `add_quantifier_word` option for adding quantifier words as in the Manchester syntax.

- [x] **Add** `get_assertion_axioms()` method at `deeponto.onto.Ontology`.
- [x] **Add** `get_axiom_type()` method at `deeponto.onto.Ontology`.
- [x] **Add** `owl_individuals` attribute at `deeponto.onto.Ontology`.

### Changed

- [x] **Change** `get_owl_objects()` method to be anonymous as it is only used for creating pre-processed entity index at `deeponto.onto.Ontology`.
- [x] **Change** `get_owl_object_from_iri()` method to `get_owl_object()` at `deeponto.onto.Ontology`.
- [x] <del>Change the log level of the ELK reasoner to `ERROR`</del>.

### Fixed

- [x] **Fix** the file path problem of loading ontologies for Windows systems.
- [x] **Fix** the version of ELK to the latest by manually adding in the dependencies. See download link at <https://github.com/liveontologies/elk-reasoner/wiki/GettingElk>.

## v0.8.5 (2023 September)

### Added

- [x] **Add** `set_seed()` method at `deeponto.utils`.

### Changed

- [x] **Change** the layout of all utility methods by making them stand-alone instead of static methods.
- [x] **Change** the `.verbalise_class_expression()` method by adding an option to keep entity IRIs without verbalising them using `.vocabs` at `deeponto.onto.OntologyVerbaliser`.
- [x] **Change** default `apply_lowercasing` value to `False` for both `.get_annotations()` and `.build_annotation_index()` methods at `deeponto.onto.Ontology`.
- [x] **Change** the method `.get_owl_object_annotations()` to `.get_annotations()` at `deeponto.onto.Ontology`.
- [x] **Change** the LogMap debugger memory options for BERTMap's mapping repair.
- [x] **Change** the default jar command timeout to 1 hour.

### Fixed

- [x] **Fix** duplicate logging in running BERTMap due to progapagation.

## v0.8.4 (2023 July)

### Added

- [x] **Add** specific check of `use_in_alignment` annotation in BERTMap for the OAEI.
- [x] **Add** OAEI utilities at `deeponto.align.oaei`.

### Changed

- [x] **Change** the `read_table_mappings` method to allow `None` for threshold.

### Fixed

- [x] **Fix** BERTMap error and add corresponding warning when an input ontology has no sibling class group, related to Issue #10.
- [x] **Fix** BERTMap error and add corresponding warning when an input ontology has some class with no label (annotation), related to Issue #10.

## v0.8.3 (2023 July)

### Changed

- [x] **Change** the mapping extension from using reasoner to direct assertions.
- [x] **Change** the name of pruning function in `deeponto.onto.OntologyPruner`.
- [x] **Change** the verbalisation function by setting quantifier words as optional (by default not adding).
- [x] **Change** sibing retrieval from using reasoner to direct assertions.

### Fixed

- [x] **Fix** the minor bug for the `f1` and `MRR` method in `deeponto.align.evaluation.AlignmentEvaluator`.

## v0.8.0 (2023 June)

### Added

- [x] **Add** the ontology normaliser at `deeponto.onto.OntologyNormaliser`.
- [x] **Add** the ontology projector at `deeponto.onto.OntologyProjector`.

### Changed

- [x] **Change** the dependency `transformers` to `transformers[torch]`.

## v0.7.5 (2023 June)

### Changed

- [x] **Change** Java dependencies from using `lib` from mowl to direct import.
- [x] **Change** `get_owl_object_annotations` by adding `uniqify` at the end to preserve the order.

### Fixed

- [x] **Fix** BERTMap's non-synonym sampling when the class labels are not available using the try-catch block.

## v0.7.0 (2023 April)

### Added

- [x] **Add** the BERTSubs module at `deeponto.subs.bertsubs`; its inter-ontology setting is also imported at `deeponto.align.bertsubs`.

### Changed

- [x] **Move** the pruning functionality into `deeponto.onto.OntologyPruner` as a separate module.
- [x] **Amend** JVM checking before displaying the JVM memory prompt from importing `deeponto.onto.Ontology`; if started already, skip this step.
- [x] **Change** the function `get_owl_object_annotations` at `deeponto.onto.Ontology` by preserving the relative order of annotation retrieval, i.e., create `set` first and use the `.add()` function instead of casting the `list` into `set` in the end.

### Fixed

- [x] **Fix** the function `check_deprecated` at `deeponto.onto.Ontology` by adding a check for the $\texttt{owl:deprecated}$ annotation property -- if this property does not exist in the current ontology, return `False` (not deprecated).

## v0.6.1 (2023 April)

### Added

- [x] **Add** the method `remove_axiom` for removing an axiom from the ontology at `deeponto.onto.Ontology` (note that the counterpart `add_axiom` has already been available).
- [x] **Add** the method `check_named_entity` for checking if an entity is named at `deeponto.onto.Ontology`.
- [x] **Add** the method `get_subsumption_axioms` for getting subsumption axioms subject to different entity types at `deeponto.onto.Ontology`.
- [x] **Add** the method `get_asserted_complex_classes` for getting all complex classes that occur in ontology (subsumption and/or equivalence) axioms at `deeponto.onto.Ontology`.
- [x] **Add** the methods `get_asserted_parents` and `get_asserted_children` for getting asserted parent and children for a given entity at `deeponto.onto.Ontology`.
- [x] **Add** the method `check_deprecation` for checking an owl object's deprecation (annotated) at `deeponto.onto.Ontology`.

### Changed

- [x] **Move** the spacy `en_core_web_sm` download into the initialisation of `OntologyVerbaliser`.
- [x] **Change** the method of getting equivalence axioms by adding support to different entity types at `deeponto.onto.Ontology`.
- [x] **Rename** the methods of getting inferred super-entities and sub-entities at `deeponto.onto.OntologyReasoner`:
  - `super_entities_of` $\rightarrow$ `get_inferred_super_entities`
  - `sub_entities_of` $\rightarrow$ `get_inferred_sub_entities`

### Fixed

- [x] **Fix** the top and bottom data property iris (from "https:" to "http:") at `deeponto.onto.Ontology`.

## v0.6.0 (2023 Mar)

- [x] **Add** the OntoLAMA module at `deeponto.lama`.
- [x] **Add** the verb auto-correction and more precise documentation for `deeponto.onto.verbalisation`.

## v0.5.x (2023 Jan - Feb)

- [x] **Add** the preliminary ontology verbalisation module at `deeponto.onto.verbalisation`.
- [x] **Fix** PyPI issues based on the new code layout.
- [x] **Change** code layout to the `src/` layout.
- [x] **Rebuild** the whole package based on the OWLAPI.
- [x] **Remove** owlready2 from the essential dependencies.

## Deprecated (before 2023)

The code before v0.5.0 is no longer available.
