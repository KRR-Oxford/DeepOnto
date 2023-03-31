# Changelog :newspaper:

<!-- Added for new features.
Changed for changes in existing functionality.
Deprecated for soon-to-be removed features.
Removed for now removed features.
Fixed for any bug fixes.
Security in case of vulnerabilities. -->


## Unreleased

- [ ] **Add** the ontology normaliser at `deeponto.onto.normaliser`.
- [ ] **Add** the BERTSubs module at `deeponto.onto.subs`.
- [ ] **Add** the `subclass_dict` method for `deeponto.onto.Ontology`.
- [ ] **Add** the `owl_complex_classes` attribute for `deeponto.onto.Ontology`.
- [ ] **Add** the [detailed instructions](../verbaliser) for how to use the ontology verbaliser. 
- [X] **Add** `check_deprecated` method into the `deeponto.onto.Ontology` class.

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