# Copyright 2021 Yuan He. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging

logging.basicConfig(level=logging.INFO)

from . import Ontology

from de.tudresden.inf.lat.jcel.ontology.normalization import OntologyNormalizer  # type: ignore
from de.tudresden.inf.lat.jcel.ontology.axiom.extension import IntegerOntologyObjectFactoryImpl  # type: ignore
from de.tudresden.inf.lat.jcel.owlapi.translator import ReverseAxiomTranslator  # type: ignore
from de.tudresden.inf.lat.jcel.owlapi.translator import Translator  # type: ignore
from org.semanticweb.owlapi.model.parameters import Imports  # type: ignore
from java.util import HashSet  # type: ignore


class OntologyNormaliser:
    r"""Class for ontology normalisation.

    !!! note "Credit"

        The code of this class originates from the [mOWL library](https://mowl.readthedocs.io/en/latest/index.html),
        which utilises the normalisation functionality from the Java library `Jcel`.

    The normalisation process transforms ontology axioms into **normal forms** in the Description Logic $\mathcal{EL}$, including:

    - $C \sqsubseteq D$
    - $C \sqcap C' \sqsubseteq D$
    - $C \sqsubseteq \exists r.D$
    - $\exists r.C \sqsubseteq D$

    where $C$ and $C'$ can be named concepts or $\top$, $D$ is a named concept or $\bot$, $r$ is a role (property).



    Attributes:
        onto (Ontology): The input ontology to be normalised.
        temp_super_class_index (Dict[OWLCLassExpression, OWLClass]): A dictionary in the form of `{complex_sub_class: temp_super_class}`, which means
            `temp_super_class` is created during the normalisation of a complex subsumption axiom that has `complex_sub_class` as the sub-class.
    """

    def __init__(self):
        return

    def normalise(self, ontology: Ontology):
        r"""Performs the $\mathcal{EL}$ normalisation.

        Args:
            ontology (Ontology): An ontology to be normalised.

        Returns:
            (List[OWLAxiom]): A list of normalised TBox axioms.
        """

        processed_owl_onto = self.preprocess_ontology(ontology)
        root_ont = processed_owl_onto
        translator = Translator(
            processed_owl_onto.getOWLOntologyManager().getOWLDataFactory(), IntegerOntologyObjectFactoryImpl()
        )
        axioms = HashSet()
        axioms.addAll(root_ont.getAxioms())
        translator.getTranslationRepository().addAxiomEntities(root_ont)

        for ont in root_ont.getImportsClosure():
            axioms.addAll(ont.getAxioms())
            translator.getTranslationRepository().addAxiomEntities(ont)

        intAxioms = translator.translateSA(axioms)

        normaliser = OntologyNormalizer()

        factory = IntegerOntologyObjectFactoryImpl()
        normalised_ontology = normaliser.normalize(intAxioms, factory)
        self.rTranslator = ReverseAxiomTranslator(translator, processed_owl_onto)
        
        normalised_axioms = []
        # revert the jcel axioms to the original OWLAxioms
        for ax in normalised_ontology:
            try:
                axiom = self.rTranslator.visit(ax)
                normalised_axioms.append(axiom)
            except Exception as e:
                logging.info("Reverse translation. Ignoring axiom: %s", ax)
                logging.info(e)
                
        return list(set(axioms))

    def preprocess_ontology(self, ontology: Ontology):
        """Preprocess the ontology to remove axioms that are not supported by the normalisation process."""
        
        tbox_axioms = ontology.owl_onto.getTBoxAxioms(Imports.fromBoolean(True))
        new_tbox_axioms = HashSet()

        for axiom in tbox_axioms:
            axiom_as_str = axiom.toString()

            if "UnionOf" in axiom_as_str:
                continue
            elif "MinCardinality" in axiom_as_str:
                continue
            elif "ComplementOf" in axiom_as_str:
                continue
            elif "AllValuesFrom" in axiom_as_str:
                continue
            elif "MaxCardinality" in axiom_as_str:
                continue
            elif "ExactCardinality" in axiom_as_str:
                continue
            elif "Annotation" in axiom_as_str:
                continue
            elif "ObjectHasSelf" in axiom_as_str:
                continue
            elif "urn:swrl" in axiom_as_str:
                continue
            elif "EquivalentObjectProperties" in axiom_as_str:
                continue
            elif "SymmetricObjectProperty" in axiom_as_str:
                continue
            elif "AsymmetricObjectProperty" in axiom_as_str:
                continue
            elif "ObjectOneOf" in axiom_as_str:
                continue
            else:
                new_tbox_axioms.add(axiom)

        processed_owl_onto = ontology.owl_manager.createOntology(new_tbox_axioms)
        # NOTE: the returned object is `owlapi.OWLOntology` not `deeponto.onto.Ontology`
        return processed_owl_onto
