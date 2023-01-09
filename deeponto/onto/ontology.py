# Copyright 2021 Yuan He (KRR-Oxford). All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Ontology Classs, extended from OWLAPI"""

from __future__ import annotations

import os
from typing import Optional, List, Union
from collections import defaultdict
from yacs.config import CfgNode
from owlready2 import get_ontology, default_world
from owlready2.entity import ThingClass
from pathlib import Path

from deeponto.utils import FileProcessor, TextProcessor, Tokenizer, InvertedIndex
from deeponto import init_jvm

# initialise JVM for python-java interaction
init_jvm("2g")

from java.io import File  # type: ignore
from org.semanticweb.owlapi.apibinding import OWLManager  # type: ignore
from org.semanticweb.owlapi.model import IRI, OWLObject, OWLClassExpression, OWLObjectPropertyExpression  # type: ignore
from org.semanticweb.HermiT import ReasonerFactory  # type: ignore
from org.semanticweb.owlapi.util import OWLObjectDuplicator  # type: ignore
from org.semanticweb.owlapi.search import EntitySearcher  # type: ignore

# IRIs for special entities
OWL_THING = "http://www.w3.org/2002/07/owl#Thing"
OWL_NOTHING = "http://www.w3.org/2002/07/owl#Nothing"
OWL_TOP_OBJECT_PROPERTY = "http://www.w3.org/2002/07/owl#topObjectProperty"
OWL_BOTTOM_OBJECT_PROPERTY = "http://www.w3.org/2002/07/owl#bottomObjectProperty"
OWL_TOP_DATA_PROPERTY = "https://www.w3.org/2002/07/owl#topDataProperty"
OWL_BOTTOM_DATA_PROPERTY = "https://www.w3.org/2002/07/owl#bottomDataProperty"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"

TOP_BOTTOMS = CfgNode(
    {
        "Classes": {"TOP": OWL_THING, "BOTTOM": OWL_NOTHING},
        "ObjectProperties": {"TOP": OWL_TOP_OBJECT_PROPERTY, "BOTTOM": OWL_BOTTOM_OBJECT_PROPERTY},
        "DataProperties": {"TOP": OWL_TOP_DATA_PROPERTY, "BOTTOM": OWL_BOTTOM_DATA_PROPERTY},
    }
)


class Ontology:
    """Ontology class that extends from the Java library OWLAPI.

    Attributes:
        owl_path (str): A string indicating the path to the OWL ontology file.
        owl_manager (OWLOntologyManager): An instance of OWLOntologyManager.
        owl_onto (OWLOntology): An instance of OWLOntology loaded from `owl_path`.
        owl_iri (str): The IRI of the imported OWLOntology instance.
        owl_classes (dict): A dictionary that stores the (iri, ontology class) pairs.
        owl_object_properties (dict): A dictionary that stores the (iri, ontology object property) pairs.
        owl_data_properties (dict): A dictionary that stores the (iri, ontology data property) pairs.
        owl_data_factor (OWLDataFactory): An instance of OWLDataFactory for manipulating axioms.
    """

    def __init__(self, owl_path: str):
        self.owl_path = os.path.abspath(owl_path)
        self.owl_manager = OWLManager.createOWLOntologyManager()
        self.owl_onto = self.owl_manager.loadOntologyFromOntologyDocument(
            IRI.create("file:///" + self.owl_path)
        )
        self.owl_iri = str(self.owl_onto.getOntologyID().getOntologyIRI().get())
        self.owl_classes = self.get_owl_objects("Classes")
        self.owl_object_properties = self.get_owl_objects("ObjectProperties")
        self.owl_data_properties = self.get_owl_objects("DataProperties")
        self.owl_data_factory = self.owl_manager.getOWLDataFactory()
        self.owl_annotation_properties = self.get_owl_objects("AnnotationProperties")

    @property
    def OWLThing(self):
        return self.owl_data_factory.getOWLThing()

    @property
    def OWLNothing(self):
        return self.owl_data_factory.getOWLNothing()

    @property
    def OWLTopObjectProperty(self):
        return self.owl_data_factory.getOWLTopObjectProperty()

    @property
    def OWLBottomObjectProperty(self):
        return self.owl_data_factory.getOWLBottomObjectProperty()

    @property
    def OWLTopDataProperty(self):
        return self.owl_data_factory.getOWLTopDataProperty()

    @property
    def OWLBottomDataProperty(self):
        return self.owl_data_factory.getOWLBottomDataProperty()

    def __str__(self) -> str:
        self.info = {
            "loaded_from": os.path.normpath(self.owl_path).split(os.path.sep)[-1],
            "num_classes": len(self.owl_classes),
        }
        return FileProcessor.print_dict(self.info)

    def get_owl_objects(self, entity_type: str):
        """Get OWLObject instances from the ontology.

        Args:
            entity_type (str): Options are "Classes", "ObjectProperties", "DatasetProperties", etc.

        Returns:
            dict: A dictionary that stores the (IRI, OWLObject) pairs
        """
        owl_objects = dict()
        source = getattr(self.owl_onto, f"get{entity_type}InSignature")
        for cl in source():
            owl_objects[str(cl.getIRI())] = cl
        return owl_objects

    def get_owl_object_from_iri(self, iri: str):
        """Get an OWLObject instance given its IRI.
        """
        if iri in self.owl_classes.keys():
            return self.owl_classes[iri]
        elif iri in self.owl_object_properties.keys():
            return self.owl_object_properties[iri]
        elif iri in self.owl_data_properties.keys():
            return self.owl_data_properties[iri]
        elif iri in self.owl_annotation_properties.keys():
            return self.owl_annotation_properties[iri]
        else:
            raise KeyError(f"Cannot retrieve unknown IRI: {iri}.")

    def get_owl_object_annotations(
        self,
        owl_object: Union[OWLObject, str],
        annotation_property_iri: Optional[str] = None,
        annotation_language_tag: Optional[str] = None,
        apply_lowercasing: bool = True,
    ):
        """Get the annotations of the given OWLObject instance.

        Args:
            owl_object (Union[OWLObject, str]): An OWLObject instance or its IRI.
            annotation_property_iri (Optional[str], optional): 
                Any particular annotation property IRI of interest. Defaults to None.
            annotation_language_tag (Optional[str], optional): 
                Any particular annotation language tag of interest; 
                NOTE that not every annotation has a language tag. 
                Defaults to None. Options are "en", "de" etc.
            apply_lowercasing (bool): Whether or not to apply lowercasing to annotation literals. 
                Defaults to True.
        Returns:
            List[str]: A list of annotation literals of the given OWLObject.
        """
        if isinstance(owl_object, str):
            owl_object = self.get_owl_object_from_iri(owl_object)

        annotation_property = None
        if annotation_property_iri:
            # return an empty list if `annotation_property_iri` does not exist in this OWLOntology`
            annotation_property = self.get_owl_object_from_iri(annotation_property_iri)

        annotations = []
        for annotation in EntitySearcher.getAnnotations(
            owl_object, self.owl_onto, annotation_property
        ):

            annotation = annotation.getValue()
            # boolean that indicates whether the annotation's language is of interest
            fit_language = False
            if not annotation_language_tag:
                # it is set to `True` if `annotation_langauge` is not specified
                fit_language = True
            else:
                # restrict the annotations to a language if specified
                try:
                    # NOTE: not every annotation has a language attribute
                    fit_language = annotation.getLang() == annotation_language_tag
                except:
                    pass

            if fit_language:
                # only get annotations that have a literal value
                if annotation.isLiteral():
                    annotations.append(
                        TextProcessor.process_annotation_literal(
                            str(annotation.getLiteral()), apply_lowercasing
                        )
                    )

        return set(annotations)

    def build_annotation_index(
        self,
        annotation_property_iris: Optional[List[str]] = None,
        entity_type: str = "Classes",
        apply_lowercasing: bool = True,
    ):
        """Build an annotation index for a given type of entities.

        NOTE: only English labels are considererd (with English language tag)
        
        Args:
            annotation_property_iris (List[str]): A list of annotation property IRIs; 
                if not provided, all English annotations are considered. Defaults to None.
            entity_type (str, optional): The entity type to be considered. Defaults to "Classes". 
                Options are "Classes", "ObjectProperties", "DatasetProperties", etc.
            apply_lowercasing (bool): Whether or not to apply lowercasing to annotation literals. 
                Defaults to True.
        """

        annotation_index = defaultdict(set)
        # example: Classes => owl_classes; ObjectProperties => owl_object_properties
        entity_type = (
            "owl_" + TextProcessor.split_java_identifier(entity_type).replace(" ", "_").lower()
        )
        entity_index = getattr(self, entity_type)

        if not annotation_property_iris:
            annotation_property_iris = [RDFS_LABEL]  # rdfs:label is the default annotation property

        # preserve available annotation properties
        annotation_property_iris = [
            airi
            for airi in annotation_property_iris
            if airi in self.owl_annotation_properties.keys()
        ]

        # build the annotation index without duplicated literals
        for airi in annotation_property_iris:
            for iri, entity in entity_index.items():
                annotation_index[iri].update(
                    self.get_owl_object_annotations(
                        owl_object=entity,
                        annotation_property_iri=airi,
                        annotation_language_tag=None,
                        apply_lowercasing=apply_lowercasing,
                    )
                )

        return annotation_index, annotation_property_iris

    def build_inverted_annotation_index(self, annotation_index: dict, tokenizer: Tokenizer):
        """Builds an inverted annotation index given an annotation index and a tokenizer.
        """
        return InvertedIndex(annotation_index, tokenizer)

    def save_onto(self, save_path: str):
        """Save the OWL Ontology file to the given path
        """
        self.owl_onto.saveOntology(IRI.create(File(save_path).toURI()))

