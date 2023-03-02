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

from __future__ import annotations

import os
from typing import Optional, List, Union
from collections import defaultdict
from yacs.config import CfgNode
import warnings
import itertools

from deeponto.utils import TextUtils, Tokenizer, InvertedIndex, FileUtils
from deeponto.utils.decorators import paper
from deeponto import init_jvm

# initialise JVM for python-java interaction
import click

memory = click.prompt("Please enter the maximum memory located to JVM", type=str, default="8g")
print()
init_jvm(memory)

from java.io import File  # type: ignore
from java.util import Collections  # type: ignore
from org.semanticweb.owlapi.apibinding import OWLManager  # type: ignore
from org.semanticweb.owlapi.model import IRI, OWLObject, OWLClassExpression, OWLObjectPropertyExpression, OWLDataPropertyExpression, OWLNamedIndividual, OWLAxiom, AddAxiom  # type: ignore
from org.semanticweb.HermiT import ReasonerFactory  # type: ignore
from org.semanticweb.owlapi.util import OWLObjectDuplicator, OWLEntityRemover  # type: ignore
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

    !!! note "Typing from OWLAPI"

        Types with `OWL` prefix are mostly imported from the OWLAPI library by, for example,
        `from org.semanticweb.owlapi.model import OWLObject`.

    Attributes:
        owl_path (str): The path to the OWL ontology file.
        owl_manager (OWLOntologyManager): A ontology manager for creating `OWLOntology`.
        owl_onto (OWLOntology): An `OWLOntology` created by `owl_manger` from `owl_path`.
        owl_iri (str): The IRI of the `owl_onto`.
        owl_classes (Dict[str, OWLClass]): A dictionary that stores the `(iri, ontology_class)` pairs.
        owl_object_properties (Dict[str, OWLObjectProperty]): A dictionary that stores the `(iri, ontology_object_property)` pairs.
        owl_data_properties (Dict[str, OWLDataProperty]): A dictionary that stores the `(iri, ontology_data_property)` pairs.
        owl_data_factory (OWLDataFactory): A data factory for manipulating axioms.
        owl_annotation_properties (Dict[str, OWLAnnotationProperty]): A dictionary that stores the `(iri, ontology_annotation_property)` pairs.
        reasoner (OntologyReasoner): A reasoner for ontology inference.
    """

    def __init__(self, owl_path: str):
        """Initialise a new ontology.

        Args:
            owl_path (str): The path to the OWL ontology file.
        """
        self.owl_path = os.path.abspath(owl_path)
        self.owl_manager = OWLManager.createOWLOntologyManager()
        self.owl_onto = self.owl_manager.loadOntologyFromOntologyDocument(IRI.create("file:///" + self.owl_path))
        self.owl_iri = str(self.owl_onto.getOntologyID().getOntologyIRI().get())
        self.owl_classes = self.get_owl_objects("Classes")
        self.owl_object_properties = self.get_owl_objects("ObjectProperties")
        self.owl_data_properties = self.get_owl_objects("DataProperties")
        self.owl_data_factory = self.owl_manager.getOWLDataFactory()
        self.owl_annotation_properties = self.get_owl_objects("AnnotationProperties")

        # reasoning
        self.reasoner = OntologyReasoner(self)

        # hidden attributes
        self._multi_children_classes = None
        self._sibling_class_groups = None
        self._equiv_axioms = None

        # summary
        self.info = {
            type(self).__name__: {
                "loaded_from": os.path.basename(self.owl_path),
                "num_classes": len(self.owl_classes),
                "num_object_properties": len(self.owl_object_properties),
                "num_data_properties": len(self.owl_data_properties),
                "num_annotation_properties": len(self.owl_annotation_properties),
            }
        }

    @property
    def name(self):
        """Return the name of the ontology file."""
        return os.path.normpath(self.owl_path).split(os.path.sep)[-1]

    @property
    def OWLThing(self):
        """Return `OWLThing`."""
        return self.owl_data_factory.getOWLThing()

    @property
    def OWLNothing(self):
        """Return `OWLNoThing`."""
        return self.owl_data_factory.getOWLNothing()

    @property
    def OWLTopObjectProperty(self):
        """Return `OWLTopObjectProperty`."""
        return self.owl_data_factory.getOWLTopObjectProperty()

    @property
    def OWLBottomObjectProperty(self):
        """Return `OWLBottomObjectProperty`."""
        return self.owl_data_factory.getOWLBottomObjectProperty()

    @property
    def OWLTopDataProperty(self):
        """Return `OWLTopDataProperty`."""
        return self.owl_data_factory.getOWLTopDataProperty()

    @property
    def OWLBottomDataProperty(self):
        """Return `OWLBottomDataProperty`."""
        return self.owl_data_factory.getOWLBottomDataProperty()

    @staticmethod
    def get_entity_type(entity: OWLObject, is_singular: bool = False):
        """A handy method to get the `type` of an `OWLObject` entity."""
        if isinstance(entity, OWLClassExpression):
            return "Classes" if not is_singular else "Class"
        elif isinstance(entity, OWLObjectPropertyExpression):
            return "ObjectProperties" if not is_singular else "ObjectProperty"
        elif isinstance(entity, OWLDataPropertyExpression):
            return "DataProperties" if not is_singular else "DataProperty"
        else:
            # NOTE: add further options in future
            pass

    def __str__(self) -> str:
        return FileUtils.print_dict(self.info)

    def get_owl_objects(self, entity_type: str):
        """Get an index of `OWLObject` of certain type from the ontology.

        Args:
            entity_type (str): Options are `"Classes"`, `"ObjectProperties"`, `"DatasetProperties"`, etc.

        Returns:
            (dict): A dictionary that stores the `(iri, owl_object)` pairs
        """
        owl_objects = dict()
        source = getattr(self.owl_onto, f"get{entity_type}InSignature")
        for cl in source():
            owl_objects[str(cl.getIRI())] = cl
        return owl_objects

    def get_owl_object_from_iri(self, iri: str):
        """Get an `OWLObject` given its IRI."""
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
        normalise_identifiers: bool = False,
    ):
        """Get the annotations of the given `OWLObject`.

        Args:
            owl_object (Union[OWLObject, str]): An `OWLObject` or its IRI.
            annotation_property_iri (str, optional):
                Any particular annotation property IRI of interest. Defaults to `None`.
            annotation_language_tag (str, optional):
                Any particular annotation language tag of interest; NOTE that not every
                annotation has a language tag, in this case assume it is in English.
                Defaults to `None`. Options are `"en"`, `"ge"` etc.
            apply_lowercasing (bool): Whether or not to apply lowercasing to annotation literals.
                Defaults to `True`.
            normalise_identifiers (bool): Whether to normalise annotation text that is in the Java identifier format.
                Defaults to `False`.
        Returns:
            (Set[str]): A set of annotation literals of the given `OWLObject`.
        """
        if isinstance(owl_object, str):
            owl_object = self.get_owl_object_from_iri(owl_object)

        annotation_property = None
        if annotation_property_iri:
            # return an empty list if `annotation_property_iri` does not exist in this OWLOntology`
            annotation_property = self.get_owl_object_from_iri(annotation_property_iri)

        annotations = []
        for annotation in EntitySearcher.getAnnotations(owl_object, self.owl_onto, annotation_property):

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
                    # in the case when this annotation has no language tag
                    # we assume it is in English
                    if annotation_language_tag == "en":
                        fit_language = True

            if fit_language:
                # only get annotations that have a literal value
                if annotation.isLiteral():
                    annotations.append(
                        TextUtils.process_annotation_literal(
                            str(annotation.getLiteral()), apply_lowercasing, normalise_identifiers
                        )
                    )

        return set(annotations)

    @property
    def sibling_class_groups(self) -> List[List[str]]:
        """Return grouped sibling classes (with a common *direct* parent);

        NOTE that only groups with size > 1 will be considered
        """
        if not self._sibling_class_groups:

            self._multi_children_classes = dict()
            self._sibling_class_groups = []
            all_class_iris = list(self.owl_classes.keys()) + [OWL_THING]  # including the root node

            for cl_iri in all_class_iris:

                if cl_iri == OWL_THING:
                    cl = self.OWLThing
                else:
                    cl = self.get_owl_object_from_iri(cl_iri)

                children_iris = self.reasoner.sub_entities_of(cl, direct=True)
                self._multi_children_classes[cl_iri] = children_iris

                if len(children_iris) > 1:
                    # classes that have siblings form a sibling group
                    if children_iris not in self._sibling_class_groups:
                        # it is possible that some groups appear more than once be they have mutltiple
                        # common parents
                        self._sibling_class_groups.append(children_iris)

        return self._sibling_class_groups

    @property
    def equivalence_axioms(self):
        """Return all the equivalence class axioms asserted in this `OWLOntology`."""
        if not self._equiv_axioms:
            self._equiv_axioms = []
            for cl in self.owl_classes.values():
                self._equiv_axioms += self.owl_onto.getEquivalentClassesAxioms(cl)
            self._equiv_axioms = list(set(self._equiv_axioms))
        return self._equiv_axioms

    def save_onto(self, save_path: str):
        """Save the ontology file to the given path."""
        self.owl_onto.saveOntology(IRI.create(File(save_path).toURI()))

    def build_annotation_index(
        self,
        annotation_property_iris: List[str] = [RDFS_LABEL],
        entity_type: str = "Classes",
        apply_lowercasing: bool = True,
        normalise_identifiers: bool = False,
    ):
        """Build an annotation index for a given type of entities.

        Args:
            annotation_property_iris (List[str]): A list of annotation property IRIs (it is possible
                that not every annotation property IRI is in use); if not provided, the built-in
                `rdfs:label` is considered. Defaults to `[RDFS_LABEL]`.
            entity_type (str, optional): The entity type to be considered. Defaults to `"Classes"`.
                Options are `"Classes"`, `"ObjectProperties"`, `"DatasetProperties"`, etc.
            apply_lowercasing (bool): Whether or not to apply lowercasing to annotation literals.
                Defaults to `True`.
            normalise_identifiers (bool): Whether to normalise annotation text that is in the Java identifier format.
                Defaults to `False`.

        Returns:
            (Tuple[dict, List[str]]): The built annotation index, and the list of annotation property IRIs that are in use.
        """

        annotation_index = defaultdict(set)
        # example: Classes => owl_classes; ObjectProperties => owl_object_properties
        entity_type = "owl_" + TextUtils.split_java_identifier(entity_type).replace(" ", "_").lower()
        entity_index = getattr(self, entity_type)

        # preserve available annotation properties
        annotation_property_iris = [
            airi for airi in annotation_property_iris if airi in self.owl_annotation_properties.keys()
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
                        normalise_identifiers=normalise_identifiers,
                    )
                )

        return annotation_index, annotation_property_iris

    @staticmethod
    def build_inverted_annotation_index(annotation_index: dict, tokenizer: Tokenizer):
        """Build an inverted annotation index given an annotation index and a tokenizer."""
        return InvertedIndex(annotation_index, tokenizer)

    def add_axiom(self, owl_axiom: OWLAxiom, return_undo: bool = True):
        """Add an axiom into the current ontology.

        Args:
            owl_axiom (OWLAxiom): An axiom to be added.
            return_undo (bool, optional): Returning the undo operation or not. Defaults to True.
        """
        change = AddAxiom(self.owl_onto, owl_axiom)
        result = self.owl_onto.applyChange(change)
        print(f"[{str(result)}] Adding the axiom {str(owl_axiom)} into the ontology.")
        if return_undo:
            return change.reverseChange()

    def replace_entity(self, owl_object: OWLObject, entity_iri: str, replacement_iri: str):
        """Replace an entity in a class expression with another entity.

        Args:
            owl_object (OWLObject): An `OWLObject` entity to be manipulated.
            entity_iri (str): IRI of the entity to be replaced.
            replacement_iri (str): IRI of the entity to replace.

        Returns:
            (OWLObject): The changed `OWLObject` entity.
        """
        iri_dict = {IRI.create(entity_iri): IRI.create(replacement_iri)}
        replacer = OWLObjectDuplicator(self.owlDataFactory, iri_dict)
        return replacer.duplicateObject(owl_object)

    @paper(
        "Machine Learning-Friendly Biomedical Datasets for Equivalence and Subsumption Ontology Matching (ISWC 2022)",
        "https://link.springer.com/chapter/10.1007/978-3-031-19433-7_33",
    )
    def apply_pruning(self, class_iris_to_be_removed: List[str]):
        r"""Run pruning given a list of classes that will be pruned.

        !!! credit "paper"

            This refers to the ontology pruning algorithm introduced in the paper:
            [*Machine Learning-Friendly Biomedical Datasets for Equivalence and Subsumption Ontology Matching (ISWC 2022)*](https://link.springer.com/chapter/10.1007/978-3-031-19433-7_33).

        For each class $c$ to be pruned, subsumption axioms will be created between $c$'s parents and children so as to preserve the
        relevant hierarchy.

        Args:
            class_iris_to_be_removed (List[str]): Classes with IRIs in this list will be pruned and the relevant hierarchy will be repaired.
        """

        # create the subsumption axioms first
        for cl_iri in class_iris_to_be_removed:
            cl = self.get_owl_object_from_iri(cl_iri)
            cl_parents = self.reasoner.super_entities_of(cl, direct=True)
            cl_children = self.reasoner.sub_entities_of(cl, direct=True)
            for parent, child in itertools.product(cl_parents, cl_children):
                parent = self.get_owl_object_from_iri(parent)
                child = self.get_owl_object_from_iri(child)
                sub_axiom = self.owl_data_factory.getOWLSubClassOfAxiom(child, parent)
                self.add_axiom(sub_axiom)

        # apply pruning
        class_remover = OWLEntityRemover(Collections.singleton(self.owl_onto))
        for cl_iri in class_iris_to_be_removed:
            cl = self.get_owl_object_from_iri(cl_iri)
            cl.accept(class_remover)
        self.owl_manager.applyChanges(class_remover.getChanges())

        # remove IRIs in dictionaries?
        # TODO Test it


class OntologyReasoner:
    """Ontology reasoner class that extends from the Java library OWLAPI.

    Attributes:
        onto (Ontology): The input `deeponto` ontology.
        owl_reasoner_factory (OWLReasonerFactory): A reasoner factory for creating a reasoner.
        owl_reasoner (OWLReasoner): The created reasoner.
        owl_data_factory (OWLDataFactory): A data factory (inherited from `onto`) for manipulating axioms.
    """

    def __init__(self, onto: Ontology):
        """Initialise an ontology reasoner.

        Args:
            onto (Ontology): The input ontology to conduct reasoning on.
        """
        self.onto = onto
        self.owl_reasoner_factory = ReasonerFactory()
        self.owl_reasoner = self.owl_reasoner_factory.createReasoner(self.onto.owl_onto)
        self.owl_data_factory = self.onto.owl_data_factory

    def reload_reasoner(self):
        """Reload the reasoner for the current ontology (possibly changed)."""
        # release the memory
        self.owl_reasoner.dispose()
        # conduct reasoning on the possibly changed ontology
        self.owl_reasoner = self.owl_reasoner_factory.createReasoner(self.onto.owl_onto)

    @staticmethod
    def get_entity_type(entity: OWLObject, is_singular: bool = False):
        """A handy method to get the type of an entity (`OWLObject`).

        NOTE: This method is inherited from the Ontology Class.
        """
        return Ontology.get_entity_type(entity, is_singular)

    @staticmethod
    def has_iri(entity: OWLObject):
        """Check if an entity has an IRI."""
        try:
            entity.getIRI()
            return True
        except:
            return False

    def super_entities_of(self, entity: OWLObject, direct: bool = False):
        r"""Return the IRIs of super-entities of a given `OWLObject` according to the reasoner.

        A mixture of `getSuperClasses`, `getSuperObjectProperties`, `getSuperDataProperties`
        functions imported from the OWLAPI reasoner. The type of input entity will be
        automatically determined. The top entity such as `owl:Thing` is ignored.


        Args:
            entity (OWLObject): An `OWLObject` entity of interest.
            direct (bool, optional): Return parents (`direct=True`) or
                ancestors (`direct=False`). Defaults to `False`.

        Returns:
            (List[str]): A list of IRIs of the super-entities of the given `OWLObject` entity.
        """
        entity_type = self.get_entity_type(entity)
        get_super = f"getSuper{entity_type}"
        TOP = TOP_BOTTOMS[entity_type].TOP  # get the corresponding TOP entity
        super_entities = getattr(self.owl_reasoner, get_super)(entity, direct).getFlattened()
        super_entity_iris = [str(s.getIRI()) for s in super_entities]
        # the root node is owl#Thing
        if TOP in super_entity_iris:
            super_entity_iris.remove(TOP)
        return super_entity_iris

    def sub_entities_of(self, entity: OWLObject, direct: bool = False):
        """Return the IRIs of sub-entities of a given `OWLObject` according to the reasoner.

        A mixture of `getSubClasses`, `getSubObjectProperties`, `getSubDataProperties`
        functions imported from the OWLAPI reasoner. The type of input entity will be
        automatically determined. The bottom entity such as `owl:Nothing` is ignored.

        Args:
            entity (OWLObject): An `OWLObject` entity of interest.
            direct (bool, optional): Return parents (`direct=True`) or
                ancestors (`direct=False`). Defaults to `False`.

        Returns:
            (List[str]): A list of IRIs of the sub-entities of the given `OWLObject` entity.
        """
        entity_type = self.get_entity_type(entity)
        get_sub = f"getSub{entity_type}"
        BOTTOM = TOP_BOTTOMS[entity_type].BOTTOM
        sub_entities = getattr(self.owl_reasoner, get_sub)(entity, direct).getFlattened()
        sub_entity_iris = [str(s.getIRI()) for s in sub_entities]
        # the root node is owl#Thing
        if BOTTOM in sub_entity_iris:
            sub_entity_iris.remove(BOTTOM)
        return sub_entity_iris

    def check_subsumption(self, sub_entity: OWLObject, super_entity: OWLObject):
        """Check if the first entity is subsumed by the second entity according to the reasoner."""
        entity_type = self.get_entity_type(sub_entity, is_singular=True)
        assert entity_type == self.get_entity_type(super_entity, is_singular=True)

        sub_axiom = getattr(self.owl_data_factory, f"getOWLSub{entity_type}OfAxiom")(sub_entity, super_entity)

        return self.owl_reasoner.isEntailed(sub_axiom)

    def check_disjoint(self, entity1: OWLObject, entity2: OWLObject):
        """Check if two OWL class expressions are disjoint according to the reasoner."""
        entity_type = self.get_entity_type(entity1)
        assert entity_type == self.get_entity_type(entity2)

        disjoint_axiom = getattr(self.owl_data_factory, f"getOWLDisjoint{entity_type}Axiom")([entity1, entity2])

        return self.owl_reasoner.isEntailed(disjoint_axiom)

    def check_common_descendants(self, entity1: OWLObject, entity2: OWLObject):
        """Check if two entities have a common decendant.

        Entities can be **OWL class or property expressions**, and can be either **atomic
        or complex**. It takes longer computation time for the complex ones. Complex
        entities do not have an IRI. This method is optimised in the way that if
        there exists an atomic entity `A`, we compute descendants for `A` and
        compare them against the other entity which could be complex.
        """
        entity_type = self.get_entity_type(entity1)
        assert entity_type == self.get_entity_type(entity2)

        if not self.has_iri(entity1) and not self.has_iri(entity2):
            warnings.warn("Computing descendants for two complex entities is not efficient.")

        # `computed` is the one we compute the descendants
        # `compared` is the one we compare `computed`'s descendant one-by-one
        # we set the atomic entity as `computed` for efficiency if there is one
        computed, compared = entity1, entity2
        if not self.has_iri(entity1) and self.has_iri(entity2):
            computed, compared = entity2, entity1

        # for every inferred child of `computed`, check if it is subsumed by `compared``
        for descendant_iri in self.sub_entities_of(computed, direct=False):
            # print("check a subsumption")
            if self.check_subsumption(self.onto.get_owl_object_from_iri(descendant_iri), compared):
                return True
        return False

    def instances_of(self, owl_class: OWLClassExpression, direct: bool = False):
        """Return the list of named individuals that are instances of a given OWL class expression.

        Args:
            owl_class (OWLClassExpression): An ontology class of interest.
            direct (bool, optional): Return direct instances (`direct=True`) or
                also include the sub-classes' instances (`direct=False`). Defaults to `False`.

        Returns:
            (List[OWLNamedIndividual]): A list of named individuals that are instances of `owl_class`.
        """
        return list(self.owl_reasoner.getInstances(owl_class, direct).getFlattened())

    def check_instance(self, owl_instance: OWLNamedIndividual, owl_class: OWLClassExpression):
        """Check if a named individual is an instance of an OWL class."""
        assertion_axiom = self.owl_data_factory.getOWLClassAssertionAxiom(owl_class, owl_instance)
        return self.owl_reasoner.isEntailed(assertion_axiom)

    def check_common_instances(self, owl_class1: OWLClassExpression, owl_class2: OWLClassExpression):
        """Check if two OWL class expressions have a common instance.

        Class expressions can be **atomic or complex**, and it takes longer computation time
        for the complex ones. Complex classes do not have an IRI. This method is optimised
        in the way that if there exists an atomic class `A`, we compute instances for `A` and
        compare them against the other class which could be complex.

        !!! note "Difference with [`check_common_descendants`][deeponto.onto.OntologyReasoner.check_common_descendants]"
            The inputs of this function are restricted to **OWL class expressions**. This is because
            `descendant` is related to hierarchy and both class and property expressions have a hierarchy,
            but `instance` is restricted to classes.
        """

        if not self.has_iri(owl_class1) and not self.has_iri(owl_class2):
            warnings.warn("Computing instances for two complex classes is not efficient.")

        # `computed` is the one we compute the instances
        # `compared` is the one we compare `computed`'s descendant one-by-one
        # we set the atomic entity as `computed` for efficiency if there is one
        computed, compared = owl_class1, owl_class2
        if not self.has_iri(owl_class1) and self.has_iri(owl_class2):
            computed, compared = owl_class2, owl_class2

        # for every inferred instance of `computed`, check if it is subsumed by `compared``
        for instance in self.instances_of(computed, direct=False):
            if self.check_instance(instance, compared):
                return True
        return False

    def check_assumed_disjoint(self, owl_class1: OWLClassExpression, owl_class2: OWLClassExpression):
        r"""Check if two OWL class expressions satisfy the Assumed Disjointness.

        !!! credit "Paper"

            The definition of **Assumed Disjointness** comes from the paper:
            *[still-under-review](link)*.

        !!! note "Assumed Disjointness (Definition)"
            Two class expressions $C$ and $D$ are assumed to be disjoint if they meet the followings:

            1. By adding the disjointness axiom of them into the ontology, $C$ and $D$ are **still satisfiable**.
            2. $C$ and $D$ **do not have a common descendant** (otherwise $C$ and $D$ can be satisfiable but their
            common descendants become the bottom $\bot$.)

        Note that the special case where $C$ and $D$ are already disjoint is covered by the first check.
        The paper also proposed a practical alternative to decide Assumed Disjointness.
        See [`check_assumed_disjoint_alternative`][deeponto.onto.OntologyReasoner.check_assumed_disjoint_alternative].

        Examples:
            Suppose pre-load an ontology `onto` from the disease ontology file `doid.owl`.

            ```python
            >>> c1 = onto.get_owl_object_from_iri("http://purl.obolibrary.org/obo/DOID_4058")
            >>> c2 = onto.get_owl_object_from_iri("http://purl.obolibrary.org/obo/DOID_0001816")
            >>> onto.reasoner.check_assumed_disjoint(c1, c2)
            [SUCCESSFULLY] Adding the axiom DisjointClasses(<http://purl.obolibrary.org/obo/DOID_0001816> <http://purl.obolibrary.org/obo/DOID_4058>) into the ontology.
            [CHECK1 True] input classes are still satisfiable;
            [SUCCESSFULLY] Removing the axiom from the ontology.
            [CHECK2 False] input classes have NO common descendant.
            [PASSED False] assumed disjointness check done.
            False
            ```
        """
        # banner_message("Check Asssumed Disjointness")

        entity_type = self.get_entity_type(owl_class1)
        assert entity_type == self.get_entity_type(owl_class2)

        # adding the disjointness axiom of `class1`` and `class2``
        disjoint_axiom = getattr(self.owl_data_factory, f"getOWLDisjoint{entity_type}Axiom")([owl_class1, owl_class2])
        undo_change = self.onto.add_axiom(disjoint_axiom, return_undo=True)
        self.reload_reasoner()

        # check if they are still satisfiable
        still_satisfiable = self.owl_reasoner.isSatisfiable(owl_class1)
        still_satisfiable = still_satisfiable and self.owl_reasoner.isSatisfiable(owl_class2)
        print(f"[CHECK1 {still_satisfiable}] input classes are still satisfiable;")

        # remove the axiom and re-construct the reasoner
        undo_change_result = self.onto.owl_onto.applyChange(undo_change)
        print(f"[{str(undo_change_result)}] Removing the axiom from the ontology.")
        self.reload_reasoner()

        # failing first check, there is no need to do the second.
        if not still_satisfiable:
            print("Failed `satisfiability check`, skip the `common descendant` check.")
            print(f"[PASSED {still_satisfiable}] assumed disjointness check done.")
            return False

        # otherwise, the classes are still satisfiable and we should conduct the second check
        has_common_descendants = self.check_common_descendants(owl_class1, owl_class2)
        print(f"[CHECK2 {not has_common_descendants}] input classes have NO common descendant.")
        print(f"[PASSED {not has_common_descendants}] assumed disjointness check done.")
        return not has_common_descendants

    def check_assumed_disjoint_alternative(self, owl_class1: OWLClassExpression, owl_class2: OWLClassExpression):
        r"""Check if two OWL class expressions satisfy the Assumed Disjointness.

        !!! credit "Paper"

            The definition of **Assumed Disjointness** comes from the paper:
            *[still-under-review](link)*.

        The practical alternative version of [`check_assumed_disjoint`][deeponto.onto.OntologyReasoner.check_assumed_disjoint]
        with following conditions:


        !!! note "Assumed Disjointness (Practical Alternative)"
            Two class expressions $C$ and $D$ are assumed to be disjoint if they

            1. **do not** have a **subsumption relationship** between them,
            2. **do not** have a **common descendant** (in TBox),
            3. **do not** have a **common instance** (in ABox).

        If either of the conditions have been met, then we assume `class1` and `class2` as disjoint.

        Examples:
            Suppose pre-load an ontology `onto` from the disease ontology file `doid.owl`.

            ```python
            >>> c1 = onto.get_owl_object_from_iri("http://purl.obolibrary.org/obo/DOID_4058")
            >>> c2 = onto.get_owl_object_from_iri("http://purl.obolibrary.org/obo/DOID_0001816")
            >>> onto.reasoner.check_assumed_disjoint(c1, c2)
            [CHECK1 True] input classes have NO subsumption relationship;
            [CHECK2 False] input classes have NO common descendant;
            Failed the `common descendant check`, skip the `common instance` check.
            [PASSED False] assumed disjointness check done.
            False
            ```
            In this alternative implementation, we do no need to add and remove axioms which will then
            be time-saving.
        """
        # banner_message("Check Asssumed Disjointness (Alternative)")

        # # Check for entailed disjointness (short-cut)
        # if self.check_disjoint(owl_class1, owl_class2):
        #     print(f"Input classes are already entailed as disjoint.")
        #     return True

        # Check for entailed subsumption,
        # common descendants and common instances

        has_subsumption = self.check_subsumption(owl_class1, owl_class2)
        has_subsumption = has_subsumption or self.check_subsumption(owl_class2, owl_class1)
        print(f"[CHECK1 {not has_subsumption}] input classes have NO subsumption relationship;")
        if has_subsumption:
            print("Failed the `subsumption check`, skip the `common descendant` check.")
            print(f"[PASSED {not has_subsumption}] assumed disjointness check done.")
            return False

        has_common_descendants = self.check_common_descendants(owl_class1, owl_class2)
        print(f"[CHECK2 {not has_common_descendants}] input classes have NO common descendant;")
        if has_common_descendants:
            print("Failed the `common descendant check`, skip the `common instance` check.")
            print(f"[PASSED {not has_common_descendants}] assumed disjointness check done.")
            return False

        # TODO: `check_common_instances` is still experimental because we have not tested it with ontologies of rich ABox.
        has_common_instances = self.check_common_instances(owl_class1, owl_class2)
        print(f"[CHECK3 {not has_common_instances}] input classes have NO common instance;")
        print(f"[PASSED {not has_common_instances}] assumed disjointness check done.")
        return not has_common_instances
