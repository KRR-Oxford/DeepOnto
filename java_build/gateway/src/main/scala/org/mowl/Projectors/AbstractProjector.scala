package org.mowl.Projectors

// OWL API imports
import org.semanticweb.owlapi.model._
import org.semanticweb.owlapi.apibinding.OWLManager
import org.semanticweb.owlapi.model.parameters.Imports


// Java imports
import collection.JavaConverters._

import org.mowl.Types._

trait AbstractProjector{

  val ontManager = OWLManager.createOWLOntologyManager()
  val dataFactory = ontManager.getOWLDataFactory()
  val imports = Imports.fromBoolean(true)

 
  def project(ontology: OWLOntology) = {
    val imports = Imports.fromBoolean(true)

    val ontClasses = ontology.getClassesInSignature(imports).asScala.toList
    printf("INFO: Number of ontology classes: %d\n", ontClasses.length)

    val edges = ontClasses.foldLeft(List[Triple]()){(acc, x) => acc ::: processOntClass(x, ontology)}

    edges.asJava
  }

  //Abstract methods
  def project(ontology: OWLOntology, withIndividuals: Boolean, verbose: Boolean): java.util.List[Triple]

  def projectAxiom(ontClass: OWLClass, axiom: OWLClassAxiom): List[Triple]
  def projectAxiom(ontClass: OWLClass, axiom: OWLClassAxiom, ontology: OWLOntology): List[Triple]
  def projectAxiom(axiom: OWLClassAxiom): List[Triple]
  def projectAxiom(axiom: OWLAxiom): List[Triple]
  def projectAxiom(axiom: OWLAxiom, withIndividuals: Boolean, verbose: Boolean): List[Triple]
  //////////////////////

  def processOntClass(ontClass: OWLClass, ontology: OWLOntology): List[Triple] = {
    val axioms = ontology.getAxioms(ontClass, imports).asScala.toList
    val edges = axioms.flatMap(projectAxiom(ontClass, _: OWLClassAxiom))
    edges
  }

}
