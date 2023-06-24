package org.mowl.Projectors

// OWL API imports
import org.semanticweb.owlapi.model._
import org.semanticweb.owlapi.apibinding.OWLManager
import org.semanticweb.owlapi.model.parameters.Imports
import uk.ac.manchester.cs.owl.owlapi._

import org.semanticweb.owlapi.reasoner.InferenceType

import org.semanticweb.owlapi.util._
// Java imports
import java.io.File


import collection.JavaConverters._
import org.mowl.Types._
import org.mowl.Utils._

class DL2VecProjector(var bidirectional_taxonomy: Boolean = false) extends AbstractProjector{

  val quantityModifiers = List("ObjectSomeValuesFrom", "ObjectAllValuesFrom", "ObjectMaxCardinality", "ObjectMinCardinality")

  val collectors = List("ObjectIntersectionOf", "ObjectUnionOf")

  @Override
  def project(ontology: OWLOntology, withIndividuals: Boolean, verbose: Boolean) = {
    val imports = Imports.fromBoolean(true)
    val axioms = ontology.getAxioms(imports).asScala.toList

    val edges = axioms.map(projectAxiom(_, withIndividuals, verbose)).flatten
    edges.asJava
  }

  def projectAxiom(ontClass: OWLClass, axiom: OWLClassAxiom): List[Triple] = {Nil}
  def projectAxiom(ontClass: OWLClass, axiom: OWLClassAxiom, ontology: OWLOntology): List[Triple] = {Nil}
  def projectAxiom(axiom: OWLClassAxiom): List[Triple] = {Nil}
  def projectAxiom(axiom: OWLAxiom): List[Triple] = {Nil}
  def projectAxiom(axiom: OWLAxiom, withIndividuals: Boolean, verbose: Boolean): List[Triple] = {

    val axiomType = axiom.getAxiomType().getName()

    axiomType match {

      case "SubClassOf" => {
	var ax = axiom.asInstanceOf[OWLSubClassOfAxiom]
        ax.getSubClass.isInstanceOf[OWLClass] match {
          case true => 	projectSubClassOrEquivAxiom(ax.getSubClass.asInstanceOf[OWLClass], ax.getSuperClass, "http://subclassof")
          case false => Nil}
      }
      case "EquivalentClasses" => {
	var ax = axiom.asInstanceOf[OWLEquivalentClassesAxiom].getClassExpressionsAsList.asScala
        ax.find(_.isInstanceOf[OWLClass]) match {
          case Some(head) => {
            val rightSide = ax.filter((x) => x != head)
      	    rightSide.toList.flatMap(projectSubClassOrEquivAxiom(head.asInstanceOf[OWLClass], _:OWLClassExpression, "http://equivalentto")) }
          case None => Nil
        }
      }
      case "ClassAssertion" => {
        withIndividuals match {
          case true => {
            val ax = axiom.asInstanceOf[OWLClassAssertionAxiom]
            projectClassAssertionAxiom(ax, verbose)
          }
          case false => Nil
        }
      }
      case "ObjectPropertyAssertion" => {
        withIndividuals match {
          case true => {
            val ax = axiom.asInstanceOf[OWLObjectPropertyAssertionAxiom]
            projectObjectPropertyAssertionAxiom(ax, verbose)
          }
          case false => Nil
        }
      }
      case "Declaration" => {
        Nil
      }
      case _ => {
        if (verbose) println("WARNING: Axiom " + axiom + " not supported")
        Nil
      }

    }
  }

  def projectClassAssertionAxiom(axiom: OWLClassAssertionAxiom,  verbose: Boolean): List[Triple] = {
    val subject = axiom.getIndividual
    val ontClass = axiom.getClassExpression

    ontClass.getClassExpressionType.getName match {
      case "OWLClass" => {
        val triple = new Triple(subject.toStringID, "http://type", ontClass.asInstanceOf[OWLClass].toStringID)
        List(triple)
      }
      case "ObjectSomeValuesFrom" => {
        val ontClassExistential = ontClass.asInstanceOf[OWLObjectSomeValuesFrom]
        val property = ontClassExistential.getProperty.asInstanceOf[OWLObjectProperty]
        val filler = ontClassExistential.getFiller

        filler.getClassExpressionType.getName match {
          case "Class" => {
            val triple = new Triple(subject.toStringID, property.toStringID, filler.asInstanceOf[OWLClass].toStringID)
            List(triple)
          }
          case _ => {
            if (verbose) println("WARNING: Assertion axiom, filler " + filler + " in object not supported")
            Nil
          }
        }
      }
      case _ => {
        if (verbose) println("WARNING: Assertion axiom, object " + ontClass + " not supported")
        Nil
      }
    }
  }

  def projectObjectPropertyAssertionAxiom(axiom: OWLObjectPropertyAssertionAxiom, verbose: Boolean): List[Triple] = {
    val subject = axiom.getSubject
    val property = axiom.getProperty.asInstanceOf[OWLObjectProperty]
    val object_ = axiom.getObject



    val triple = new Triple(subject.toStringID, property.toStringID, object_.toStringID)
    List(triple)
  }

  def projectSubClassOrEquivAxiom(ontClass: OWLClass, superClass: OWLClassExpression, relName: String): List[Triple] = {
    var invRelName = ""

    if (relName == "http://subclassof"){
      invRelName = "http://superclassof"
    }else if(relName == "http://equivalentto"){
      invRelName = relName
    }
     
    val superClassType = superClass.getClassExpressionType.getName

    superClassType match {

      case m if (quantityModifiers contains m) => {
        val superClass_ = lift2QuantifiedExpression(superClass)
        val (relations, dstClass) = projectQuantifiedExpression(superClass_, Nil)
        val dstClasses = splitClass(dstClass)

        for (
          rel <- relations;
          dst <- dstClasses.filter(_.getClassExpressionType.getName == "Class").map(_.asInstanceOf[OWLClass])
        ) yield new Triple(ontClass, rel, dst)
      }

     //case c if (collectors contains c) => {
     //   val dstClasses = splitClass(superClass)
     //   dstClasses.flatMap(projectSubClassOrEquivAxiom(ontClass, _:OWLClassExpression, "http://subclassof"))
     // }

      case "Class" => {
	val dst = superClass.asInstanceOf[OWLClass]
        if (bidirectional_taxonomy){
	  new Triple(ontClass, relName, dst) :: new Triple(dst, invRelName, ontClass) :: Nil
        }else{
          new Triple(ontClass, relName, dst) :: Nil
        }
      }
      case _ => Nil
    }
   }

  def projectQuantifiedExpression(expr:QuantifiedExpression, relations:List[String]): (List[String], OWLClassExpression) = {
    val rel = expr.getProperty.asInstanceOf[OWLObjectProperty]
    val relName = rel.toStringID
    var filler = expr.getFiller

    defineQuantifiedExpression(filler) match {
      case Some(expr) => projectQuantifiedExpression(expr, relName::relations)
      case None => (relName::relations, filler)
    }
  }

  def splitClass(classExpr:OWLClassExpression): List[OWLClassExpression] = {
    val exprType = classExpr.getClassExpressionType.getName

    exprType match {
      case "Class" => classExpr.asInstanceOf[OWLClass] :: Nil
      case m if (quantityModifiers contains m) => classExpr :: Nil

      case "ObjectIntersectionOf" => {
        val classExprInt = classExpr.asInstanceOf[OWLObjectIntersectionOf]
        val operands = classExprInt.getOperands.asScala.toList
        operands.flatMap(splitClass(_: OWLClassExpression))
      }

      case "ObjectUnionOf" => {
        val classExprUnion = classExpr.asInstanceOf[OWLObjectUnionOf]
        val operands = classExprUnion.getOperands.asScala.toList
        operands.flatMap(splitClass(_: OWLClassExpression))
      }
      case _ => Nil
    }
  }


}
