package org.mowl

import org.semanticweb.owlapi.model._
import uk.ac.manchester.cs.owl.owlapi.OWLClassImpl

object Types {

    val Top = new OWLClassImpl(IRI.create("http://www.w3.org/2002/07/owl#Thing")).asOWLClass
    val Bottom = new OWLClassImpl(IRI.create("http://www.w3.org/2002/07/owl#Nothing")).asOWLClass

    type GOClass = String
    type Relation = String


    sealed trait QuantifiedExpression {
      def getProperty(): OWLObjectPropertyExpression
      def getFiller(): OWLClassExpression
      def unlift():OWLClassExpression
    }
   
    case class Universal(val expression: OWLObjectAllValuesFrom) extends QuantifiedExpression{
        def getProperty() = expression.getProperty
      def getFiller() = expression.getFiller
      def unlift() = expression.asInstanceOf[OWLClassExpression]
    }

    case class Existential(val expression: OWLObjectSomeValuesFrom) extends QuantifiedExpression{
        def getProperty() = expression.getProperty
        def getFiller() = expression.getFiller
        def unlift() = expression.asInstanceOf[OWLClassExpression]

    }
    case class MinCardinality(val expression: OWLObjectMinCardinality) extends QuantifiedExpression{
        def getProperty() = expression.getProperty
        def getFiller() = expression.getFiller
        def unlift() = expression.asInstanceOf[OWLClassExpression]
    }
   case class MaxCardinality(val expression: OWLObjectMaxCardinality) extends QuantifiedExpression{
        def getProperty() = expression.getProperty
        def getFiller() = expression.getFiller
        def unlift() = expression.asInstanceOf[OWLClassExpression]
   }


    sealed trait Expression
    case object GOClass extends Expression
    case class ObjectSomeValuesFrom(rel: Relation, expr: Expression) extends Expression

    sealed trait Axiom
    case class SubClassOf(subClass: GOClass, superClass: Expression) extends Axiom
    case class Equivalent(leftSide: GOClass, rightSide: List[Expression])


    class Triple(val src:GOClass, val rel:Relation, val dst:GOClass){
        def this(src: OWLClass, rel: Relation, dst: OWLClass) = this(goClassToStr(src), removeBrackets(rel), goClassToStr(dst))
        def this(src: String, rel: Relation, dst: OWLClass) = this(src, removeBrackets(rel), goClassToStr(dst))
        def this(src: OWLClass, rel: Relation, dst: String) = this(goClassToStr(src), removeBrackets(rel), dst)
    }

  def goClassToStr(goClass: OWLClass) = removeBrackets(goClass.toStringID)

  def annotationSubject2Str(subject: OWLAnnotationSubject): String = subject.toString

  def getNodes(triples: List[Triple]) = {
        
        val triples_sc = triples
        val srcs = triples_sc.map((e) => e.src)
        val dsts = triples_sc.map((e) => e.dst)
        (srcs ::: dsts).toSet
    }

  def removeBrackets(name: String): String = {
    if (name.startsWith("<")) name.tail.init
    else name
  }

}
