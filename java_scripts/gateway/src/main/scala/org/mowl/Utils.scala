package org.mowl

import org.semanticweb.owlapi.model._
import org.mowl.Types._

object Utils {

  def defineQuantifiedExpression(classExpression: OWLClassExpression): Option[QuantifiedExpression] = {

    val expressionType = classExpression.getClassExpressionType().getName()

    expressionType match {
      case "ObjectSomeValuesFrom" => Some(Existential(classExpression.asInstanceOf[OWLObjectSomeValuesFrom]))
      case "ObjectAllValuesFrom" => Some(Universal(classExpression.asInstanceOf[OWLObjectAllValuesFrom]))
      case "ObjectMinCardinality" => Some(MinCardinality(classExpression.asInstanceOf[OWLObjectMinCardinality]))
      case "ObjectMaxCardinality" => Some(MaxCardinality(classExpression.asInstanceOf[OWLObjectMaxCardinality]))
      case _ => None
    }
  }

  //Unsafe version of defineQuantifiedExpression method.
  def lift2QuantifiedExpression(classExpression: OWLClassExpression): QuantifiedExpression = {
    val expressionType = classExpression.getClassExpressionType().getName()
    expressionType match {
      case "ObjectSomeValuesFrom" => Existential(classExpression.asInstanceOf[OWLObjectSomeValuesFrom])
      case "ObjectAllValuesFrom" => Universal(classExpression.asInstanceOf[OWLObjectAllValuesFrom])
      case "ObjectMinCardinality" => MinCardinality(classExpression.asInstanceOf[OWLObjectMinCardinality])
      case "ObjectMaxCardinality" => MaxCardinality(classExpression.asInstanceOf[OWLObjectMaxCardinality])
    }
  }
}
