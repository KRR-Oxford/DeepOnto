package org.mowl.Projectors

// OWL API imports
import org.semanticweb.owlapi.model._
import org.semanticweb.owlapi.apibinding.OWLManager
import org.semanticweb.owlapi.model.parameters.Imports
import uk.ac.manchester.cs.owl.owlapi._

import org.semanticweb.owlapi.reasoner.InferenceType

import org.semanticweb.owlapi.util._
import org.semanticweb.owlapi.search._
// Java imports
import java.io.File
import java.util
import scala.collection.mutable.ListBuffer
import collection.JavaConverters._
import org.mowl.Types._
import org.mowl.Utils._

class OWL2VecStarProjector(
  var bidirectional_taxonomy: Boolean,
  var only_taxonomy: Boolean,
  var include_literals: Boolean
  // var avoid_properties: java.util.HashSet[String],
  // var additional_preferred_labels_annotations: java.util.HashSet[String],
  // var additional_synonyms_annotations: java.util.HashSet[String],
  // var memory_reasoner: String = "10240"
) extends AbstractProjector{

  
  val searcher = new EntitySearcher()
  var subRoles = scala.collection.mutable.Map[String, List[String]]()
  var inverseRoles = scala.collection.mutable.Map[String, String]()

  override def project(ontology: OWLOntology) = {

    var edgesFromObjectProperties = List[Triple]()
    val tboxAxioms = ontology.getTBoxAxioms(imports).asScala.toList
    val aboxAxioms = ontology.getABoxAxioms(imports).asScala.toList
    val rboxAxioms = ontology.getRBoxAxioms(imports).asScala.toList


    var classAssertionAxiom = ListBuffer[OWLClassAssertionAxiom]()
    var objectPropertyAssertionAxiom = ListBuffer[OWLObjectPropertyAssertionAxiom]()
    var subclassOfAxioms = ListBuffer[OWLSubClassOfAxiom]()
    var equivalenceAxioms = ListBuffer[OWLEquivalentClassesAxiom]()
    var annotationAxioms = ListBuffer[OWLAnnotationAssertionAxiom]()
    var domainAxioms = ListBuffer[OWLObjectPropertyDomainAxiom]()
    var rangeAxioms = ListBuffer[OWLObjectPropertyRangeAxiom]()
    var otherAxioms = ListBuffer[OWLAxiom]()

    // dictionary of roles and subroles

    for (axiom <- rboxAxioms) {
      axiom match {
        case axiom: OWLObjectPropertyDomainAxiom => {
          domainAxioms += axiom
        }
        case axiom: OWLObjectPropertyRangeAxiom => {
          rangeAxioms += axiom
        }
        case axiom: OWLSubObjectPropertyOfAxiom => {
          val subProperty = axiom.getSubProperty()
          val superProperty = axiom.getSuperProperty()

          if (subProperty.isInstanceOf[OWLObjectProperty] && superProperty.isInstanceOf[OWLObjectProperty]){
            val subPropertyStr = subProperty.asInstanceOf[OWLObjectProperty].toStringID()
            val superPropertyStr = superProperty.asInstanceOf[OWLObjectProperty].toStringID()
            subRoles += (superPropertyStr -> (subPropertyStr :: subRoles.getOrElse(subPropertyStr, List())))
          }
        }
        case axiom: OWLInverseObjectPropertiesAxiom => {
          val firstProperty = axiom.getFirstProperty()
          val secondProperty = axiom.getSecondProperty()

          if (firstProperty.isInstanceOf[OWLObjectProperty] && secondProperty.isInstanceOf[OWLObjectProperty]){
            val firstPropertyStr = firstProperty.asInstanceOf[OWLObjectProperty].toStringID()
            val secondPropertyStr = secondProperty.asInstanceOf[OWLObjectProperty].toStringID()
            inverseRoles += (firstPropertyStr -> secondPropertyStr)
            inverseRoles += (secondPropertyStr -> firstPropertyStr)

          }
        }
        case _ => {
          otherAxioms += axiom
        }

      }
    }

    println("subRoles: " + subRoles)
    println("inverseRoles: " + inverseRoles)

    val axioms = tboxAxioms ++ aboxAxioms ++ rboxAxioms

    for (axiom <- axioms){

      axiom.getAxiomType.getName match {
        case "SubClassOf" => subclassOfAxioms += axiom.asInstanceOf[OWLSubClassOfAxiom]
        case "AnnotationAssertion" => {
          if (include_literals)
          annotationAxioms += axiom.asInstanceOf[OWLAnnotationAssertionAxiom]
        }
        case "EquivalentClasses" => equivalenceAxioms += axiom.asInstanceOf[OWLEquivalentClassesAxiom]
        case "ClassAssertion" => classAssertionAxiom += axiom.asInstanceOf[OWLClassAssertionAxiom]
        case "ObjectPropertyAssertion" => objectPropertyAssertionAxiom += axiom.asInstanceOf[OWLObjectPropertyAssertionAxiom]
        case "ObjectPropertyDomain" => domainAxioms += axiom.asInstanceOf[OWLObjectPropertyDomainAxiom]
        case "ObjectPropertyRange" => rangeAxioms += axiom.asInstanceOf[OWLObjectPropertyRangeAxiom]
        case _ => {
          //println(axiom)
          otherAxioms += axiom
        }
      }
    }

    val subclassOfTriples = subclassOfAxioms.flatMap(x => processSubClassAxiom(x.getSubClass, x.getSuperClass, ontology))
    val equivalenceTriples = equivalenceAxioms.flatMap(
      x => {
        val subClass::superClass::rest= x.getClassExpressionsAsList.asScala.toList
        superClass.getClassExpressionType.getName match{
          case "Class" => processSubClassAxiom(subClass, superClass, ontology)
          case "ObjectIntersectionOf" => superClass.asInstanceOf[OWLObjectIntersectionOf].getOperands.asScala.toList.flatMap(processSubClassAxiom(subClass, _, ontology))
          case "ObjectUnionOf" => superClass.asInstanceOf[OWLObjectUnionOf].getOperands.asScala.toList.flatMap(processSubClassAxiom(subClass, _, ontology))
          case _ => Nil
        }
      }
    )
    val annotationTriples = annotationAxioms.map(processAnnotationAxiom(_)).flatten
    val classAssertionTriples = classAssertionAxiom.map(processClassAssertionAxiom(_)).flatten
    val objectPropertyAssertionTriples = objectPropertyAssertionAxiom.map(processObjectPropertyAssertionAxiom(_, ontology)).flatten
    val domainAndRangeTriples = processDomainAndRangeAxioms(domainAxioms, rangeAxioms, ontology)
    (subclassOfTriples.toList ::: equivalenceTriples.toList ::: annotationTriples.toList ::: classAssertionTriples.toList ::: objectPropertyAssertionTriples.toList ::: domainAndRangeTriples.toList).distinct.asJava
  }

  // CLASSES PROCESSING

  override def processOntClass(ontClass: OWLClass, ontology: OWLOntology): List[Triple] = {

    var annotationEdges = List[Triple]()

    if (include_literals){ //ANNOTATION PROCESSSING
      val annotProperties = ontClass.getAnnotationPropertiesInSignature.asScala.toList
      val annotationAxioms = ontology.getAnnotationAssertionAxioms(ontClass.getIRI).asScala.toList
      annotationEdges = annotationAxioms.map(annotationAxiom2Edge).flatten
    }

    val axioms = ontology.getAxioms(ontClass, imports).asScala.toList
    val edges = axioms.flatMap(projectAxiom(ontClass, _: OWLClassAxiom))
    edges ::: annotationEdges
  }



  def projectAxiom(ontClass: OWLClass, axiom: OWLClassAxiom, ontology: OWLOntology): List[Triple] = {

    val axiomType = axiom.getAxiomType().getName()

    axiomType match {
      case "SubClassOf" => {
	var ax = axiom.asInstanceOf[OWLSubClassOfAxiom]
	projectSubClassOrEquivAxiom(ax.getSubClass.asInstanceOf[OWLClass], ax.getSuperClass, ontology)
      }
      case "EquivalentClasses" => {
	var ax = axiom.asInstanceOf[OWLEquivalentClassesAxiom].getClassExpressionsAsList.asScala
        assert(ontClass == ax.head)
        val rightSide = ax.tail
	projectSubClassOrEquivAxiom(ontClass, new OWLObjectIntersectionOfImpl(rightSide.toSet.asJava), ontology)
      }
      case _ => Nil
    }
  }

  def processSubClassAxiom(subClass: OWLClassExpression, superClass: OWLClassExpression, ontology: OWLOntology): List[Triple] = {

    val firstCase = processSubClassAxiomComplexSubClass(subClass, superClass, ontology)

    if (firstCase == Nil){
      processSubClassAxiomComplexSuperClass(subClass, superClass, ontology)
    }else{
      firstCase
    }
  }

  def processSubClassAxiomComplexSubClass(subClass: OWLClassExpression, superClass: OWLClassExpression, ontology: OWLOntology): List[Triple] = {
    // When subclass is complex, superclass must be atomic

    val quantityModifiers = List("ObjectSomeValuesFrom", "ObjectAllValuesFrom", "ObjectMaxCardinality", "ObjectMinCardinality")
    val superClassType = superClass.getClassExpressionType.getName

    if (superClassType != "Class") {
      Nil
    }else{

      val superClass_ = superClass.asInstanceOf[OWLClass]
      val subClassType = subClass.getClassExpressionType.getName

      subClassType match {

        case m if (quantityModifiers contains m) && !only_taxonomy => {

	  val subClass_ = lift2QuantifiedExpression(subClass)
          projectQuantifiedExpression(subClass_, ontology) match {

            case Some((rel, inverseRels, subRels, dstClass)) => {
              val dstClasses = splitClass(dstClass)

              val outputEdges = for (dst <- dstClasses)
              yield new Triple(superClass_, rel, dst) :: subRels.map(x => new Triple(superClass_, x, dst)) ::: inverseRels.map(x => new Triple(dst, x, superClass_))
              outputEdges.flatten
            }
            case None => Nil
          }
        }
        case _ => Nil
      }
    }
  }


  def processSubClassAxiomComplexSuperClass(subClass: OWLClassExpression, superClass: OWLClassExpression, ontology: OWLOntology): List[Triple] = {

    // When superclass is complex, subclass must be atomic

    val subClassType = subClass.getClassExpressionType.getName
    if (subClassType != "Class"){
      Nil
    }else{
      val subClass_  = subClass.asInstanceOf[OWLClass]
      val quantityModifiers = List("ObjectSomeValuesFrom", "ObjectAllValuesFrom", "ObjectMaxCardinality", "ObjectMinCardinality")
      val superClassType = superClass.getClassExpressionType.getName

      superClassType match {

        case m if (quantityModifiers contains m) && !only_taxonomy => {
	  val superClass_ = lift2QuantifiedExpression(superClass)
          projectQuantifiedExpression(superClass_, ontology) match {

            case Some((rel, inverseRels, subRels, dstClass)) => {
              val dstClasses = splitClass(dstClass)

              val outputEdges = for (dst <- dstClasses)
              yield new Triple(subClass_, rel, dst) :: subRels.map(new Triple(subClass_, _, dst)) ::: inverseRels.map(new Triple(dst, _, subClass_))
              outputEdges.flatten
            }
            case None => Nil
          }
        }
        case "Class" => {
	  val dst = superClass.asInstanceOf[OWLClass]
          if (bidirectional_taxonomy){
	    new Triple(subClass_, "http://subclassof", dst) :: new Triple(dst, "http://superclassof", subClass_) :: Nil
          }else{
            new Triple(subClass_, "http://subclassof", dst) :: Nil
          }
        }
        case _ => Nil
      }
    }
  }



  def processAnnotationAxiom(axiom: OWLAnnotationAssertionAxiom): Option[Triple]= {
    val property = stripValue(axiom.getProperty.toString)

    property match {
      case m if (lexicalAnnotationURIs contains m) => {
        val subject = axiom.getSubject.toString
        val value = axiom.getValue
       
        val valueStr = value.isLiteral match {
          case true => {
            val datatype = value.asLiteral.get.getDatatype

            if (datatype.isString) value.asInstanceOf[OWLLiteralImplString].getLiteral
            else if(datatype.isRDFPlainLiteral) value.asInstanceOf[OWLLiteralImplPlain].getLiteral
            else {
              println("Warning: datatype not detected: ", datatype)
              stripValue(axiom.getValue.toString)
            } 
          }
          case false => stripValue(axiom.getValue.toString)
        }
        Some(new Triple(subject, m, valueStr))
      }
      case _ => {
        //println("C ",property)
        None
      }
    }
  }

  def projectSubClassOrEquivAxiom(ontClass: OWLClass, superClass: OWLClassExpression, ontology: OWLOntology): List[Triple] = {

    val quantityModifiers = List("ObjectSomeValuesFrom", "ObjectAllValuesFrom", "ObjectMaxCardinality", "ObjectMinCardinality")
    val superClassType = superClass.getClassExpressionType.getName

    superClassType match {
      case m if (quantityModifiers contains m) && !only_taxonomy => {
	val superClass_ = lift2QuantifiedExpression(superClass)
        projectQuantifiedExpression(superClass_, ontology) match {

          case Some((rel, inverseRels, subRels, dstClass)) => {
            val dstClasses = splitClass(dstClass)
            val outputEdges = for (dst <- dstClasses)
            yield new Triple(ontClass, rel, dst) :: subRels.map(new Triple(ontClass, _, dst)) ::: inverseRels.map(new Triple(dst, _, ontClass))
            outputEdges.flatten
          }
          case None => Nil
        }
      }

      case "Class" => {
	val dst = superClass.asInstanceOf[OWLClass]
        if (bidirectional_taxonomy){
	  new Triple(ontClass, "http://subclassof", dst) :: new Triple(dst, "http://superclassof", ontClass) :: Nil
        }else{
          new Triple(ontClass, "http://subclassof", dst) :: Nil
        }
      }
      case _ => Nil
    }
  }

  def projectQuantifiedExpression(expr:QuantifiedExpression, ontology: OWLOntology): Option[(String, List[String], List[String], OWLClassExpression)] = {

    val rel = expr.getProperty.asInstanceOf[OWLObjectProperty]
    val relName = rel.toStringID
    val rel_ = Left(rel)
    val inverseRelName = getRelationInverseNames(rel_, ontology)
    val subRoleNames = getSubRelationNames(rel_, ontology)
    val filler = expr.getFiller
    val fillerType = filler.getClassExpressionType.getName

    fillerType match {
      case "Class" => Some((relName, inverseRelName, subRoleNames,
        filler.asInstanceOf[OWLClass]))
      case _ => None
    }
  }

  def splitClass(classExpr:OWLClassExpression): List[OWLClass] = {
    val exprType = classExpr.getClassExpressionType.getName

    exprType match {
      case "Class" => classExpr.asInstanceOf[OWLClass] :: Nil
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

  ////////////////////////////////////////////////////
  ////////////////////////////////////////////////////

  //OBJECT PROPERTIES PROCESSING

  def processObjectProperty(property: OWLObjectProperty): List[Triple] = {
    Nil
  }


  def getRelationInverseNames(relation: Either[OWLObjectProperty, String], ontology: OWLOntology): List[String] = {
    val relName = relation match {
      case Left(r) => r.toStringID
      case Right(r) => r
    }

    if (inverseRoles.contains(relName)){
      List(inverseRoles(relName))
    }else{
      Nil
    }
  }

  def getSubRelationNames(relation: Either[OWLObjectProperty, String], ontology: OWLOntology): List[String] = {
    val relName = relation match {
      case Left(r) => r.toStringID
      case Right(r) => r
    }

    if (subRoles.contains(relName)){
      subRoles(relName)
    }else{
      Nil
    }
  }

  def stripBracket(value: OWLObjectPropertyExpression) = {
    val valueStr = value.toString

    valueStr.head match {
      case '<' => valueStr.tail.init
      case _ => valueStr
    }

  }

  //////////////////////////////////////////////////
  //////////////////////////////////////////////////

  //ANNOTATION PROPERTIES PROCESSING

  val mainLabelURIs = List(
    "http://www.w3.org/2000/01/rdf-schema#label",
    "http://www.w3.org/2004/02/skos/core#prefLabel",
    "rdfs:label",
    "rdfs:comment",
    "http://purl.obolibrary.org/obo/IAO_0000111",
    "http://purl.obolibrary.org/obo/IAO_0000589"
  )
  
  val synonymLabelURIs = List(
    "http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym",
    "http://www.geneontology.org/formats/oboInOwl#hasExactSynonym",
    "http://www.geneontology.org/formats/oboInOWL#hasExactSynonym",
    "http://www.geneontology.org/formats/oboInOwl#hasRelatedSynonym",
    "http://purl.bioontology.org/ontology/SYN#synonym",
    "http://scai.fraunhofer.de/CSEO#Synonym",
    "http://purl.obolibrary.org/obo/synonym",
    "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#FULL_SYN",
    "http://www.ebi.ac.uk/efo/alternative_term",
    "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#Synonym",
    "http://bioontology.org/projects/ontologies/fma/fmaOwlDlComponent_2_0#Synonym",
    "http://www.geneontology.org/formats/oboInOwl#hasDefinition",
    "http://bioontology.org/projects/ontologies/birnlex#preferred_label",
    "http://bioontology.org/projects/ontologies/birnlex#synonyms",
    "http://www.w3.org/2004/02/skos/core#altLabel",
    "https://cfpub.epa.gov/ecotox#latinName",
    "https://cfpub.epa.gov/ecotox#commonName",
    "https://www.ncbi.nlm.nih.gov/taxonomy#scientific_name",
    "https://www.ncbi.nlm.nih.gov/taxonomy#synonym",
    "https://www.ncbi.nlm.nih.gov/taxonomy#equivalent_name",
    "https://www.ncbi.nlm.nih.gov/taxonomy#genbank_synonym",
    "https://www.ncbi.nlm.nih.gov/taxonomy#common_name",
    "http://purl.obolibrary.org/obo/IAO_0000118"
  )

  val lexicalAnnotationURIs = mainLabelURIs ::: synonymLabelURIs :::  List(    
    "http://www.w3.org/2000/01/rdf-schema#comment",
    "http://www.geneontology.org/formats/oboInOwl#hasDbXref",
    "http://purl.org/dc/elements/1.1/description",
    "http://purl.org/dc/terms/description",
    "http://purl.org/dc/elements/1.1/title",
    "http://purl.org/dc/terms/title",    
    "http://purl.obolibrary.org/obo/IAO_0000115",        
    "http://purl.obolibrary.org/obo/IAO_0000600",        
    "http://purl.obolibrary.org/obo/IAO_0000602",
    "http://purl.obolibrary.org/obo/IAO_0000601",
    "http://www.geneontology.org/formats/oboInOwl#hasOBONamespace"
  )

  val excludedAnnotationProperties = List("http://www.geneontology.org/formats/oboInOwl#inSubset", "'http://www.geneontology.org/formats/oboInOwl#id", "http://www.geneontology.org/formats/oboInOwl#hasAlternativeId") //List("rdfs:comment", "http://www.w3.org/2000/01/rdf-schema#comment")

  def annotationAxiom2Edge(annotationAxiom: OWLAnnotationAssertionAxiom): Option[Triple] = {

    val property = annotationAxiom.getProperty.toStringID.toString

    property match {
      case m if true || (lexicalAnnotationURIs contains m) =>  {
        val subject = annotationAxiom.getSubject.toString
        val value = annotationAxiom.getValue
          Some(new Triple(subject, m, stripValue(value.toString)))
      }
      case _ => {
        println("C ",property)
        None
      }
    }
  }

  def stripValue(valueStr: String) = {

    val value = valueStr.replaceAll("\\\\", "")
    value.head match {
      case '"' => value.tail.init
      case '<' => value.tail.init
      case _ => value
    }
  }

  def processAnnotationProperty(annotProperty: OWLAnnotationProperty, ontology: OWLOntology): List[Triple] = {

    val annotations = EntitySearcher.getAnnotations(annotProperty, ontology).asScala.toList
    val property = annotProperty.toStringID.toString

    property match {
      case m if true || (lexicalAnnotationURIs contains m) => {
        val axioms = ontology.getAnnotationAssertionAxioms(annotProperty.getIRI).asScala.toList
        val axx = axioms.map(annotationAxiom2Edge).flatten
        axx
      }

      case _ => {
        println(property)
        Nil
      }
    }
  }


  //////////////////////////////////////////////////
  // Process assertion axioms

  def processClassAssertionAxiom(axiom: OWLClassAssertionAxiom): List[Triple] = {

    val subject = axiom.getIndividual.asInstanceOf[OWLNamedIndividual].toStringID
    val predicate = "http://type"

    val obj = axiom.getClassExpression

    obj match {
      case c: OWLClass => {
        val objectStr = c.toStringID
        List(new Triple(subject, predicate, objectStr))
      }
      case _ => {
        println("Class assertion axiom not handled: ", axiom)
        Nil
      }
    }

  }

  def processObjectPropertyAssertionAxiom(axiom: OWLObjectPropertyAssertionAxiom, ontology: OWLOntology): List[Triple] = {

    val subject = axiom.getSubject.toStringID
    val predicate = axiom.getProperty.asInstanceOf[OWLObjectProperty].toStringID
    val obj = axiom.getObject.toStringID

    //val inverseRelationNames = getRelationInverseNames(Right(predicate), ontology)
    //val subRelationNames = getSubRelationNames(Right(predicate), ontology)

    //val inverseTriples = inverseRelationNames.map(inverseRelationName => new Triple(obj, inverseRelationName, subject))
    //val subRelationTriples = subRelationNames.map(subRelationName => new Triple(subject, subRelationName, obj))

    new Triple(subject, predicate, obj) :: Nil //inverseTriples ::: subRelationTriples
  }


  ////////// DOMAIN AND RANGE AXIOMS //////////////////////
  def processDomainAndRangeAxioms(domainAxioms: ListBuffer[OWLObjectPropertyDomainAxiom], rangeAxioms: ListBuffer[OWLObjectPropertyRangeAxiom], ontology: OWLOntology): List[Triple] = {

    val domainTuples = domainAxioms.map(axiom => { (axiom.getProperty, axiom.getDomain) }).toList
    val rangeTuples = rangeAxioms.map(axiom => { (axiom.getRange, axiom.getProperty) }).toList

    // get all triples for domain and range axioms that have the same property
    val domainAndRangeTuples = domainTuples.flatMap(domainTuple => {
      val domainProperty = domainTuple._1
      val domain = domainTuple._2

      // check type of domain and domain property
      if (domain.isInstanceOf[OWLClass] && domainProperty.isInstanceOf[OWLObjectProperty]) {
        val domainClass = domain.asInstanceOf[OWLClass]
        val domainPropertyObj = domainProperty.asInstanceOf[OWLObjectProperty]

        // get range axioms for this property
        val rangeAxioms = rangeTuples.filter(rangeTuple => {
          val rangeProperty = rangeTuple._2
          rangeProperty == domainPropertyObj && rangeProperty.isInstanceOf[OWLObjectProperty] && rangeTuple._1.isInstanceOf[OWLClass]
        })

        
        

        rangeAxioms.flatMap(rangeAxiom => {
          val rangeClass = rangeAxiom._1.asInstanceOf[OWLClass]
          val originalTriple = new Triple(domainClass.toStringID, domainPropertyObj.toStringID, rangeClass.toStringID)
          val inverseRelNames = getRelationInverseNames(Left(domainPropertyObj), ontology)
          val subRoles = getSubRelationNames(Left(domainPropertyObj), ontology)

          val inverseAxioms = inverseRelNames.map(inverseRelName => {
            new Triple(rangeClass.toStringID, inverseRelName, domainClass.toStringID)
          })

          val subRoleAxioms = subRoles.map(subRole => {
            new Triple(domainClass.toStringID, subRole, rangeClass.toStringID)
          })

          originalTriple :: inverseAxioms ::: subRoleAxioms
        }
        )
      } else {
        Nil
      }
    })

    domainAndRangeTuples
  }

  // Abstract methods
  def project(ontology: OWLOntology, withIndividuals: Boolean, verbose: Boolean): java.util.List[Triple] = Nil.asJava
  def projectAxiom(go_class: OWLClass, axiom: OWLClassAxiom): List[Triple] = Nil
  def projectAxiom(axiom: OWLAxiom): List[org.mowl.Types.Triple] = Nil
  def projectAxiom(axiom: OWLClassAxiom): List[org.mowl.Types.Triple] = Nil
  def projectAxiom(axiom: OWLAxiom, with_individuals: Boolean, verbose: Boolean): List[Triple] = Nil
}
