package org.mowl.Walking

import collection.JavaConverters._
import java.io._
import java.util.{ArrayList}
import scala.collection.mutable.{MutableList, ListBuffer, Stack, Map, HashMap, ArrayBuffer}
import scala.collection.immutable.HashSet
import util.control.Breaks._
import java.util.concurrent.{ExecutorService, Executors}
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration.Duration
import scala.concurrent.{ Await, Future }
import scala.concurrent.{ExecutionContext, ExecutionContextExecutorService}
import scala.util.{Failure, Success, Try}
import org.mowl.Edge

class Node2Vec (
  var edges: ArrayList[Edge],
  var numWalks: Int,
  var walkLength: Int,
  var p: Float,
  var q: Float,
  var workers: Int,
  var outfile: String,
  var nodesOfInterest: ArrayList[String]
) {


  val edgesSc = edges.asScala.map(x => (x.src, x.rel, x.dst, x.weight))
  val entities = edgesSc.map(x => List(x._1, x._2, x._3)).flatten.toSet
  val entsSrc = edgesSc.map(_._1).toSet
  val mapEntsIdx = entities.zip(Range(0, entities.size, 1)).toMap
  val mapIdxEnts = Range(0, entities.size, 1).zip(entities).toMap
  val entsIdx = entities.map(mapEntsIdx(_))
  val entsSrcIdx = entsSrc.map(mapEntsIdx(_))
  val nodes = edgesSc.map(x => List(x._1, x._3)).flatten.toSet
  val nodesIdx = nodes.map(mapEntsIdx(_))
  val (graph, weights) = processEdges()


  val rand = scala.util.Random

  val (pathsPerWorker, newWorkers) = numPathsPerWorker()

  var aliasNodes = Map[Int, (Array[Int], Array[Float])]()
  var aliasEdges = Map[(Int, Int), (Array[Int], Array[Float])]()

  val nodesOfInterestIdx =  HashSet() ++ nodesOfInterest.asScala.map(mapEntsIdx(_)).toSet

  private[this] val lock = new Object()

  val walksFile = new File(outfile)
  val bw = new BufferedWriter(new FileWriter(walksFile))


  def processEdges() = {
    val graph: Map[Int, ArrayBuffer[(Int, Int)]] = Map()
    val weights: HashMap[(Int, Int), Float] = HashMap()
    for ((src, rel, dst, weight) <- edgesSc){
      val srcIdx = mapEntsIdx(src)
      val relIdx = mapEntsIdx(rel)
      val dstIdx = mapEntsIdx(dst)

      if (!graph.contains(srcIdx)){
        graph(srcIdx) = ArrayBuffer((relIdx, dstIdx))
      }else{
        graph(srcIdx) += ((relIdx, dstIdx))
      }

      weights((srcIdx, dstIdx)) = weight
    }
    (graph.mapValues(_.sortBy(_._2).toArray), weights)
  }

  def walk() = {

    //preprocessTransitionProbs
    val argsList = for (
      i <- Range(0, newWorkers, 1)
    ) yield (i, pathsPerWorker(i), walkLength, p, q)

    print("Starting pool...")

    val executor: ExecutorService = Executors.newFixedThreadPool(workers)
    implicit val executionContext: ExecutionContextExecutorService = ExecutionContext.fromExecutorService(executor)

    println(s"+ started preprocessing probabilities...")
    val start = System.nanoTime() / 1000000

    var listsN: ListBuffer[ListBuffer[Int]] = ListBuffer.fill(newWorkers)(ListBuffer())
    for (i <- entsSrcIdx){
      listsN(i%newWorkers) += i
    }
    val argsListN = Range(0, newWorkers, 1).zip(listsN.toList.map(_.toList))
    val futNodes = Future.traverse(argsListN)(threadNodes)

    Await.ready(futNodes, Duration.Inf)

    futNodes.onComplete {
      case Success(msg) => println("* processing probabilities for nodes is over")
      case Failure(t) => println("An error has ocurred in preprocessing probabilities for nodes: " + t.getMessage + " - " + t.printStackTrace)
    }

    var listsE: ListBuffer[ListBuffer[(Int, Int)]] = ListBuffer.fill(newWorkers)(ListBuffer())
    val numEdges = weights.keySet.size
    for ((i, edge) <- Range(0, numEdges, 1).zip(weights.keySet.toList)){
      listsE(i%newWorkers) += edge
    }
    val argsListE = Range(0, newWorkers, 1).zip(listsE.toList.map(_.toList))
    val futEdges = Future.traverse(argsListE)(threadEdges)

    Await.ready(futEdges, Duration.Inf)

    futEdges.onComplete {
      case Success(msg) => println("* processing probabilities for edges is over")
      case Failure(t) => println("An error has ocurred in preprocessing probabilities for edges: " + t.getMessage + " - " + t.printStackTrace)
    }


    val end = System.nanoTime() / 1000000
    val duration = (end - start) / 1000
    println(s"- finished preprocessing probabilities after $duration seconds")


    val futWalks = Future.traverse(argsList)(writeWalksToDisk)

    Await.ready(futWalks, Duration.Inf)

    futWalks.onComplete {
      case Success(msg) => {
        println("* Walking is done, shutting down the executor")
        executionContext.shutdown()
        bw.close
      }
      case Failure(t) =>
        {
          println("An error has ocurred in preprocessing generating random walks: " + t.getMessage + " - " + t.printStackTrace)
          executionContext.shutdown()
          bw.close
        }
    }
  }


  def writeWalksToDisk(params: (Int, Int, Int, Float, Float))(implicit ec: ExecutionContext): Future[Unit] = Future {

    val (index, numWalks, walkLength, p, q) = params

    println(s"+ started processing thread $index")
    val start = System.nanoTime() / 1000000

    for (i <- 0 until numWalks){
      val nodesR = rand.shuffle(nodesIdx)
      for (n <- nodesR){
        randomWalk(walkLength, p, q, n)
      }
    }
     
    val end = System.nanoTime() / 1000000
    val duration = (end - start)
    println(s"- finished processing thread $index after $duration")
  }

  def randomWalk(walkLength: Int, p: Float, q: Float, start: Int ) = {

    val walk = Array.fill(2*walkLength-1){-1}

    walk(0) = start
    breakable {

      var i = 1
      while (i < 2*walkLength-1){
        
        val curNode = walk(i-1)
        val curNbrs = graph.contains(curNode) match {
          case true => graph(curNode)
          case false => Array[Tuple2[Int, Int]]()
        }

        if (curNbrs.size > 0) {

          if (walk(1) == -1){
            val (idx1, idx2) = aliasNodes(curNode)
            val (rel, next) = curNbrs(aliasDraw(idx1, idx2))
            walk(i) = rel
            walk(i+1) = next
          }else {
            val prevNode = walk(i-3)
            val (idx1, idx2) = aliasEdges((prevNode, curNode))
            val (rel, next) = curNbrs(aliasDraw(idx1, idx2))
         
            walk(i) = rel
            walk(i+1) = next
          }
          i = i+2
          
        }else{
          break
        }
      }
    }

    val toWrite = walk.filter(_ != -1).map(x => mapIdxEnts(x)).mkString(" ") + "\n"

    if (nodesOfInterest.size > 0){
      val walkSet = HashSet() ++ walk.toSet
      val intersection = walkSet & nodesOfInterestIdx

      if (intersection.size > 0){
        
        lock.synchronized {
          bw.write(toWrite)
        }
      }
    }else{
      lock.synchronized {
        bw.write(toWrite)
      }
    }

  }


  def getAliasEdge(src: Int, dst: Int) = {
    if (graph.contains(dst)){
      val dstNbrs = graph(dst).map(_._2)
      val lenDstNbrs = dstNbrs.length
      val unnormalizedProbs: Array[Float] = Array.fill(lenDstNbrs){-1}

      for (i <- 0 until lenDstNbrs)  {
        val dstNbr = dstNbrs(i)
        val prob = weights((dst, dstNbr))

        if (dstNbr == src){
          unnormalizedProbs(i) = prob/p
        }else if( weights.contains((dstNbr, src))){
          unnormalizedProbs(i) = prob
        }else{
          unnormalizedProbs(i) = prob/q
        }
      }

      val normConst = unnormalizedProbs.sum
      val normalizedProbs = unnormalizedProbs.map(x => x/normConst)

      aliasSetup(normalizedProbs.toArray)

    }else{
      (Array[Int](), Array[Float]())
    }

  }



  def threadNodes(params: (Int, List[Int]))(implicit ec: ExecutionContext): Future[Unit] = Future {
    val (idx, indices) = params
    val l = indices.length
    //println(s"Thread $idx, nodes to process $l")

    for (i <- indices){
      val node = i
      val unnormalizedProbs = graph(node).map(nbr => weights((node, nbr._2)))
      val normConst = unnormalizedProbs.sum
      val normalizedProbs = unnormalizedProbs.map(x => x/normConst)

      val alias = aliasSetup(normalizedProbs)
      lock.synchronized {
        aliasNodes += (node -> alias)
      }
    }
  }

  def threadEdges(params:  (Int, List[(Int, Int)]))(implicit ec: ExecutionContext): Future[Unit] = Future {

    val (idx, edges) = params
    val l = edges.length
    //println(s"Thread $idx, edges to process $l")

    for ((src, dst) <- edges){
      val alias = getAliasEdge(src, dst)
      lock.synchronized {
        aliasEdges += ((src, dst) -> alias)
      }
    }
  }

  //////////////////////////////////////////
  //https://lips.cs.princeton.edu/the-alias-method-efficient-sampling-with-many-discrete-outcomes/

  def aliasSetup(probs: Array[Float]) = {

    val K = probs.length
    val q: Array[Float] = new Array[Float](K)
    val J: Array[Int] = new Array[Int](K)

    val smaller = Stack[Int]()
    val larger = Stack[Int]()

    for ((kk, prob) <- Range(0, K, 1).zip(probs)){
      val qkk = K*prob
      q(kk) = qkk

      if (qkk < 1){
        smaller.push(kk)
      }else {
        larger.push(kk)
      }
    }

    var smallLen = smaller.length
    var largeLen = larger.length
    var qlarge: Float  = 0
    while (smallLen > 0 && largeLen > 0){
      val small = smaller.pop
      val large = larger.pop

      J(small) = large

      qlarge = q(large) + q(small) - 1
      q(large) = qlarge

      if (qlarge < 1){
        smaller.push(large)
        largeLen -=1
      }else{
        larger.push(large)
        smallLen -= 1
      }
    }

    (J, q)

  }

  def aliasDraw(J: Array[Int], q: Array[Float]): Int  = {

    val K = J.length
    val kk = rand.nextInt(K)
    
    if (rand.nextFloat< q(kk)){
      kk
    }else{
      J(kk)
    }
  }
  /////////////////////////////////////////

  def numPathsPerWorker(): (List[Int], Int) = {

    if (numWalks <= workers) {
      val newWorkers = numWalks

      var pathsPerWorker = for (
        i <- Range(0, numWalks, 1)
      ) yield 1

      (pathsPerWorker.toList, newWorkers)
    }else{
      val newWorkers = workers
      val remainder = numWalks % workers
      var aux = workers - remainder

      val ppw = ((numWalks+aux)/workers).floor.toInt
      var pathsPerWorker = ListBuffer(ppw)

      for (i <- 0 until (workers -1)){

        pathsPerWorker += ppw
      }

      var i = 0
      while (aux > 0){
        pathsPerWorker(i%workers) =  pathsPerWorker(i%workers) - 1
        i = i+1
        aux = aux -1
      }
      (pathsPerWorker.toList, newWorkers)
    }
  }

}


