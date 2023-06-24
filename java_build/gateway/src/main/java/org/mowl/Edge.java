package org.mowl;
   
public class Edge {

    public String src;
    public String rel = "null";
    public String dst;
    public float weight = 1.0f;

    public Edge(String src, String dst){
	this.src = src;
	this.dst = dst;	    
    }

    public Edge(String src, String rel, String dst){
	this.src = src;
	this.rel = rel;
	this.dst = dst;
    }
    
    public Edge(String src, String dst, float weight){
	this.src = src;
	this.dst = dst;
	this.weight = weight;  
    }

    public Edge(String src, String rel, String dst, float weight){
	this.src = src;
	this.rel = rel;
	this.dst = dst;
	this.weight = weight;  
    }


}

    
