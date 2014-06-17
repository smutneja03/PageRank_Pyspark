
#!/usr/bin/env python

import re, sys
from operator import add

from pyspark import SparkContext


def computeAuth(urls, hub):
    """Calculates hub contributions to the auth of other URLs."""
    num_urls = len(urls)
    for url in urls: yield (url, hub)

def computeHub(urls, auth):
    """Calculates auth contributions to the hub of other URLs."""
    num_urls = len(urls)
    for url in urls: yield (url, auth)

def outNeighbors(urls):
    """Parses a urls pair string into urls pair."""
    parts = re.split(r'\s+', urls)
    return parts[0], parts[1]

def inNeighbors(urls):
    """Parses a urls pair string into urls pair."""
    parts = re.split(r'\s+', urls)
    return parts[1], parts[0]

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print >> sys.stderr, "Usage: pagerank <master> <file> <number_of_iterations>"
        exit(-1)

    # Initialize the spark context.
    sc = SparkContext(sys.argv[1], "HitsPageRank")

    # Loads in input file. It should be in format of:
    #     URL         neighbor URL
    #     URL         neighbor URL
    #     URL         neighbor URL
    #     ...
    lines = sc.textFile(sys.argv[2], 1)

    # Loads all URLs from input file and initialize their neighbors.
    out_links = lines.map(lambda urls: outNeighbors(urls)).distinct().groupByKey().cache()
    in_links = lines.map(lambda urls: inNeighbors(urls)).distinct().groupByKey().cache()
    
    # Loads all URLs with other URL(s) link to from input file and initialize ranks of them to one.
    hubs = out_links.map(lambda (url, neighbors): (url, 1.0))
    auths = in_links.map(lambda (url, neighbors): (url, 1.0))
    
    # Calculates and updates URL ranks continuously using Hits algorithm.
    for iteration in xrange(int(sys.argv[3])):
        # Calculates URL contributions to the rank of other URLs.
        # Here we are contributing auth of a link present in the outgoing list of a link whose hub is given
        auth_contribs = out_links.join(hubs).flatMap(lambda (url, (urls, hub)): 
            computeAuth(urls, hub))
        auths = auth_contribs.reduceByKey(add)
        max_value = max(auths.collect(), key=lambda x:x[1])[1]
        auths = auths.mapValues(lambda rank: rank/(max_value))
        # Here we are contributing hub of a link present in the incoming list of a link whose auth is given
        hub_contribs = in_links.join(auths).flatMap(lambda (url, (urls, auth)):
            computeHub(urls, auth))
        hubs = hub_contribs.reduceByKey(add)
        max_value = max(hubs.collect(), key=lambda x:x[1])[1]
        hubs = hubs.mapValues(lambda rank:rank/(max_value))
        # Re-calculates URL ranks based on neighbor contributions.
        
    # Collects all URL ranks and dump them to console.
    for (link, rank) in auths.collect():
        print "%s has auth: %s" % (link, rank)
    for (link, rank) in hubs.collect():
        print "%s has hub: %s" % (link, rank)

