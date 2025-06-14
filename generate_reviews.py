"""
Generate synthetic reviews for academic papers and load them into an RDFS graph.
"""

import json
import logging
import os
import random
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List

import rdflib
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS, XSD, Namespace
from rdflib.plugins.stores.sparqlstore import SPARQLUpdateStore

# The following query gives a list of all possible reviewers for each paper,
# excluding the authors of the paper itself. However, it is too slow to run
# on the entire graph, so we'll go one paper at a time.
# q = (
#     "select ?paper (GROUP_CONCAT(DISTINCT ?reviewer; SEPARATOR=) AS ?reviewers) where {",
#     "    ?reviewer P:writesPaper ?paper2 ."
#     "    ?paper2 a P:Paper ;"
#     "        P:paperIsAbout ?topic ."
#     "    ?paper a P:Paper ;"
#     "        P:paperIsAbout ?topic ."
#     "    ?topic a P:PaperTopic ."
#     "    FILTER NOT EXISTS { ?reviewer P:writesPaper ?paper }"
#     "} GROUP BY ?paper",
# )


def generate_paper_reviews(paper: URIRef, rg: Graph, wg: Graph, P: Namespace, add_type_axioms: bool = True):
    """
    For a paper, generate between 3 and 8 reviews.
    Reviews are made by randomly selecting a subset of authors
    of papers that share the same topic, but not the same paper.
    (Paper) -[hasTopic]-> (Topic) <-[hasTopic]- (Paper2) <-[writes]- (Author2)
    Such that there isn't an edge (Author2) -[writes]-> (Paper).
    """
    q = (
        "SELECT DISTINCT ?reviewer WHERE {\n"
        "    ?reviewer P:writesPaper ?paper2 .\n"
        "    ?paper2 a P:Paper ;\n"
        "        P:paperIsAbout ?topic .\n"
        f"    <{paper}> a P:Paper ;\n"
        "        P:paperIsAbout ?topic .\n"
        "    ?topic a P:PaperTopic .\n"
        f"    FILTER NOT EXISTS {{ ?reviewer P:writesPaper <{paper}> }}\n"
        "} LIMIT 100"  # Limit to 100 reviewers to avoid performance issues
    )
    reviewers = list(rg.query(q))
    logger.info(f"Found {len(reviewers)} potential reviewers for paper {paper}")
    # Select a random subset of reviewers
    num_reviews = random.randint(3, 8)
    selected_reviewers = random.sample(reviewers, min(num_reviews, len(reviewers)))
    for reviewer in selected_reviewers:
        reviewer_uri = reviewer.reviewer
        review_uri = P[str(uuid.uuid4())]
        if add_type_axioms:
            wg.add((review_uri, RDF.type, P.Review))
        wg.add((reviewer_uri, P.writesReview, review_uri))
        wg.add((review_uri, P.reviewIsAbout, paper))
        wg.add((review_uri, P.reviewContent, Literal(f"Review for {paper}", datatype=XSD.string)))
        wg.add((review_uri, P.reviewVerdict, Literal(random.choice([True, False]), datatype=XSD.boolean)))


logger = logging.getLogger(__name__)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load paper data into an RDFS graph.")
    parser.add_argument("--host", type=str, default="localhost", help="SPARQL endpoint host")
    parser.add_argument("--port", type=int, default=7200, help="SPARQL endpoint port")
    parser.add_argument("--repository", type=str, default="academia-sdm", help="SPARQL repository name")
    parser.add_argument(
        "--graph-name",
        type=str,
        default=None,
        help="Graph name to use in the SPARQL store. If none, defaults to '<repo_url>/default'.",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default=None,
        help="Namespace URI to use for the RDF graph. If none, defaults to '<graph_url>/ontology#'.",
    )
    parser.add_argument(
        "--add-type-axioms",
        action="store_true",
        help="If set, adds type axioms for the properties and classes in the ontology.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="(%(asctime)s) %(levelname)s@%(name)s.%(funcName)s:%(lineno)d # %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    repo_url = f"http://{args.host}:{args.port}/repositories/{args.repository}"
    update_url = f"http://{args.host}:{args.port}/repositories/{args.repository}/statements"
    store = SPARQLUpdateStore(query_endpoint=repo_url)
    store.open((repo_url, update_url))

    P = Namespace(args.namespace or f"{repo_url}/default/ontology#")

    # I don't understand why, but if I give this graph an identifier, it fails to load the data.
    rg = Graph(store=store)
    rg.bind("P", P)
    rg.bind("rdf", RDF)
    rg.bind("rdfs", RDFS)
    rg.bind("xsd", XSD)

    wg = Graph(store=store, identifier=URIRef(args.graph_name or f"{repo_url}/default"))
    wg.bind("P", P)
    wg.bind("rdf", RDF)
    wg.bind("rdfs", RDFS)
    wg.bind("xsd", XSD)

    papers = rg.subjects(RDF.type, P.Paper)

    max_workers = os.cpu_count() or 1
    logger.info(f"Using {max_workers} worker threads for loading papers and references.")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for paper in papers:
            executor.submit(generate_paper_reviews, paper, rg, wg, P, add_type_axioms=args.add_type_axioms)
