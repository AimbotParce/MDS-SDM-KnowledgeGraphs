"""
Generate synthetic reviews for academic papers and load them into an RDFS graph.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List

import rdflib
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS, XSD, Namespace
from rdflib.plugins.stores.sparqlstore import SPARQLUpdateStore


def generate_reviews(g: Graph, P: Namespace, add_type_axioms: bool = True):
    # Load all the papers from the graph
    for r in g.subjects(RDF.type, P.Paper):
        print(r)


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
    g = Graph(store=store, identifier=URIRef(args.graph_name or f"{repo_url}/default"))
    P = Namespace(args.namespace or f"{repo_url}/default/ontology#")

    g.bind("P", P)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)

    generate_reviews(g, P, add_type_axioms=args.add_type_axioms)
