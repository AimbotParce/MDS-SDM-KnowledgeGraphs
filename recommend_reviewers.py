"""
Using Knowledge Graph Embeddings, recommend reviewers for a new paper.
This script assumes that the paper isn't already in the database, so it requires
a document specifying the following:
- Topics (list of topic IDs, strings)
- Publication venue (publicationVenue ID, string)
- Authors (list of strings, the first is the corresponding author)
- References (list of paper IDs, strings)
- Citations (list of paper IDs, strings, if available)
In a yaml file.
"""

import heapq
import logging
from pathlib import Path
from typing import Dict, List, Tuple, TypeGuard

import pandas as pd
import pykeen
import pykeen.models
import torch
from rdflib import Graph, URIRef
from rdflib.namespace import RDF, RDFS, XSD, Namespace
from rdflib.plugins.stores.sparqlstore import SPARQLStore

from lib.models.review_recommendations import PaperInfo, is_paper_info

logger = logging.getLogger(__name__)


def validate_paper_info(paper_info: PaperInfo, g: Graph, namespace: Namespace) -> TypeGuard[PaperInfo]:
    """
    Validate the values of the paper_info dictionary against the RDF graph.
    """
    # Check that the elements exist in the graph
    for topic_uri in paper_info["topics"]:
        if not (namespace[topic_uri], RDF.type, namespace.PaperTopic) in g:
            logger.error(f"Topic {topic_uri} does not exist in the graph.")
            return False
    if not (namespace[paper_info["publication_venue"]], RDF.type, namespace.PublicationVenue) in g:
        logger.error(f"Publication venue {paper_info['publication_venue']} does not exist in the graph.")
        return False
    for author_uri in paper_info["authors"]:
        if not (namespace[author_uri], RDF.type, namespace.Author) in g:
            logger.error(f"Author {author_uri} does not exist in the graph.")
            return False
    for reference_uri in paper_info["references"]:
        if not (namespace[reference_uri], RDF.type, namespace.Paper) in g:
            logger.error(f"Reference {reference_uri} does not exist in the graph.")
            return False
    for citation_uri in paper_info["citations"]:
        if not (namespace[citation_uri], RDF.type, namespace.Paper) in g:
            logger.error(f"Citation {citation_uri} does not exist in the graph.")
            return False
    return True


def load_rotatE_kge(model_dir: Path) -> tuple[pykeen.models.RotatE, Dict[str, int], Dict[str, int]]:
    pykeen_model = torch.load(model_dir / "trained_model.pkl", weights_only=False)
    file_path = model_dir / "training_triples/relation_to_id.tsv.gz"
    df = pd.read_csv(file_path, sep="\t", compression="gzip", header=0)
    rel_to_id = dict(zip(df.iloc[:, 1], df.iloc[:, 0]))
    file_path = model_dir / "training_triples/entity_to_id.tsv.gz"
    df = pd.read_csv(file_path, sep="\t", compression="gzip", header=0)
    ent_to_id = dict(zip(df.iloc[:, 1], df.iloc[:, 0]))

    return pykeen_model, rel_to_id, ent_to_id


def uri_str(uri: URIRef) -> int:
    """
    Convert a URIRef to an ID using the entity to ID mapping.
    """
    return "<" + str(uri) + ">"


def approximate_paper_embedding(
    paper_info: PaperInfo,
    model: pykeen.models.RotatE,
    rel_to_id: Dict[str, int],
    ent_to_id: Dict[str, int],
    namespace: Namespace,
) -> torch.ComplexType:
    """
    Approximate the embedding of a new paper based on its information.
    """
    device = model.device

    # Create a tensor for the paper embedding
    paper_embedding = torch.zeros(1, model.entity_representations[0].shape[0], device=device, dtype=torch.complex64)
    combined_embeddings = 0
    entity_embeddings = model.entity_representations[0]
    relation_embeddings = model.relation_representations[0]

    # Add topic embeddings
    # Get the embedding of the "paperIsAbout"
    paper_is_about_uri = uri_str(namespace["paperIsAbout"])
    if paper_is_about_uri not in rel_to_id:
        logger.warning(f"Relation {paper_is_about_uri} not found in relation to ID mapping.")
    else:
        paper_is_about_id = rel_to_id[paper_is_about_uri]
        paper_is_about_embedding = relation_embeddings(indices=torch.as_tensor([paper_is_about_id], device=device))
        # Invert the paper_is_about embedding (it is a complex number of norm 1, a rotation. We want the inverse rotation)
        paper_is_about_embedding_conj = torch.conj(paper_is_about_embedding)
        for topic in paper_info["topics"]:
            topic_id = uri_str(namespace[topic])
            if not topic_id in ent_to_id:
                logger.warning(f"Topic {topic} not found in entity to ID mapping.")
                continue
            topic_embedding = entity_embeddings(indices=torch.as_tensor([ent_to_id[topic_id]], device=device))
            paper_embedding += torch.mul(topic_embedding, paper_is_about_embedding_conj)
            combined_embeddings += 1

    # Add publication venue embedding
    # Get the embedding of the "isPublishedIn"
    is_published_in_uri = uri_str(namespace["isPublishedIn"])
    if is_published_in_uri not in rel_to_id:
        logger.warning(f"Relation {is_published_in_uri} not found in relation to ID mapping.")
    else:
        is_published_in_id = rel_to_id[is_published_in_uri]
        is_published_in_embedding = relation_embeddings(indices=torch.as_tensor([is_published_in_id], device=device))
        # Invert the is_published_in embedding
        is_published_in_embedding_conj = torch.conj(is_published_in_embedding)
        publication_venue_id = uri_str(namespace[paper_info["publication_venue"]])
        if publication_venue_id not in ent_to_id:
            logger.warning(f"Publication venue {paper_info['publication_venue']} not found in entity to ID mapping.")
        else:
            publication_venue_embedding = entity_embeddings(
                indices=torch.as_tensor([ent_to_id[publication_venue_id]], device=device)
            )
            paper_embedding += torch.mul(publication_venue_embedding, is_published_in_embedding_conj)
            combined_embeddings += 1

    # Add author embeddings
    # Get the embedding of the "writesPaper"
    writes_paper_uri = uri_str(namespace["writesPaper"])
    if writes_paper_uri not in rel_to_id:
        logger.warning(f"Relation {writes_paper_uri} not found in relation to ID mapping.")
    else:
        writes_paper_id = rel_to_id[writes_paper_uri]
        writes_paper_embedding = relation_embeddings(indices=torch.as_tensor([writes_paper_id], device=device))
        for author in paper_info["authors"]:
            author_id = uri_str(namespace[author])
            if author_id not in ent_to_id:
                logger.warning(f"Author {author} not found in entity to ID mapping.")
                continue
            author_embedding = entity_embeddings(indices=torch.as_tensor([ent_to_id[author_id]], device=device))
            paper_embedding += torch.mul(author_embedding, writes_paper_embedding)
            combined_embeddings += 1

    # Add reference and citation embeddings
    # Get the embedding of the "paperCites"
    paper_cites_uri = uri_str(namespace["paperCites"])
    if paper_cites_uri not in rel_to_id:
        logger.warning(f"Relation {paper_cites_uri} not found in relation to ID mapping.")
    else:
        paper_cites_id = rel_to_id[paper_cites_uri]
        paper_cites_embedding = relation_embeddings(indices=torch.as_tensor([paper_cites_id], device=device))
        # Invert the cites embedding
        paper_cites_embedding_conj = torch.conj(paper_cites_embedding)

        # Add references
        for reference in paper_info["references"]:
            reference_id = uri_str(namespace[reference])
            if reference_id not in ent_to_id:
                logger.warning(f"Reference {reference} not found in entity to ID mapping.")
                continue
            reference_embedding = entity_embeddings(indices=torch.as_tensor([ent_to_id[reference_id]], device=device))
            paper_embedding += torch.mul(reference_embedding, paper_cites_embedding_conj)
            combined_embeddings += 1

        # Add citations
        for citation in paper_info["citations"]:
            citation_id = uri_str(namespace[citation])
            if citation_id not in ent_to_id:
                logger.warning(f"Citation {citation} not found in entity to ID mapping.")
                continue
            citation_embedding = entity_embeddings(indices=torch.as_tensor([ent_to_id[citation_id]], device=device))
            paper_embedding += torch.mul(citation_embedding, paper_cites_embedding)
            combined_embeddings += 1

    # Normalize the embedding
    if combined_embeddings == 0:
        raise ValueError("No valid embeddings found for the paper information.")
    return paper_embedding / combined_embeddings


def approximate_potential_reviewer(
    paper_embedding: torch.Tensor,
    model: pykeen.models.RotatE,
    rel_to_id: Dict[str, int],
    ent_to_id: Dict[str, int],
    namespace: Namespace,
) -> torch.ComplexType:
    """
    Concatenate the two transformations "reviewIsAbout*" and "writesReview*"
    to get a potential reviewer embedding
    """
    device = model.device

    # Get the embedding of the "reviewIsAbout"
    review_is_about_uri = uri_str(namespace["reviewIsAbout"])
    if review_is_about_uri not in rel_to_id:
        raise ValueError(f"Relation {review_is_about_uri} not found in relation to ID mapping.")
    review_is_about_id = rel_to_id[review_is_about_uri]
    review_is_about_embedding = model.relation_representations[0](
        indices=torch.as_tensor([review_is_about_id], device=device)
    )
    # Invert the review_is_about embedding
    review_is_about_embedding_conj = torch.conj(review_is_about_embedding)

    # Get the embedding of the "writesReview"
    writes_review_uri = uri_str(namespace["writesReview"])
    if writes_review_uri not in rel_to_id:
        raise ValueError(f"Relation {writes_review_uri} not found in relation to ID mapping.")
    writes_review_id = rel_to_id[writes_review_uri]
    writes_review_embedding = model.relation_representations[0](
        indices=torch.as_tensor([writes_review_id], device=device)
    )

    # Invert the writes_review embedding
    writes_review_embedding_conj = torch.conj(writes_review_embedding)

    # Combine the embeddings
    potential_review_embedding = torch.mul(paper_embedding, review_is_about_embedding_conj)
    potential_reviewer_embedding = torch.mul(potential_review_embedding, writes_review_embedding_conj)
    return potential_reviewer_embedding


def get_all_authors(g: Graph, namespace: Namespace) -> set[str]:
    authors = set()
    for author in g.subjects(RDF.type, namespace["Author"]):
        authors.add(uri_str(author))
    return authors


def get_k_nearest(
    potential_reviewer_embedding: torch.Tensor,
    model: pykeen.models.RotatE,
    ent_to_id: Dict[str, int],
    authors: set[str],
    k: int = 10,
) -> list[str]:
    """
    Get the k nearest reviewers based on the potential reviewer embedding.
    """
    device = model.device
    entity_embeddings = model.entity_representations[0]

    # Get all entity embeddings
    best_authors: List[Tuple[float, str]] = []  # Max heap to store the k nearest authors
    for auth, auth_id in ent_to_id.items():
        if auth not in authors:
            continue
        auth_rep = entity_embeddings(indices=torch.as_tensor([auth_id], device=device))
        diff_real = auth_rep.real - potential_reviewer_embedding.real
        diff_imag = auth_rep.imag - potential_reviewer_embedding.imag
        dist_squared = (diff_real**2 + diff_imag**2).sum(dim=1)
        if len(best_authors) < k:
            heapq.heappush(best_authors, (-dist_squared.item(), auth))
        else:
            heapq.heappushpop(best_authors, (-dist_squared.item(), auth))
    # Sort the best authors by distance
    best_authors.sort()
    best_auth = [auth for _, auth in best_authors]
    best_auth_dist = [-dist for dist, _ in best_authors]
    return best_auth, best_auth_dist


if __name__ == "__main__":
    import argparse

    import yaml

    parser = argparse.ArgumentParser(
        description="Recommend reviewers for a new paper using Knowledge Graph Embeddings."
    )
    parser.add_argument(
        "paper_yaml",
        type=str,
        help="Path to the YAML file containing paper information.",
    )
    parser.add_argument("--host", type=str, default="localhost", help="SPARQL endpoint host")
    parser.add_argument("--port", type=int, default=7200, help="SPARQL endpoint port")
    parser.add_argument("--repository", type=str, default="academia-sdm", help="SPARQL repository name")
    parser.add_argument(
        "--namespace",
        type=str,
        default=None,
        help="Namespace URI to use for the RDF graph. If none, defaults to '<graph_url>/ontology#'.",
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        required=True,
        help="Directory containing the trained RotatE model and entity/relation mappings.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="(%(asctime)s) %(levelname)s@%(name)s.%(funcName)s:%(lineno)d # %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    repo_url = f"http://{args.host}:{args.port}/repositories/{args.repository}"

    store = SPARQLStore(query_endpoint=repo_url)
    g = Graph(store=store)
    P = Namespace(args.namespace or f"{repo_url}/default/ontology#")
    g.bind("P", P)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)

    with open(args.paper_yaml, "r") as file:
        paper_info: Dict = yaml.safe_load(file)

    if not is_paper_info(paper_info):
        logger.error("Invalid paper information structure.")
        exit(1)

    if not validate_paper_info(paper_info, g, P):
        logger.error("Paper information validation failed.")
        exit(1)

    # Load the RotatE model and mappings
    try:
        model, rel_to_id, ent_to_id = load_rotatE_kge(args.model_dir)
    except Exception as e:
        logger.error(f"Failed to load RotatE model: {e}")
        exit(1)

    logger.info("RotatE model loaded successfully.")

    # Approximate the paper embedding
    try:
        paper_embedding = approximate_paper_embedding(paper_info, model, rel_to_id, ent_to_id, P)
    except ValueError as e:
        logger.error(f"Failed to approximate paper embedding: {e}")
        exit(1)

    logger.info("Paper embedding approximated successfully.")

    # Approximate the potential reviewer embedding
    try:
        potential_reviewer_embedding = approximate_potential_reviewer(paper_embedding, model, rel_to_id, ent_to_id, P)
    except ValueError as e:
        logger.error(f"Failed to approximate potential reviewer embedding: {e}")
        exit(1)

    logger.info("Potential reviewer embedding approximated successfully.")

    # Predict the top 10 potential reviewers
    authors = get_all_authors(g, P)
    if not authors:
        logger.error("No authors found in the graph.")
        exit(1)
    logger.info(f"Found {len(authors)} authors in the graph.")

    # Filter out the authors who are already in the paper's authors list
    paper_authors = set(uri_str(P[a]) for a in paper_info["authors"])
    authors = authors - paper_authors

    best_auth, best_auth_dist = get_k_nearest(potential_reviewer_embedding, model, ent_to_id, authors, k=10)
    if not best_auth:
        logger.error("No potential reviewers found.")
        exit(1)
    logger.info(f"Recommended reviewers with their distances:")
    for auth, dist in zip(best_auth, best_auth_dist):
        logger.info(f"  Author: {auth}, Distance: {dist:.4f}")
