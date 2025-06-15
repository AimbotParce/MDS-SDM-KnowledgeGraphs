"""
Helper function to create a paper information structure YAML file with proper
data.

This script will prompt you for the following information:
- Topics
- Publication venue
- Authors
- References
- Citations
To do so, it will let you choose from a list of existing topics, venues, and authors
in the database.
"""

import logging
from pathlib import Path
from typing import Dict, List, Literal, overload

import rdflib
import yaml
from rdflib import Graph, URIRef
from rdflib.namespace import RDF, RDFS, XSD, Namespace
from rdflib.plugins.stores.sparqlstore import SPARQLStore

from lib.models.review_recommendations import PaperInfo, is_paper_info


@overload
def _prompt_choice(prompt: str, choices: List[str]) -> int: ...
@overload
def _prompt_choice(prompt: str, choices: List[str], multiple: Literal[False]) -> int: ...
@overload
def _prompt_choice(prompt: str, choices: List[str], multiple: Literal[True]) -> List[int]: ...
def _prompt_choice(prompt: str, choices: List[str], multiple: bool = False) -> int | List[int]:
    """Prompt the user to choose multiple options from a list of choices."""
    print(f"{prompt}:")
    maxsize = len(str(len(choices)))
    for i, choice in enumerate(choices):
        print(f"  [{i + 1:>{maxsize}}] {choice}")
    while True:
        if multiple:
            selection = input("Choice (comma-separated numbers): ")
            res = []
            for part in selection.split(","):
                try:
                    index = int(part.strip()) - 1
                    if 0 <= index < len(choices):
                        res.append(index)
                    else:
                        print(f"Please enter numbers between 1 and {len(choices)}.")
                        break
                except ValueError:
                    print(f"Invalid input '{part.strip()}'. Please enter numbers between 1 and {len(choices)}.")
                    break
            else:
                return res
        else:
            try:
                selection = int(input("Choice (number): "))
                if 1 <= selection <= len(choices):
                    return selection - 1
                print(f"Please enter a number between 1 and {len(choices)}.")
            except ValueError:
                print(f"Invalid input. Please enter a number between 1 and {len(choices)}.")


def prompt_paper_info(g: Graph, namespace: Namespace) -> PaperInfo:
    P = namespace  # For simplicity in the code
    topic_names: List[str] = []
    topic_uris: List[URIRef] = []
    for topic_uri in g.subjects(RDF.type, P.PaperTopic):
        topic_name = g.value(topic_uri, P.topicKeyword, default=None)
        if topic_name is None:
            logging.warning(f"Topic {topic_uri} has no keyword.")
            topic_names.append(str(topic_uri))
        else:
            topic_names.append(str(topic_name))
        topic_uris.append(topic_uri)
    if not topic_names:
        logging.error("No topics found in the graph. Please add some topics first.")
        exit(1)
    topic_indices = _prompt_choice("Select topics", topic_names, multiple=True)
    topics = [str(topic_uris[i]) for i in topic_indices]
    print(f"Selected topics: {', '.join(topics)}")

    journal_or_proceedings = _prompt_choice(
        "Is the paper published in a journal or proceedings?", ["Journal", "Proceedings", "Other"]
    )
    if journal_or_proceedings == 0:  # Journal
        journal_names: List[str] = []
        journal_uris: List[URIRef] = []
        for journal_uri in g.subjects(RDF.type, P.Journal):
            journal_name = g.value(journal_uri, P.journalName, default=None)
            if journal_name is None:
                logging.warning(f"Journal {journal_uri} has no name.")
                journal_names.append(str(journal_uri))
            else:
                journal_names.append(str(journal_name))
            journal_uris.append(journal_uri)
        if not journal_names:
            logging.error("No journals found in the graph. Please add some journals first.")
            exit(1)
        journal_index = _prompt_choice("Select journal", journal_names)
        journal_uri = journal_uris[journal_index]
        # Prompt for the volume
        journal_volume_numbers: List[int] = []
        journal_volume_uris: List[URIRef] = []
        for volume_uri in g.subjects(P.isVolumeOf, journal_uri):
            volume_number = g.value(volume_uri, P.journalVolumeNumber, default=None)
            if volume_number is None:
                logging.warning(f"Journal volume {volume_uri} has no number.")
                journal_volume_numbers.append(str(volume_uri))
            else:
                journal_volume_numbers.append(int(volume_number))
            journal_volume_uris.append(volume_uri)
        if not journal_volume_numbers:
            logging.error("No journal volumes found in the graph. Please add some journal volumes first.")
            exit(1)
        journal_volume_index = _prompt_choice("Select journal volume", journal_volume_numbers)
        publication_venue = str(journal_volume_uris[journal_volume_index])

    elif journal_or_proceedings == 1:  # Proceedings
        conference_or_workshop = _prompt_choice(
            "Are the proceedings of a conference or a workshop?", ["Conference", "Workshop"]
        )
        if conference_or_workshop == 0:  # Conference
            conference_names: List[str] = []
            conference_uris: List[URIRef] = []
            for conference_uri in g.subjects(RDF.type, P.Conference):
                conference_name = g.value(conference_uri, P.conferenceName, default=None)
                if conference_name is None:
                    logging.warning(f"Conference {conference_uri} has no name.")
                    conference_names.append(str(conference_uri))
                else:
                    conference_names.append(str(conference_name))
                conference_uris.append(conference_uri)
            if not conference_names:
                logging.error("No conferences found in the graph. Please add some conferences first.")
                exit(1)
            conference_index = _prompt_choice("Select conference", conference_names)
            conference_uri = conference_uris[conference_index]
            # Prompt for the proceedings
            proceedings_years: List[str] = []
            proceedings_uris: List[URIRef] = []
            for proceedings_uri in g.subjects(P.isProceedingsOfConference, conference_uri):
                proceedings_year = g.value(proceedings_uri, P.proceedingsYear, default=None)
                if proceedings_year is None:
                    logging.warning(f"Proceedings {proceedings_uri} has no year.")
                    proceedings_years.append(str(proceedings_uri))
                else:
                    proceedings_years.append(str(proceedings_year))
                proceedings_uris.append(proceedings_uri)
            if not proceedings_years:
                logging.error(
                    "No proceedings found in the graph for the given conference. Please add some proceedings first."
                )
                exit(1)
            proceedings_index = _prompt_choice("Select proceedings year", proceedings_years)
            publication_venue = str(proceedings_uris[proceedings_index])
        elif conference_or_workshop == 1:  # Workshop
            workshop_names: List[str] = []
            workshop_uris: List[URIRef] = []
            for workshop_uri in g.subjects(RDF.type, P.Workshop):
                workshop_name = g.value(workshop_uri, P.workshopName, default=None)
                if workshop_name is None:
                    logging.warning(f"Workshop {workshop_uri} has no name.")
                    workshop_names.append(str(workshop_uri))
                else:
                    workshop_names.append(str(workshop_name))
                workshop_uris.append(workshop_uri)
            if not workshop_names:
                logging.error("No workshops found in the graph. Please add some workshops first.")
                exit(1)
            workshop_index = _prompt_choice("Select workshop", workshop_names)
            workshop_uri = workshop_uris[workshop_index]
            # Prompt for the proceedings
            proceedings_years: List[str] = []
            proceedings_uris: List[URIRef] = []
            for proceedings_uri in g.subjects(P.isProceedingsOfWorkshop, workshop_uri):
                proceedings_year = g.value(proceedings_uri, P.proceedingsYear, default=None)
                if proceedings_year is None:
                    logging.warning(f"Proceedings {proceedings_uri} has no year.")
                    proceedings_years.append(str(proceedings_uri))
                else:
                    proceedings_years.append(str(proceedings_year))
                proceedings_uris.append(proceedings_uri)
            if not proceedings_years:
                logging.error(
                    "No proceedings found in the graph for the given workshop. Please add some proceedings first."
                )
                exit(1)
            proceedings_index = _prompt_choice("Select proceedings year", proceedings_years)
            publication_venue = str(proceedings_uris[proceedings_index])
    elif journal_or_proceedings == 2:  # Other
        venue_names: List[str] = []
        venue_uris: List[URIRef] = []
        for venue_uri in g.subjects(RDF.type, P.PublicationVenue):
            # Check that the venue is not a journal or proceedings
            if (venue_uri, RDF.type, P.Journal) in g or (venue_uri, RDF.type, P.Proceedings) in g:
                continue
            venue_name = g.value(venue_uri, P.venueName, default=None)
            if venue_name is None:
                logging.warning(f"Venue {venue_uri} has no name.")
                venue_names.append(str(venue_uri))
            else:
                venue_names.append(str(venue_name))
            venue_uris.append(venue_uri)
        if not venue_names:
            logging.error("No 'Other' venues found in the graph. Please add some venues first.")
            exit(1)
        venue_index = _prompt_choice("Select venue", venue_names)
        publication_venue = str(venue_uris[venue_index])
    else:
        # This should never happen, but just in case
        logging.error("Invalid choice for publication venue type.")
        exit(1)
    print(f"Selected publication venue: {publication_venue}")

    # Select the authors
    author_names: List[str] = []
    author_uris: List[URIRef] = []
    for author_uri in g.subjects(RDF.type, P.Author):
        author_name = g.value(author_uri, P.authorName, default=None)
        if author_name is None:
            logging.warning(f"Author {author_uri} has no name.")
            author_names.append(str(author_uri))
        else:
            author_names.append(str(author_name))
        author_uris.append(author_uri)
    if not author_names:
        logging.error("No authors found in the graph. Please add some authors first.")
        exit(1)
    author_indices = _prompt_choice("Select authors", author_names, multiple=True)
    authors = [str(author_uris[i]) for i in author_indices]
    print(f"Selected authors: {', '.join(authors)}")

    # Select the references
    reference_names: List[str] = []
    reference_uris: List[URIRef] = []
    for reference_uri in g.subjects(RDF.type, P.Paper):
        reference_name = g.value(reference_uri, P.paperTitle, default=None)
        if reference_name is None:
            logging.warning(f"Reference {reference_uri} has no title.")
            reference_names.append(str(reference_uri))
        else:
            reference_names.append(str(reference_name))
        reference_uris.append(reference_uri)
    if not reference_names:
        logging.error("No references found in the graph. Please add some references first.")
        exit(1)
    reference_indices = _prompt_choice("Select the references for this paper", reference_names, multiple=True)
    references = [str(reference_uris[i]) for i in reference_indices]
    print(f"Selected references: {', '.join(references)}")

    # Select the citations
    citation_names: List[str] = []
    citation_uris: List[URIRef] = []
    for citation_uri in g.subjects(RDF.type, P.Paper):
        citation_name = g.value(citation_uri, P.paperTitle, default=None)
        if citation_name is None:
            logging.warning(f"Citation {citation_uri} has no title.")
            citation_names.append(str(citation_uri))
        else:
            citation_names.append(str(citation_name))
        citation_uris.append(citation_uri)
    if not citation_names:
        logging.error("No citations found in the graph. Please add some citations first.")
        exit(1)
    citation_indices = _prompt_choice("Select papers that cite this paper", citation_names, multiple=True)
    citations = [str(citation_uris[i]) for i in citation_indices]
    print(f"Selected citations: {', '.join(citations)}")

    paper_info: PaperInfo = {
        "topics": [t.replace(str(namespace), "") for t in topics],
        "publication_venue": publication_venue.replace(str(namespace), ""),
        "authors": [a.replace(str(namespace), "") for a in authors],
        "references": [r.replace(str(namespace), "") for r in references],
        "citations": [c.replace(str(namespace), "") for c in citations],
    }
    return paper_info


if __name__ == "__main__":
    import argparse

    import yaml

    parser = argparse.ArgumentParser(
        description="Recommend reviewers for a new paper using Knowledge Graph Embeddings."
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Path to the output YAML file. If not provided, print to stdout."
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

    paper_info = prompt_paper_info(g, P)
    if not is_paper_info(paper_info):
        # This should never happen, but just in case
        logging.error("Invalid paper information structure.")
        exit(1)

    if args.output is not None:
        with open(args.output, "w") as file:
            yaml.dump(paper_info, file, default_flow_style=False)
        logging.info(f"Paper information saved to {args.output}.")
    else:
        print(yaml.dump(paper_info, default_flow_style=False))
        logging.info("Paper information printed to stdout.")
