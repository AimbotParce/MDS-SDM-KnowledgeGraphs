"""
Load the downloaded paper data from the local file system into an RDFS graph.
"""

import json
import logging
import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from pathlib import Path
from typing import Dict, Iterable, List

import rdflib
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import RDF, RDFS, XSD, Namespace
from rdflib.plugins.stores.sparqlstore import SPARQLUpdateStore

from lib.models.raw import RawPaper, RawReference

logger = logging.getLogger(__name__)


def yieldFromJSONLFiles(files: List[Path]):
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            yield from map(json.loads, f)


def load_paper(g: Graph, namespace: Namespace, paper: RawPaper, add_type_axioms: bool = True) -> None:
    if not "paperId" in paper or paper["paperId"] is None or paper["paperId"] == "":
        logger.error(f"Paper has no id: {paper}")
        return
    P = namespace  # To simplify the code
    paper_uri = P[paper["paperId"]]
    if add_type_axioms:
        g.add((paper_uri, RDF.type, P.Paper))
    g.add((paper_uri, P.paperTitle, Literal(paper["title"])))
    g.add((paper_uri, P.paperAbstract, Literal(paper["abstract"])))
    if "tldr" in paper and paper["tldr"] is not None and "text" in paper["tldr"]:
        g.add((paper_uri, P.paperContent, Literal(paper["tldr"]["text"])))

    if "fieldsOfStudy" in paper and paper["fieldsOfStudy"] is not None:
        for fos in paper["fieldsOfStudy"]:
            fos_id = fos.replace(" ", "_").replace("/", "_").lower()  # We cannot use spaces or slashes in URIs
            if add_type_axioms:
                g.add((P[fos_id], RDF.type, P.PaperTopic))
            g.add((paper_uri, P.paperIsAbout, P[fos_id]))
            g.add((P[fos_id], P.topicKeyword, Literal(fos)))

    if not "publicationVenue" in paper or paper["publicationVenue"] is None:
        logger.warning(f"Paper {paper['paperId']} has no publication venue.")
    else:
        venue = paper["publicationVenue"]
        if not "id" in venue:
            logger.warning(f"Paper {paper['paperId']} has a publication venue without id.")
        elif not "type" in venue:
            # We don't know the type of the venue, so we add it as a generic PublicationVenue
            logger.warning(f"Paper {paper['paperId']} has a publication venue without type.")
            if add_type_axioms:
                g.add((P[venue["id"]], RDF.type, P.PublicationVenue))
            g.add((paper_uri, P.isPublishedIn, P[venue["id"]]))
            if "name" in venue:
                g.add((P[venue["id"]], P.venueName, Literal(venue["name"])))
        elif venue["type"] == "journal":
            if add_type_axioms:
                g.add((P[venue["id"]], RDF.type, P.Journal))
            if "name" in venue:
                g.add((P[venue["id"]], P.journalName, Literal(venue["name"])))

            if not paper.get("journal") or not paper["journal"].get("volume"):
                logger.warning(f"Paper {paper['paperId']} has a journal venue without volume information.")
                volume_num = "unk"
            else:
                volume_num = paper["journal"]["volume"]
                if not volume_num.isdigit():
                    logger.warning(
                        f"Paper {paper['paperId']} has a journal venue with non-numeric volume: {volume_num}"
                    )
                    volume_num = "unk"

            journal_volume_id = f"{venue['id']}-{volume_num}"
            if add_type_axioms:
                g.add((P[journal_volume_id], RDF.type, P.JournalVolume))
            g.add((paper_uri, P.isPublishedInJournalVolume, P[journal_volume_id]))
            g.add((P[journal_volume_id], P.isVolumeOf, P[venue["id"]]))

            if volume_num != "unk":
                g.add((P[journal_volume_id], P.journalVolumeNumber, Literal(int(volume_num))))
        elif venue["type"] == "conference":
            if add_type_axioms:
                g.add((P[venue["id"]], RDF.type, P.Conference))
            if "name" in venue:
                g.add((P[venue["id"]], P.conferenceName, Literal(venue["name"])))

            # We don't have the edition of the proceedings, so we will assume that there's only one edition per year
            if "year" in paper:
                proceedings_year = paper["year"]
            else:
                logger.warning(f"Paper {paper['paperId']} has a conference venue without year information.")
                proceedings_year = "unk"

            proceedings_id = f"{venue['id']}-{proceedings_year}"
            if add_type_axioms:
                g.add((P[proceedings_id], RDF.type, P.Proceedings))
            g.add((paper_uri, P.isPublishedInProceedings, P[proceedings_id]))
            g.add((P[proceedings_id], P.isProceedingsOfConference, P[venue["id"]]))
            if proceedings_year != "unk":
                g.add((P[proceedings_id], P.proceedingsYear, Literal(int(proceedings_year))))
        elif venue["type"] == "workshop":
            if add_type_axioms:
                g.add((P[venue["id"]], RDF.type, P.Workshop))
            if "name" in venue:
                g.add((P[venue["id"]], P.workshopName, Literal(venue["name"])))

            # We don't have the edition of the proceedings, so we will assume that there's only one edition per year
            if "year" in paper:
                proceedings_year = paper["year"]
            else:
                logger.warning(f"Paper {paper['paperId']} has a workshop venue without year information.")
                proceedings_year = "unk"

            proceedings_id = f"{venue['id']}-{proceedings_year}"
            if add_type_axioms:
                g.add((P[proceedings_id], RDF.type, P.Proceedings))
            g.add((paper_uri, P.isPublishedInProceedings, P[proceedings_id]))
            g.add((P[proceedings_id], P.isProceedingsOfWorkshop, P[venue["id"]]))
            if proceedings_year != "unk":
                g.add((P[proceedings_id], P.proceedingsYear, Literal(int(proceedings_year))))
        else:
            logger.error(f"Paper {paper['paperId']} has an unknown venue type.")

    if "authors" not in paper or paper["authors"] is None:
        logger.error(f"Paper {paper['paperId']} has no authors.")
    else:
        for j, author in enumerate(paper["authors"], 1):
            if not "authorId" in author or author["authorId"] is None:
                logger.error(f"Paper {paper['paperId']}'s author {j} has no id.")
                continue
            if author["authorId"] == "":
                logger.error(f"Paper {paper['paperId']}'s author {j} has an empty id.")
                continue
            author_uri = P[author["authorId"]]
            if add_type_axioms:
                g.add((author_uri, RDF.type, P.Author))
            if "name" in author:
                g.add((author_uri, P.authorName, Literal(author["name"])))

            if j == 1:
                # We assume the first author in the list is the corresponding author
                g.add((author_uri, P.isCorrespondingAuthor, paper_uri))
            else:
                g.add((author_uri, P.writesPaper, paper_uri))


def load_reference(g: Graph, namespace: Namespace, reference: RawReference, add_type_axioms: bool = True) -> None:
    def check(ref: Dict, key: str) -> bool:
        if key not in ref or ref[key] is None:
            logger.error(f"Reference {reference['referenceId']} has no {key}.")
            return False
        if ref[key] == "":
            logger.error(f"Reference {reference['referenceId']} has an empty {key}.")
            return False
        if not "paperId" in ref[key] or ref[key]["paperId"] is None:
            logger.error(f"Reference {reference['referenceId']} has no {key} paper id.")
            return False
        if ref[key]["paperId"] == "":
            logger.error(f"Reference {reference['referenceId']} has an empty {key} paper id.")
            return False
        return True

    if not check(reference, "citedPaper"):
        return
    if not check(reference, "citingPaper"):
        return
    P = namespace  # To simplify the code
    cited_paper_uri = P[reference["citedPaper"]["paperId"]]
    citing_paper_uri = P[reference["citingPaper"]["paperId"]]
    if add_type_axioms:
        g.add((cited_paper_uri, RDF.type, P.Paper))
        g.add((citing_paper_uri, RDF.type, P.Paper))
    g.add((citing_paper_uri, P.paperCites, cited_paper_uri))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load paper data into an RDFS graph.")
    parser.add_argument("data_dir", type=Path, help="Directory containing the JSONL files with paper data.")
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

    store = SPARQLUpdateStore()
    store.open((repo_url, update_url))
    g = Graph(store=store, identifier=URIRef(args.graph_name or f"{repo_url}/default"))
    P = Namespace(args.namespace or f"{repo_url}/default/ontology#")

    g.bind("P", P)
    g.bind("rdf", RDF)
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)

    papers: List[Path] = args.data_dir.glob("raw-papers-*.jsonl")
    references: List[Path] = args.data_dir.glob("raw-references-*.jsonl")
    max_workers = os.cpu_count() or 1
    logger.info(f"Using {max_workers} worker threads for loading papers and references.")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = set()
        # Load all the papers and references in parallel.
        # Because the number of papers and references can be large, we will only submit the tasks as workers become
        # available. Otherwise, we might run out of memory.
        for paper in yieldFromJSONLFiles(papers):
            while len(futures) >= max_workers:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
            future = executor.submit(load_paper, g, P, paper, add_type_axioms=args.add_type_axioms)
            futures.add(future)
        for reference in yieldFromJSONLFiles(references):
            while len(futures) >= max_workers:
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
            future = executor.submit(load_reference, g, P, reference, add_type_axioms=args.add_type_axioms)
            futures.add(future)
