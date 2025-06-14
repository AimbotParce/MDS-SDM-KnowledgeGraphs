import logging
from pathlib import Path

from rdflib import Graph, URIRef
from rdflib.namespace import RDF, RDFS, XSD, Namespace
from rdflib.plugins.stores.sparqlstore import SPARQLUpdateStore

from lib.graph_viz import save_rdfs_visualization, visualize_rdfs


def add_tbox(g: Graph, namespace: Namespace, add_type_axioms: bool = True):
    P = namespace  # To simplify the code

    if add_type_axioms:
        g.add((P.Author, RDF.type, RDFS.Class))
        g.add((P.City, RDF.type, RDFS.Class))
        g.add((P.Journal, RDF.type, RDFS.Class))
        g.add((P.Paper, RDF.type, RDFS.Class))
        g.add((P.PaperTopic, RDF.type, RDFS.Class))
        g.add((P.PublicationVenue, RDF.type, RDFS.Class))
        g.add((P.JournalVolume, RDF.type, RDFS.Class))
        g.add((P.Proceedings, RDF.type, RDFS.Class))
        g.add((P.Review, RDF.type, RDFS.Class))
        g.add((P.ScientificForum, RDF.type, RDFS.Class))
        g.add((P.Conference, RDF.type, RDFS.Class))
        g.add((P.Workshop, RDF.type, RDFS.Class))

    g.add((P.JournalVolume, RDFS.subClassOf, P.PublicationVenue))
    g.add((P.Proceedings, RDFS.subClassOf, P.PublicationVenue))
    g.add((P.Conference, RDFS.subClassOf, P.ScientificForum))
    g.add((P.Workshop, RDFS.subClassOf, P.ScientificForum))

    # Properties
    if add_type_axioms:
        g.add((P.editionWasHeldIn, RDF.type, RDF.Property))
        g.add((P.isProceedingsOf, RDF.type, RDF.Property))
        g.add((P.isProceedingsOfConference, RDF.type, RDF.Property))
        g.add((P.isProceedingsOfWorkshop, RDF.type, RDF.Property))
        g.add((P.isPublishedIn, RDF.type, RDF.Property))
        g.add((P.isPublishedInJournalVolume, RDF.type, RDF.Property))
        g.add((P.isPublishedInProceedings, RDF.type, RDF.Property))
        g.add((P.isVolumeOf, RDF.type, RDF.Property))
        g.add((P.paperCites, RDF.type, RDF.Property))
        g.add((P.paperIsAbout, RDF.type, RDF.Property))
        g.add((P.reviewIsAbout, RDF.type, RDF.Property))
        g.add((P.writesPaper, RDF.type, RDF.Property))
        g.add((P.isCorrespondingAuthor, RDF.type, RDF.Property))
        g.add((P.writesReview, RDF.type, RDF.Property))

    g.add((P.editionWasHeldIn, RDFS.domain, P.Proceedings))
    g.add((P.editionWasHeldIn, RDFS.range, P.City))

    g.add((P.isProceedingsOf, RDFS.domain, P.Proceedings))
    g.add((P.isProceedingsOf, RDFS.range, P.ScientificForum))

    g.add((P.isProceedingsOfConference, RDFS.subPropertyOf, P.isProceedingsOf))
    g.add((P.isProceedingsOfConference, RDFS.domain, P.Proceedings))
    g.add((P.isProceedingsOfConference, RDFS.range, P.Conference))

    g.add((P.isProceedingsOfWorkshop, RDFS.subPropertyOf, P.isProceedingsOf))
    g.add((P.isProceedingsOfWorkshop, RDFS.domain, P.Proceedings))
    g.add((P.isProceedingsOfWorkshop, RDFS.range, P.Workshop))

    g.add((P.isPublishedIn, RDFS.domain, P.Paper))
    g.add((P.isPublishedIn, RDFS.range, P.PublicationVenue))

    g.add((P.isPublishedInJournalVolume, RDFS.subPropertyOf, P.isPublishedIn))
    g.add((P.isPublishedInJournalVolume, RDFS.domain, P.Paper))
    g.add((P.isPublishedInJournalVolume, RDFS.range, P.JournalVolume))

    g.add((P.isPublishedInProceedings, RDFS.subPropertyOf, P.isPublishedIn))
    g.add((P.isPublishedInProceedings, RDFS.domain, P.Paper))
    g.add((P.isPublishedInProceedings, RDFS.range, P.Proceedings))

    g.add((P.isVolumeOf, RDFS.domain, P.JournalVolume))
    g.add((P.isVolumeOf, RDFS.range, P.Journal))

    g.add((P.paperCites, RDFS.domain, P.Paper))
    g.add((P.paperCites, RDFS.range, P.Paper))

    g.add((P.paperIsAbout, RDFS.domain, P.Paper))
    g.add((P.paperIsAbout, RDFS.range, P.PaperTopic))

    g.add((P.reviewIsAbout, RDFS.domain, P.Review))
    g.add((P.reviewIsAbout, RDFS.range, P.Paper))

    g.add((P.writesPaper, RDFS.domain, P.Author))
    g.add((P.writesPaper, RDFS.range, P.Paper))

    g.add((P.isCorrespondingAuthor, RDFS.subPropertyOf, P.writesPaper))
    g.add((P.isCorrespondingAuthor, RDFS.domain, P.Author))
    g.add((P.isCorrespondingAuthor, RDFS.range, P.Paper))

    g.add((P.writesReview, RDFS.domain, P.Author))
    g.add((P.writesReview, RDFS.range, P.Review))

    # Literals

    if add_type_axioms:
        g.add((P.authorName, RDF.type, RDF.Property))
        g.add((P.reviewVerdict, RDF.type, RDF.Property))
        g.add((P.reviewContent, RDF.type, RDF.Property))
        g.add((P.paperTitle, RDF.type, RDF.Property))
        g.add((P.paperAbstract, RDF.type, RDF.Property))
        g.add((P.paperContent, RDF.type, RDF.Property))
        g.add((P.topicKeyword, RDF.type, RDF.Property))
        g.add((P.venueName, RDF.type, RDF.Property))
        g.add((P.journalName, RDF.type, RDF.Property))
        g.add((P.journalVolumeNumber, RDF.type, RDF.Property))
        g.add((P.journalVolumeYear, RDF.type, RDF.Property))
        g.add((P.proceedingsYear, RDF.type, RDF.Property))
        g.add((P.cityName, RDF.type, RDF.Property))
        g.add((P.scientificForumName, RDF.type, RDF.Property))
        g.add((P.conferenceName, RDF.type, RDF.Property))
        g.add((P.workshopName, RDF.type, RDF.Property))

    g.add((P.authorName, RDFS.domain, P.Author))
    g.add((P.authorName, RDFS.range, XSD.string))

    g.add((P.reviewVerdict, RDFS.domain, P.Review))
    g.add((P.reviewVerdict, RDFS.range, XSD.boolean))

    g.add((P.reviewContent, RDFS.domain, P.Review))
    g.add((P.reviewContent, RDFS.range, XSD.string))

    g.add((P.paperTitle, RDFS.domain, P.Paper))
    g.add((P.paperTitle, RDFS.range, XSD.string))

    g.add((P.paperAbstract, RDFS.domain, P.Paper))
    g.add((P.paperAbstract, RDFS.range, XSD.string))

    g.add((P.paperContent, RDFS.domain, P.Paper))
    g.add((P.paperContent, RDFS.range, XSD.string))

    g.add((P.topicKeyword, RDFS.domain, P.PaperTopic))
    g.add((P.topicKeyword, RDFS.range, XSD.string))

    g.add((P.venueName, RDFS.domain, P.PublicationVenue))
    g.add((P.venueName, RDFS.range, XSD.string))

    g.add((P.journalName, RDFS.domain, P.Journal))
    g.add((P.journalName, RDFS.range, XSD.string))

    g.add((P.journalVolumeNumber, RDFS.domain, P.JournalVolume))
    g.add((P.journalVolumeNumber, RDFS.range, XSD.unsignedInt))

    g.add((P.journalVolumeYear, RDFS.domain, P.JournalVolume))
    g.add((P.journalVolumeYear, RDFS.range, XSD.unsignedInt))

    g.add((P.proceedingsYear, RDFS.domain, P.Proceedings))
    g.add((P.proceedingsYear, RDFS.range, XSD.unsignedInt))

    g.add((P.cityName, RDFS.domain, P.City))
    g.add((P.cityName, RDFS.range, XSD.string))

    g.add((P.scientificForumName, RDFS.domain, P.ScientificForum))
    g.add((P.scientificForumName, RDFS.range, XSD.string))

    g.add((P.conferenceName, RDFS.subPropertyOf, P.scientificForumName))
    g.add((P.conferenceName, RDFS.domain, P.Conference))
    g.add((P.conferenceName, RDFS.range, XSD.string))

    g.add((P.workshopName, RDFS.subPropertyOf, P.scientificForumName))
    g.add((P.workshopName, RDFS.domain, P.Workshop))
    g.add((P.workshopName, RDFS.range, XSD.string))


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
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="If provided, saves the generated TBox to this file in Turtle format.",
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

    add_tbox(g, P, add_type_axioms=args.add_type_axioms)

    if args.save is not None:
        g.serialize(args.save, format="turtle")
    visualize_rdfs(g)
    save_rdfs_visualization(g, "img/tbox.png")
