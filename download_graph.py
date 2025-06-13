import json
import logging
import os
from argparse import ArgumentParser
from pathlib import Path

from lib.io import BatchedWriter
from lib.semantic_scholar import S2GraphAPI

logger = logging.getLogger(__name__)


def main(args):
    connector = S2GraphAPI(api_key=os.getenv("S2_API_KEY"), default_max_retries=args.max_retries)

    if args.dry_run:
        logger.info("Running in DRY RUN mode. No data will be saved.")

    logger.info("Retrieving papers...")
    papers = connector.bulk_retrieve_papers(
        args.query,
        minCitationCount=args.min_citations,
        year=args.year,
        fieldsOfStudy=args.fields,
        limit=args.limit,
        sort="citationCount:desc",
        stream=True,
    )
    paper_ids: list[str] = list(paper["paperId"] for paper in papers)
    logger.info(f"Retrieved {len(paper_ids)} papers.")
    del papers  # Free up memory

    logger.info("Retrieving paper details...")
    missing_fields = (
        "paperId",
        "embedding",
        "tldr",
        "url",
        "title",
        "abstract",
        "year",
        "isOpenAccess",
        "openAccessPdf",
        "publicationTypes",
        # Author fields
        "authors.authorId",
        "authors.url",
        "authors.name",
        "authors.affiliations",
        "authors.homepage",
        "authors.hIndex",
        # Field of study fields
        "fieldsOfStudy",
        # Publication venue or journal
        "journal",
        "publicationVenue",
        # If the paper is published in a Journal, "journal" has the information about the page number and things
        # about the journal. Also, publicationVenue might have some extra information about it.
        # If it is published in the Proceedings of a conference, "publicationVenue" has the information about the
        # conference and "journal" might have some extra information about where in the proceedings the paper is.
    )
    paper_details = connector.bulk_retrieve_details(paper_ids, missing_fields, stream=True)
    if not args.dry_run:
        with BatchedWriter(args.output / "raw-papers-{batch}.jsonl", batch_size=args.batch_size) as writer:
            for i, paper in enumerate(paper_details, start=1):
                writer.write(json.dumps(paper, ensure_ascii=False) + "\n")
            logger.info(f"Stored {i} paper details.")
    else:
        for i, paper in enumerate(paper_details, start=1):
            pass
        logger.info(f"Retrieved {i} paper details.")

    # Get all the references
    logger.info("Retrieving references...")
    reference_fields = "citedPaper.paperId", "isInfluential", "contextsWithIntent"
    references = connector.bulk_retrieve_references(paper_ids, reference_fields, stream=True)
    if not args.dry_run:
        with BatchedWriter(args.output / "raw-references-{batch}.jsonl", batch_size=args.batch_size) as writer:
            for i, reference in enumerate(references, start=1):
                writer.write(json.dumps(reference, ensure_ascii=False) + "\n")
            logger.info(f"Stored {i} references.")
    else:
        for i, reference in enumerate(references, start=1):
            pass
        logger.info(f"Retrieved {i} references.")


if __name__ == "__main__":
    parser = ArgumentParser(description="Academic Paper Data Retrieval")
    parser.add_argument("query", type=str, help="Query to search for")
    parser.add_argument("--min-citations", type=int, default=None, help="Minimum number of citations")
    parser.add_argument("--year", type=str, help="Year of publication (YYYY | YYYY-YYYY | YYYY- | -YYYY)", default=None)
    parser.add_argument("--fields", type=str, help="Fields of study to filter by", nargs="*", default=None)
    parser.add_argument("--output", type=Path, help="Output folder", default=None)
    parser.add_argument("--limit", type=int, help="Limit the number of papers to retrieve", default=None)
    parser.add_argument("--batch-size", type=int, help="Batch size for retrieving details", default=float("inf"))
    parser.add_argument("--max-retries", type=int, help="Maximum number of retries for each request", default=3)
    parser.add_argument("--dry-run", action="store_true", help="Run without saving any data")
    args = parser.parse_args()
    if not args.dry_run and not args.output:
        parser.error("--output is required when not running in dry-run mode.")

    logging.basicConfig(
        level=logging.INFO,
        format="(%(asctime)s) %(levelname)s@%(name)s.%(funcName)s:%(lineno)d # %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main(args)
