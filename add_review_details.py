import logging
import os
import random
from typing import Generator, List

logger = logging.getLogger(__name__)

REVIEWS_QUERY = "MATCH (:Author)-[r:Reviewed]->(:Publication) RETURN ELEMENTID(r) AS reviewID"
UPDATE_QUERY = (
    "MATCH (:Author)-[r:Reviewed]->(:Publication) "
    "WHERE ELEMENTID(r)='{review_id}' "
    "SET r.accepted=toBoolean({accepted}) "
    "SET r.minorRevisions=toInteger({minor_revisions}) "
    "SET r.majorRevisions=toInteger({major_revisions}) "
    "SET r.reviewContent='{review_content}' "
)


def dummy_bulk_retrieve_review_details(review_ids: List[str]) -> Generator[dict, None, None]:
    """
    Dummy function to simulate the retrieval of review details.
    In a real scenario, this would call an external API.
    """
    for review_id in review_ids:
        # Simulate a successful response with dummy data
        res = {
            "review_id": review_id,
            "accepted": bool(random.randint(0, 1)),
            "minor_revisions": random.randint(0, 20),
            "major_revisions": random.randint(0, 5),
            "review_content": "This is a dummy review content.",
        }
        if random.random() < 0.1:
            res["review_content"] = None  # Simulate a case where review content is not available
        yield res


def main(args):
    neo4j = GraphDatabase.driver(
        os.getenv("NEO4J_URL", "neo4j://localhost:7687"),
        auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
    )

    total_reviews_updated = 0

    with neo4j.session() as session:
        # Find all the authors in the database
        result = session.run(REVIEWS_QUERY)
        review_ids: List[str] = [record["reviewID"] for record in result]
        logger.info(f"Found {len(review_ids)} reviews in the database")
        # Get the author details from the API
        for details in dummy_bulk_retrieve_review_details(review_ids):
            if details is None:
                logger.warning("No details found for review")
                continue

            review_id = details["review_id"]
            accepted = details["accepted"]
            minor_revisions = details["minor_revisions"]
            major_revisions = details["major_revisions"]
            review_content = details["review_content"]
            session.run(
                UPDATE_QUERY.format(
                    review_id=review_id,
                    accepted=accepted,
                    minor_revisions=minor_revisions,
                    major_revisions=major_revisions,
                    review_content=review_content,
                )
            )
            total_reviews_updated += 1

    logger.info(f"Updated {total_reviews_updated} reviews in the database")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Add affiliations to authors")
    parser.add_argument("--max-retries", type=int, default=3, help="Maximum number of retries for API requests")
    parser.add_argument("--dry-run", action="store_true", help="Don't download anything")
    args = parser.parse_args()

    main(args)
