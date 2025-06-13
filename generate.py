import csv
import glob
import json
import logging
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import List

import requests
import yake
from tqdm import tqdm

from lib.io import BatchedWriter

logger = logging.getLogger(__name__)


def yieldFromCSVFiles(files: List[Path]):
    """
    Loads CSV files sequentially and yields the rows one by one in a dictionary format.
    """
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield row


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic data for the academic graph.")
    parser.add_argument(
        "types",
        type=str,
        nargs="+",
        help="Data types to generate",
        choices=["reviews", "cities", "proceedings-cities", "keywords"],
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output directory where the already prepared dataset is stored",
        required=True,
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=float("inf"),
        help="Batch size to write the generated data",
    )
    args = parser.parse_args()

    types: list[str] = args.types

    batch_size: int = args.batch_size
    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    if "reviews" in types:
        logger.info("Generating reviews")
        # Check if there exists already a "papers" csv file, as well as an "authors"
        # csv file, and a "wrote" csv file
        papers_files = sorted(output_dir.glob("nodes-papers-*.csv"))
        authors_files = sorted(output_dir.glob("nodes-authors-*.csv"))
        wrote_files = sorted(output_dir.glob("edges-wrote-*.csv"))

        if not papers_files:
            logger.error("No papers files found in the output directory")
            exit(1)
        if not authors_files:
            logger.error("No authors files found in the output directory")
            exit(1)
        if not wrote_files:
            logger.error("No wrote files found in the output directory")
            exit(1)

        random.seed(42)  # For reproducibility, we'll set a seed here

        with BatchedWriter(output_dir / "edges-reviewed-{batch}.csv", batch_size) as output_file:
            writer = csv.DictWriter(output_file, fieldnames=["authorID", "paperID"])
            writer.writeheader()

            author_pool: set[str] = set()
            # We'll create an author pool and we'll use it to generate the reviews
            for author in yieldFromCSVFiles(authors_files):
                author_pool.add(author["authorID"])
            author_pool = list(sorted(author_pool))  # Need to sort here for reproducibility

            authorship_generator = yieldFromCSVFiles(wrote_files)
            # We'll use a small trick here. Because we know that the authorship files, the authors and the papers
            # files were all generated in the same order, we can assume that the authorship files
            # are sorted in the same order as the papers were generated. Thus, we don't need to
            # perform a full-scale join here, we can iterate over both files in parallel.
            # We'll use the authorship to exclude the authors of the paper from the reviews

            total_papers = 0
            total_reviews = 0
            try:
                last_author = next(authorship_generator)
            except StopIteration:
                logger.error("No authorship files found in the output directory")
                exit(1)
            for paper in tqdm(yieldFromCSVFiles(papers_files), desc="Preparing Reviews", unit="reviews", leave=False):
                total_papers += 1
                paper_id = paper["paperID"]

                # Get the authors of the paper
                this_paper_authors: set[str] = set()
                # We'll use this to exclude the authors of the paper from the reviews
                while last_author["paperID"] == paper_id:
                    this_paper_authors.add(last_author["authorID"])
                    try:
                        last_author = next(authorship_generator)
                    except StopIteration:
                        break
                # Generate the reviews
                # We'll generate between 3 and 5 reviews per paper
                num_reviews = random.randint(3, 5)
                reviewers: set[str] = set()
                while len(reviewers) < num_reviews:
                    reviewer = random.choice(author_pool)
                    if reviewer not in reviewers and reviewer not in this_paper_authors:
                        reviewers.add(reviewer)
                # Write the reviews
                for reviewer in reviewers:
                    total_reviews += 1
                    writer.writerow({"authorID": reviewer, "paperID": paper_id})

        logger.info(f"Generated {total_reviews} reviews for {total_papers} papers")
    if "cities" in types:
        logger.info("Generating cities")

        # Construct a pool of cities from the "cities API"
        url = "https://countriesnow.space/api/v0.1/countries"
        for _ in range(5):
            try:
                response = requests.get(url)
                response.raise_for_status()
                break
            except requests.RequestException as e:
                logger.warning(f"Error fetching data from {url}: {e}. Retrying in 1s...")
                time.sleep(1)
        else:
            logger.error("Failed to fetch data from the API after 5 attempts.")
            exit(1)
        try:
            data = response.json()
        except:
            logger.error("Failed to parse JSON response from the API.")
            exit(1)
        if not data.get("data"):
            logger.error("No data found in the API response.")
            exit(1)
        with BatchedWriter(output_dir / "nodes-cities-{batch}.csv", batch_size) as output_file:
            writer = csv.DictWriter(output_file, fieldnames=["name"])
            writer.writeheader()

            cities = set()  # To avoid duplicates
            for country in data["data"]:
                country_name = country["country"]
                if not country_name == "Spain":
                    # We'll only include Spain for now, as it has too many cities
                    continue
                if "cities" in country:
                    for city in country["cities"]:
                        city_name = f"{country_name}/{city}"
                        cities.add(city_name)
                        writer.writerow({"name": city_name})
        logger.info(f"Generated {len(cities)} cities")

    if "proceedings-cities" in types:
        logger.info("Generating proceedings' cities")
        # Check if there exists already a "proceedings" csv file,
        # as well as a "cities" csv file

        proceedings_files = sorted(output_dir.glob("nodes-proceedings-*.csv"))
        cities_files = sorted(output_dir.glob("nodes-cities-*.csv"))

        if not proceedings_files:
            logger.error("No proceedings files found in the output directory")
            exit(1)
        if not cities_files:
            logger.error("No cities files found in the output directory")
            exit(1)

        # We'll use the cities files to generate the proceedings' cities
        cities: set[str] = set()
        for city in yieldFromCSVFiles(cities_files):
            cities.add(city["name"])
        cities = list(sorted(cities))  # Need to sort here for reproducibility

        random.seed(42)  # For reproducibility, we'll set a seed here

        with BatchedWriter(output_dir / "edges-isheldin-{batch}.csv", batch_size) as output_file:
            writer = csv.DictWriter(output_file, fieldnames=["proceedingsID", "city"])
            writer.writeheader()

            total_proceedings = 0

            # We'll use the proceedings files to generate the proceedings' cities
            for proceeding in tqdm(
                yieldFromCSVFiles(proceedings_files),
                desc="Preparing Proceedings' Cities",
                unit="proceedings",
                leave=False,
            ):
                proceeding_id = proceeding["proceedingsID"]
                # Generate a random city for the proceeding
                city = random.choice(cities)
                writer.writerow({"proceedingsID": proceeding_id, "city": city})
                total_proceedings += 1

        logger.info(f"Generated {total_proceedings} proceedings' cities")

    if "keywords" in types:
        logger.info("Generating publications' keywords")

        # Check if there exists already a "papers" csv file.
        papers_files = sorted(output_dir.glob("nodes-papers-*.csv"))
        if not papers_files:
            logger.error("No papers files found in the output directory")
            exit(1)

        kw_extractor = yake.KeywordExtractor(lan="en", n=2, top=5, dedupLim=0.9)

        with (
            BatchedWriter(output_dir / "edges-haskeyword-{batch}.csv", batch_size) as kw_edges_output_file,
            BatchedWriter(output_dir / "nodes-keywords-{batch}.csv", batch_size) as kw_output_file,
        ):
            kw_edges_writer = csv.DictWriter(kw_edges_output_file, fieldnames=["paperID", "keyword"])
            kw_edges_writer.writeheader()
            kw_nodes_writer = csv.DictWriter(kw_output_file, fieldnames=["name"])
            kw_nodes_writer.writeheader()

            unique_keywords: set[str] = set()
            for paper in tqdm(yieldFromCSVFiles(papers_files), desc="Preparing Keywords", unit="papers", leave=False):
                paper_id = paper["paperID"]
                title = paper["title"]
                tldr = paper.get("tldr", "")  # Could be missing
                abstract = paper.get("abstract", "")  # Could be missing

                # Combine title with tldr if available, otherwise fallback to abstract
                if isinstance(tldr, str) and tldr.strip():
                    combined_text = f"{title} {tldr}"
                elif isinstance(abstract, str) and abstract.strip():
                    combined_text = f"{title} {abstract}"
                else:
                    combined_text = f"{title}"

                if combined_text.strip():
                    keywords = kw_extractor.extract_keywords(combined_text)
                    for keyword, _ in keywords:
                        keyword = keyword.strip().lower().capitalize()
                        if keyword not in unique_keywords:
                            unique_keywords.add(keyword)
                            kw_nodes_writer.writerow({"name": keyword})
                        kw_edges_writer.writerow({"paperID": paper_id, "keyword": keyword})
