import csv
import glob
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import List

from tqdm import tqdm

from lib.io import BatchedWriter

logger = logging.getLogger(__name__)


def yieldFromJSONLFiles(files: List[Path]):
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            yield from map(json.loads, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare the Semantic Scholar dataset")
    parser.add_argument(
        "input_files",
        type=str,
        nargs="+",
        help="Input JSONL files to prepare. Allows wildcards (e.g. *.jsonl)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output directory to write the prepared dataset batches to",
        required=True,
    )
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        choices=["papers", "citations", "references"],
        help="Type of file being loaded",
        required=True,
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=float("inf"),
        help="Batch size to write the prepared dataset",
    )
    args = parser.parse_args()

    input_files: list[str] = args.input_files
    # If input files have wildcards, expand them
    input_files = [Path(file) for pattern in input_files for file in glob.glob(pattern)]

    batch_size: int = args.batch_size
    file_type: str = args.type
    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    if file_type in ["citations", "references"]:  # They are the same
        errors: dict[str, set[str]] = defaultdict(set)  # Error: Paper IDs
        warnings: dict[str, set[str]] = defaultdict(set)  # Warning: Paper IDs
        with BatchedWriter(output_dir / "edges-citations-{batch}.csv", batch_size) as output_file:
            writer = csv.DictWriter(
                output_file, fieldnames=["citedPaperID", "citingPaperID", "isInfluential", "contextsWithIntent"]
            )
            writer.writeheader()
            iters = 0
            for citation in tqdm(
                yieldFromJSONLFiles(input_files), desc="Preparing Citations", unit="citations", leave=False
            ):
                if not citation.get("citedPaper").get("paperId"):
                    errors["Missing Cited Paper"].add(citation["citingPaper"]["paperId"])
                    continue

                writer.writerow(
                    {
                        "citedPaperID": citation["citedPaper"]["paperId"],
                        "citingPaperID": citation["citingPaper"]["paperId"],
                        "isInfluential": citation.get("isInfluential", False),
                        "contextsWithIntent": json.dumps(citation["contextsWithIntent"])
                        .replace("\n", " ")
                        .replace("\\", ""),
                    }
                )
                iters += 1
        logger.info(f"Prepared {iters} citations in {output_file.batch_number} batches")
        if warnings:
            logger.warning("The following warnings were found:")
            for warning, paper_ids in warnings.items():
                logger.warning(f"- {warning}: {len(paper_ids)}")
        if errors:
            logger.error("The following errors were found:")
            for error, paper_ids in errors.items():
                logger.error(f"- {error}: {len(paper_ids)}")
    elif file_type == "papers":
        papers = csv.DictWriter(
            BatchedWriter(output_dir / "nodes-papers-{batch}.csv", batch_size),
            fieldnames=[
                "paperID",
                "url",
                "title",
                "abstract",
                "year",
                "isOpenAccess",
                "openAccessPDFUrl",
                "publicationTypes",
                "embedding",
                "tldr",
            ],
        )
        papers.writeheader()
        fieldsofstudy = csv.DictWriter(
            BatchedWriter(output_dir / "nodes-fieldsofstudy-{batch}.csv", batch_size),
            fieldnames=["name"],
        )
        fieldsofstudy.writeheader()
        proceedings = csv.DictWriter(
            BatchedWriter(output_dir / "nodes-proceedings-{batch}.csv", batch_size),
            fieldnames=["proceedingsID", "year"],
        )
        proceedings.writeheader()
        journalvolumes = csv.DictWriter(
            BatchedWriter(output_dir / "nodes-journalvolumes-{batch}.csv", batch_size),
            fieldnames=["journalVolumeID", "volume"],
        )
        journalvolumes.writeheader()
        journals = csv.DictWriter(
            BatchedWriter(output_dir / "nodes-journals-{batch}.csv", batch_size),
            fieldnames=["journalID", "name", "url", "alternateNames"],
        )
        journals.writeheader()
        workshops = csv.DictWriter(
            BatchedWriter(output_dir / "nodes-workshops-{batch}.csv", batch_size),
            fieldnames=["workshopID", "name", "url", "alternateNames"],
        )
        workshops.writeheader()
        conferences = csv.DictWriter(
            BatchedWriter(output_dir / "nodes-conferences-{batch}.csv", batch_size),
            fieldnames=["conferenceID", "name", "url", "alternateNames"],
        )
        conferences.writeheader()
        cities = csv.DictWriter(
            BatchedWriter(output_dir / "nodes-cities-{batch}.csv", batch_size),
            fieldnames=["name"],
        )
        otherpublicationvenues = csv.DictWriter(
            BatchedWriter(output_dir / "nodes-otherpublicationvenues-{batch}.csv", batch_size),
            fieldnames=["venueID", "name", "url", "alternateNames"],
        )
        otherpublicationvenues.writeheader()
        cities.writeheader()
        authors = csv.DictWriter(
            BatchedWriter(output_dir / "nodes-authors-{batch}.csv", batch_size),
            fieldnames=["authorID", "url", "name", "homepage", "hIndex"],
        )
        authors.writeheader()
        organizations = csv.DictWriter(
            BatchedWriter(output_dir / "nodes-organizations-{batch}.csv", batch_size),
            fieldnames=["name"],
        )
        organizations.writeheader()
        hasfieldofstudy = csv.DictWriter(
            BatchedWriter(output_dir / "edges-hasfieldofstudy-{batch}.csv", batch_size),
            fieldnames=["paperID", "fieldOfStudy"],
        )
        hasfieldofstudy.writeheader()
        wrote = csv.DictWriter(
            BatchedWriter(output_dir / "edges-wrote-{batch}.csv", batch_size),
            fieldnames=["paperID", "authorID"],
        )
        wrote.writeheader()
        mainauthor = csv.DictWriter(
            BatchedWriter(output_dir / "edges-mainauthor-{batch}.csv", batch_size),
            fieldnames=["paperID", "authorID"],
        )
        mainauthor.writeheader()
        isaffiliatedwith = csv.DictWriter(
            BatchedWriter(output_dir / "edges-isaffiliatedwith-{batch}.csv", batch_size),
            fieldnames=["authorID", "organization"],
        )
        isaffiliatedwith.writeheader()
        reviewed = csv.DictWriter(
            BatchedWriter(output_dir / "edges-reviewed-{batch}.csv", batch_size),
            fieldnames=["paperID", "authorID", "accepted", "minorRevisions", "majorRevisions", "reviewContent"],
        )
        reviewed.writeheader()
        ispublishedinotherpublicationvenue = csv.DictWriter(
            BatchedWriter(output_dir / "edges-ispublishedinotherpublicationvenue-{batch}.csv", batch_size),
            fieldnames=["paperID", "venueID", "pages"],
        )
        ispublishedinotherpublicationvenue.writeheader()
        ispublishedinjournal = csv.DictWriter(
            BatchedWriter(output_dir / "edges-ispublishedinjournal-{batch}.csv", batch_size),
            fieldnames=["paperID", "journalVolumeID", "pages"],
        )
        ispublishedinjournal.writeheader()
        ispublishedinproceedings = csv.DictWriter(
            BatchedWriter(output_dir / "edges-ispublishedinproceedings-{batch}.csv", batch_size),
            fieldnames=["paperID", "proceedingsID", "pages"],
        )
        ispublishedinproceedings.writeheader()
        iseditionofjournal = csv.DictWriter(
            BatchedWriter(output_dir / "edges-iseditionofjournal-{batch}.csv", batch_size),
            fieldnames=["journalVolumeID", "journalID"],
        )
        iseditionofjournal.writeheader()
        iseditionofconference = csv.DictWriter(
            BatchedWriter(output_dir / "edges-iseditionofconference-{batch}.csv", batch_size),
            fieldnames=["proceedingsID", "conferenceID"],
        )
        iseditionofconference.writeheader()
        iseditionofworkshop = csv.DictWriter(
            BatchedWriter(output_dir / "edges-iseditionofworkshop-{batch}.csv", batch_size),
            fieldnames=["proceedingsID", "workshopID"],
        )
        iseditionofworkshop.writeheader()
        isheldin = csv.DictWriter(
            BatchedWriter(output_dir / "edges-isheldin-{batch}.csv", batch_size),
            fieldnames=["proceedingsID", "city"],
        )
        isheldin.writeheader()

        unique_fields_of_study = set()
        unique_proceedings_ids = set()
        unique_journal_volume_ids = set()
        unique_other_publication_venue_ids = set()
        unique_journal_ids = set()
        unique_workshop_ids = set()
        unique_conference_ids = set()
        unique_city_names = set()
        unique_author_ids = set()

        errors: dict[str, set[str]] = defaultdict(set)  # Error: Paper IDs
        warnings: dict[str, set[str]] = defaultdict(set)  # Warning: Paper IDs

        iters = 0
        for paper in tqdm(yieldFromJSONLFiles(input_files), desc="Preparing Papers", unit="papers", leave=False):
            papers.writerow(
                {
                    "paperID": paper["paperId"],
                    "url": paper["url"],
                    "title": paper["title"],
                    "abstract": paper["abstract"].replace("\n", " ") if paper["abstract"] else None,
                    "year": int(paper["year"]) if paper["year"] else None,
                    "isOpenAccess": paper["isOpenAccess"],
                    "openAccessPDFUrl": paper.get("openAccessPdfUrl"),
                    "publicationTypes": paper["publicationTypes"],
                    "embedding": json.dumps(paper.get("embedding")) if paper.get("embedding") else None,
                    "tldr": (
                        json.dumps(paper.get("tldr")).replace("\n", " ").replace("\\", "")
                        if paper.get("tldr")
                        else None
                    ),
                }
            )
            fields_of_study = paper.get("fieldsOfStudy", [])
            if not fields_of_study:
                warnings["Missing Paper Fields of Study"].add(paper["paperId"])
            else:
                for fos in fields_of_study:
                    if not fos in unique_fields_of_study:
                        fieldsofstudy.writerow({"name": fos})
                        unique_fields_of_study.add(fos)
                    hasfieldofstudy.writerow({"paperID": paper["paperId"], "fieldOfStudy": fos})
            for author in paper["authors"]:
                if not author["authorId"] in unique_author_ids:
                    if not author.get("authorId"):
                        errors["Missing Author ID"].add(paper["paperId"])
                        continue
                    if not author.get("name"):
                        errors["Missing Author Name"].add(paper["paperId"])
                        continue
                    if not author.get("url"):
                        errors["Missing Author URL"].add(paper["paperId"])
                        continue
                    authors.writerow(
                        {
                            "authorID": author["authorId"],
                            "url": author["url"],
                            "name": author["name"],
                            "homepage": author.get("homepage"),
                            "hIndex": author.get("hIndex"),
                        }
                    )
                    unique_author_ids.add(author["authorId"])
                wrote.writerow({"paperID": paper["paperId"], "authorID": author["authorId"]})

            if len(paper["authors"]) == 0:
                warnings["Missing Paper Authors"].add(paper["paperId"])
            else:
                main_author = paper["authors"][0]  # We'll assume the first author is the main author
                mainauthor.writerow({"paperID": paper["paperId"], "authorID": main_author["authorId"]})

            errors["Unknown Paper Review Details"].add(paper["paperId"])

            # Publications
            venue = paper["publicationVenue"]
            if venue is None:
                warnings["Missing Paper Publication Venue"].add(paper["paperId"])
            else:
                if not "type" in venue:
                    warnings["Missing Publication Venue Type"].add(paper["paperId"])
                    if not venue["id"] in unique_other_publication_venue_ids:
                        otherpublicationvenues.writerow(
                            {
                                "venueID": venue["id"],
                                "name": venue["name"],
                                "url": venue.get("url"),
                                "alternateNames": json.dumps(venue.get("alternate_names", [])),
                            }
                        )
                        unique_other_publication_venue_ids.add(venue["id"])
                    ispublishedinotherpublicationvenue.writerow(
                        {
                            "paperID": paper["paperId"],
                            "venueID": venue["id"],
                            "pages": paper.get("journal", {}).get("pages"),
                        }
                    )
                elif venue["type"] == "journal":
                    if not venue["id"] in unique_journal_ids:
                        journals.writerow(
                            {
                                "journalID": venue["id"],
                                "name": venue["name"],
                                "url": venue.get("url"),
                                "alternateNames": json.dumps(venue.get("alternate_names", [])),
                            }
                        )
                        unique_journal_ids.add(venue["id"])
                    if not paper.get("journal") or not paper["journal"].get("volume"):
                        warnings["Missing Journal Volume"].add(paper["paperId"])
                    else:
                        journal_volume_id = (venue["id"], paper["journal"].get("volume"))
                        if not journal_volume_id in unique_journal_volume_ids:
                            journalvolumes.writerow(
                                {
                                    "journalVolumeID": json.dumps(list(journal_volume_id)),
                                    "volume": paper["journal"].get("volume"),
                                }
                            )
                            unique_journal_volume_ids.add(journal_volume_id)
                            iseditionofjournal.writerow(
                                {"journalVolumeID": json.dumps(list(journal_volume_id)), "journalID": venue["id"]}
                            )
                        ispublishedinjournal.writerow(
                            {
                                "paperID": paper["paperId"],
                                "journalVolumeID": json.dumps(list(journal_volume_id)),
                                "pages": (
                                    paper["journal"].get("pages").replace("\n", "").replace(" ", "")
                                    if paper["journal"].get("pages")
                                    else None
                                ),
                            }
                        )
                elif venue["type"] == "conference":
                    if not venue["id"] in unique_conference_ids:
                        conferences.writerow(
                            {
                                "conferenceID": venue["id"],
                                "name": venue["name"],
                                "url": venue.get("url"),
                                "alternateNames": json.dumps(venue.get("alternate_names", [])),
                            }
                        )
                        unique_conference_ids.add(venue["id"])
                    proceedings_id = (venue["id"], paper["year"])
                    if not proceedings_id in unique_proceedings_ids:
                        proceedings.writerow({"year": paper["year"], "proceedingsID": json.dumps(list(proceedings_id))})
                        unique_proceedings_ids.add(proceedings_id)
                        iseditionofconference.writerow(
                            {"proceedingsID": json.dumps(list(proceedings_id)), "conferenceID": venue["id"]}
                        )
                        errors["Unknown Proceedings City"].add(venue["id"])

                    ispublishedinproceedings.writerow(
                        {
                            "paperID": paper["paperId"],
                            "proceedingsID": json.dumps(list(proceedings_id)),
                            "pages": paper.get("journal", {}).get("pages") if paper.get("journal") else None,
                        }
                    )
                elif venue["type"] == "workshop":
                    if not venue["id"] in unique_workshop_ids:
                        workshops.writerow(
                            {
                                "workshopID": venue["id"],
                                "name": venue["name"],
                                "url": venue.get("url"),
                                "alternateNames": json.dumps(venue.get("alternate_names", [])),
                            }
                        )
                        unique_workshop_ids.add(venue["id"])
                    proceedings_id = (venue["id"], paper["year"])
                    if not proceedings_id in unique_proceedings_ids:
                        proceedings.writerow({"year": paper["year"], "proceedingsID": json.dumps(list(proceedings_id))})
                        unique_proceedings_ids.add(proceedings_id)
                        iseditionofworkshop.writerow(
                            {"proceedingsID": json.dumps(list(proceedings_id)), "workshopID": venue["id"]}
                        )
                        errors["Unknown Proceedings City"].add(venue["id"])

                    ispublishedinproceedings.writerow(
                        {
                            "paperID": paper["paperId"],
                            "proceedingsID": json.dumps(list(proceedings_id)),
                            "pages": paper.get("journal", {}).get("pages"),
                        }
                    )
                else:
                    errors["Unknown Publication Venue Type"].add(paper["paperId"])

            iters += 1

        logger.info(f"Prepared {iters} papers.")
        if warnings:
            logger.warning("The following warnings were found:")
            for warning, paper_ids in warnings.items():
                logger.warning(f"- {warning}: {len(paper_ids)}")
        if errors:
            logger.error("The following errors were found:")
            for error, paper_ids in errors.items():
                logger.error(f"- {error}: {len(paper_ids)}")
    else:
        raise ValueError(f"Unknown file type: {file_type}")
