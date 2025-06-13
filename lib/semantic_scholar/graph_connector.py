import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Literal, Optional, Tuple, Union, overload

from more_itertools import batched

from .api_connector import SemanticScholarAPI

logger = logging.getLogger(__name__)


class S2GraphAPI(SemanticScholarAPI):
    MAX_BATCH_SIZE = 500
    MAX_DATA_RETRIEVAL = 10_000

    def __init__(
        self,
        api_url: str = "https://api.semanticscholar.org/graph/v1",
        api_key: str = None,
        default_max_retries: int = 1,
        default_backoff: float = 2,
    ):
        super().__init__(api_url, api_key, default_max_retries, default_backoff)

    @overload
    def bulk_retrieve_papers(
        self,
        query: str,
        token: str = None,
        fields: list[str] = None,
        sort: str = None,
        publicationTypes: list[str] = None,
        openAccessPdf: bool = False,
        minCitationCount: int = None,
        publicationDateOrYear: str = None,
        year: str = None,
        venue: list[str] = None,
        fieldsOfStudy: list[str] = None,
        limit: int = None,
    ) -> list[dict]: ...

    @overload
    def bulk_retrieve_papers(
        self,
        query: str,
        token: str = None,
        fields: list[str] = None,
        sort: str = None,
        publicationTypes: list[str] = None,
        openAccessPdf: bool = False,
        minCitationCount: int = None,
        publicationDateOrYear: str = None,
        year: str = None,
        venue: list[str] = None,
        fieldsOfStudy: list[str] = None,
        limit: int = None,
        stream: Literal[False] = False,
    ) -> list[dict]: ...

    @overload
    def bulk_retrieve_papers(
        self,
        query: str,
        token: str = None,
        fields: list[str] = None,
        sort: str = None,
        publicationTypes: list[str] = None,
        openAccessPdf: bool = False,
        minCitationCount: int = None,
        publicationDateOrYear: str = None,
        year: str = None,
        venue: list[str] = None,
        fieldsOfStudy: list[str] = None,
        limit: int = None,
        stream: Literal[True] = False,
    ) -> Generator[dict, None, None]: ...

    def bulk_retrieve_papers(
        self,
        query: str,
        token: str = None,
        fields: list[str] = None,
        sort: str = None,
        publicationTypes: list[str] = None,
        openAccessPdf: bool = False,
        minCitationCount: int = None,
        publicationDateOrYear: str = None,
        year: str = None,
        venue: list[str] = None,
        fieldsOfStudy: list[str] = None,
        limit: int = None,
        stream: bool = False,
    ):
        """
        Retrieve a list of papers based on a query.

        Args:
            query (str): The query string to search for.
            token (str): The cursor token to use for pagination.
            fields (list[str]): The fields to return in the response.
            sort (str): The field to sort by.
            publicationTypes (list[str]): The publication types to filter by.
            openAccessPdf (bool): Whether to filter by open access PDF.
            minCitationCount (int): The minimum citation count to filter by.
            publicationDateOrYear (str): The publication date or year to filter by. (e.g. 2021-01-01, 2021 or 2015:2018)
            year (str): The year to filter by. (e.g. 2021 or 2015:2018)
            venue (list[str]): The venue to filter by.
            fieldsOfStudy (list[str]): The fields of study to filter by.

        Returns:
            papers (list[dict] | Generator[dict, None, None]): If stream is False, a list of papers.
            Otherwise, a generator of papers.
        """

        params = {"query": query}
        if token:
            params["token"] = token
        if fields:
            params["fields"] = ",".join(fields)
        if sort:
            params["sort"] = sort
        if publicationTypes:
            params["publicationTypes"] = ",".join(publicationTypes)
        if openAccessPdf:
            params["openAccessPdf"] = openAccessPdf
        if minCitationCount:
            params["minCitationCount"] = minCitationCount
        if publicationDateOrYear:
            params["publicationDateOrYear"] = publicationDateOrYear
        if year:
            params["year"] = year
        if venue:
            params["venue"] = ",".join(venue)
        if fieldsOfStudy:
            params["fieldsOfStudy"] = ",".join(fieldsOfStudy)

        def _paginate() -> Generator[List[Dict], None, None]:
            data = self.get("paper/search/bulk", params=params)
            if limit is not None and len(data["data"]) >= limit:
                # If the limit is reached, return only the first limit number of papers
                yield data["data"][0:limit]
                return
            yield data["data"]
            total_retrieved = len(data["data"])
            while data.get("token") and (limit is None or total_retrieved < limit):
                data = self.get("paper/search/bulk", params={**params, "token": data["token"]})
                if limit is not None and total_retrieved + len(data["data"]) >= limit:
                    yield data["data"][0 : limit - total_retrieved]
                    break
                yield data["data"]
                total_retrieved += len(data["data"])

        def _generator():
            for data in _paginate():
                yield from data

        if stream:
            return _generator()
        else:
            all_data = []
            for data in _paginate():
                all_data.extend(data)

            if limit is not None and len(all_data) > limit:
                all_data = all_data[:limit]
            return all_data

    @overload
    def bulk_retrieve_details(self, paper_ids: Iterable[str], fields: Iterable[str]) -> List[dict]: ...
    @overload
    def bulk_retrieve_details(
        self, paper_ids: Iterable[str], fields: Iterable[str], stream: Literal[False]
    ) -> List[dict]: ...
    @overload
    def bulk_retrieve_details(
        self, paper_ids: Iterable[str], fields: Iterable[str], stream: Literal[True]
    ) -> Generator[dict, None, None]: ...

    # This function returns a list of paper_details (dict/json) that can be identified by paperId
    def bulk_retrieve_details(self, paper_ids: Iterable[str], fields: Iterable[str], stream: bool = False):
        """
        Retrieve the details for a list of papers.

        Args:
            paper_ids (list[str]): The list of paper IDs to retrieve details for.
            fields (list[str]): The fields to return in the response.
            stream (bool): Whether to stream the results.

        Returns:
            details (Generator[dict, None, None] | List[dict]): If stream is False, a list of paper details.
            Otherwise, a generator of paper details.
        """

        def _download_chunk(chunk: list[str]) -> List[dict]:
            return self.post("paper/batch", params={"fields": ",".join(fields)}, json={"ids": chunk})

        def _generator():
            for paper_chunk in batched(paper_ids, self.MAX_BATCH_SIZE):
                papers: list[dict] = _download_chunk(paper_chunk)
                yield from papers

        if stream:
            return _generator()
        else:
            all_papers: list[dict] = []
            for paper_chunk in batched(paper_ids, self.MAX_BATCH_SIZE):
                papers = _download_chunk(paper_chunk)
                all_papers.extend(papers)
            return all_papers

    @overload
    def retrieve_citations(self, paper_id: str, fields: list[str]) -> List[dict]: ...
    @overload
    def retrieve_citations(self, paper_id: str, fields: list[str], stream: Literal[False]) -> List[dict]: ...
    @overload
    def retrieve_citations(
        self, paper_id: str, fields: list[str], stream: Literal[True]
    ) -> Generator[dict, None, None]: ...

    def retrieve_citations(
        self, paper_id: str, fields: list[str], stream: bool = False
    ) -> Union[List[dict], Generator[dict, None, None]]:
        """
        Retrieve the citations for a paper.

        Args:
            paper_id (str): The paper ID to retrieve citations for.
            fields (list[str]): The fields to return in the response.
            stream (bool): Whether to stream the results.

        Returns:
            citations (list[dict] | Generator[dict, None, None]): If stream is False, a list of citations.
            Otherwise, a generator of citations.
        """
        params = {"fields": ",".join(fields)}

        def _paginate() -> Generator[List[Dict], None, None]:
            data = self.get(f"paper/{paper_id}/citations", params=params)
            yield data["data"]
            while data.get("next"):
                if data.get("next") >= self.MAX_DATA_RETRIEVAL - 1:
                    # This seems to be a hard limit of the Semantic Scholar API
                    logger.warning(
                        f"Citation count exceeds {self.MAX_DATA_RETRIEVAL}. "
                        f"Only the first {self.MAX_DATA_RETRIEVAL} citations will be retrieved."
                    )
                    break
                data = self.get(
                    f"paper/{paper_id}/citations",
                    params={
                        **params,
                        "offset": data["next"],
                        "limit": min(self.MAX_BATCH_SIZE, self.MAX_DATA_RETRIEVAL - data["next"] - 1),
                    },
                )
                yield data["data"]

        def _generator():
            for data in _paginate():
                yield from data

        if stream:
            return _generator()
        else:
            all_data = []
            for data in _paginate():
                all_data.extend(data)
            return all_data

    @overload
    def retrieve_references(self, paper_id: str, fields: list[str]) -> List[dict]: ...
    @overload
    def retrieve_references(self, paper_id: str, fields: list[str], stream: Literal[False]) -> List[dict]: ...
    @overload
    def retrieve_references(
        self, paper_id: str, fields: list[str], stream: Literal[True]
    ) -> Generator[dict, None, None]: ...

    def retrieve_references(
        self, paper_id: str, fields: list[str], stream: bool = False
    ) -> Union[List[dict], Generator[dict, None, None]]:
        """
        Retrieve the references for a paper.

        Args:
            paper_id (str): The paper ID to retrieve references for.
            fields (list[str]): The fields to return in the response.
            stream (bool): Whether to stream the results.

        Returns:
            references (list[dict] | Generator[dict, None, None]): If stream is False, a list of references.
            Otherwise, a generator of references.
        """
        params = {"fields": ",".join(fields)}

        def _paginate() -> Generator[List[Dict], None, None]:
            data = self.get(f"paper/{paper_id}/references", params=params)
            yield data["data"]
            while data.get("next"):
                if data.get("next") >= self.MAX_DATA_RETRIEVAL - 1:
                    # This seems to be a hard limit of the Semantic Scholar API
                    logger.warning(
                        f"Reference count exceeds {self.MAX_DATA_RETRIEVAL}. "
                        f"Only the first {self.MAX_DATA_RETRIEVAL} references will be retrieved."
                    )
                    break
                data = self.get(
                    f"paper/{paper_id}/references",
                    params={
                        **params,
                        "offset": data["next"],
                        "limit": min(self.MAX_BATCH_SIZE, self.MAX_DATA_RETRIEVAL - data["next"] - 1),
                    },
                )
                yield data["data"]

        def _generator():
            for data in _paginate():
                yield from data

        if stream:
            return _generator()
        else:
            all_data = []
            for data in _paginate():
                all_data.extend(data)
            return all_data

    @overload
    def bulk_retrieve_citations(
        self, paper_ids: Iterable[str], fields: Iterable[str], stream: Literal[False]
    ) -> List[dict]: ...

    @overload
    def bulk_retrieve_citations(self, paper_ids: Iterable[str], fields: Iterable[str]) -> List[dict]: ...

    @overload
    def bulk_retrieve_citations(
        self, paper_ids: Iterable[str], fields: Iterable[str], stream: Literal[True]
    ) -> Generator[dict, None, None]: ...

    def bulk_retrieve_citations(self, paper_ids: Iterable[str], fields: Iterable[str], stream: bool = False):
        """
        Retrieve the citations for a list of papers.

        Args:
            paper_ids (list[str]): The list of paper IDs to retrieve citations for.
            fields (list[str]): The fields to return in the response.

        Returns:
            citations (Generator[dict, None, None] | List[dict]):
            If the batch size is None, a list of citations. Otherwise, a generator of citations.
        """

        def _download_citations(paper_id: str):
            citations = self.retrieve_citations(paper_id, fields, stream=stream)

            def _generator():
                for citation in citations:
                    citation["citedPaper"] = {"paperId": paper_id}
                    yield citation

            if stream:
                return _generator()
            else:
                all_citations: list[dict] = []
                for citation in citations:
                    citation["citedPaper"] = {"paperId": paper_id}
                    all_citations.append(citation)
                return all_citations

        def _generator():
            for paper_id in paper_ids:
                yield from _download_citations(paper_id)

        if not stream:
            all_citations: list[dict] = []
            for paper_id in paper_ids:
                citations = _download_citations(paper_id)
                all_citations.extend(citations)
            return all_citations
        else:
            return _generator()

    @overload
    def bulk_retrieve_references(
        self, paper_ids: Iterable[str], fields: Iterable[str], stream: Literal[False]
    ) -> List[dict]: ...

    @overload
    def bulk_retrieve_references(self, paper_ids: Iterable[str], fields: Iterable[str]) -> List[dict]: ...

    @overload
    def bulk_retrieve_references(
        self, paper_ids: Iterable[str], fields: Iterable[str], stream: Literal[True]
    ) -> Generator[dict, None, None]: ...

    def bulk_retrieve_references(self, paper_ids: list[str], fields: list[str], stream: bool = False):
        """
        Retrieve the references for a list of papers.

        Args:
            paper_ids (list[str]): The list of paper IDs to retrieve references for.
            fields (list[str]): The fields to return in the response.

        Returns:
            references (Generator[dict, None, None] | List[dict]):
            If the batch size is None, a list of references. Otherwise, a generator of references.
        """

        def _download_references(paper_id: str):
            references = self.retrieve_references(paper_id, fields, stream=stream)

            def _generator():
                for reference in references:
                    reference["citingPaper"] = {"paperId": paper_id}
                    yield reference

            if stream:
                return _generator()
            else:
                all_references = []
                for reference in references:
                    reference["citingPaper"] = {"paperId": paper_id}
                    all_references.append(reference)
                return all_references

        def _generator():
            for paper_id in paper_ids:
                yield from _download_references(paper_id)

        if not stream:
            all_references: list[dict] = []
            for paper_id in paper_ids:
                references = _download_references(paper_id)
                all_references.extend(references)
            return all_references
        else:
            return _generator()

    def retrieve_author_details(self, author_id: str, fields: list[str]) -> dict:
        """
        Retrieve the details for an author.

        Args:
            author_id (str): The author ID to retrieve details for.
            fields (list[str]): The fields to return in the response.

        Returns:
            details (dict): The author details.
        """
        params = {"fields": ",".join(fields)}
        return self.get(f"author/{author_id}", params=params)

    @overload
    def bulk_retrieve_author_details(
        self, author_ids: Iterable[str], fields: Iterable[str], stream: Literal[False]
    ) -> List[dict]: ...

    @overload
    def bulk_retrieve_author_details(self, author_ids: Iterable[str], fields: Iterable[str]) -> List[dict]: ...

    @overload
    def bulk_retrieve_author_details(
        self, author_ids: Iterable[str], fields: Iterable[str], stream: Literal[True]
    ) -> Generator[dict, None, None]: ...

    def bulk_retrieve_author_details(self, author_ids: list[str], fields: list[str], stream: bool = False):
        """
        Retrieve the details for a list of authors.

        Args:
            author_ids (list[str]): The list of author IDs to retrieve details for.
            fields (list[str]): The fields to return in the response.
            stream (bool): Whether to stream the results.

        Returns:
            details (Generator[dict, None, None] | List[dict]):
            If the batch size is None, a list of author details. Otherwise, a generator of author details.
        """

        def _download_author_batch(ids: list[str]):
            return self.post("author/batch", params={"fields": ",".join(fields)}, json={"ids": ids})

        def _generator():
            for author_batch in batched(author_ids, self.MAX_BATCH_SIZE):
                authors: list[dict] = _download_author_batch(author_batch)
                yield from authors

        if stream:
            return _generator()
        else:
            all_authors: list[dict] = []
            for author_batch in batched(author_ids, self.MAX_BATCH_SIZE):
                authors = _download_author_batch(author_batch)
                all_authors.extend(authors)
            return all_authors
