from typing import List, Optional, TypedDict


class RawPublicationVenue(TypedDict):
    id: str
    name: str
    type: Optional[str]
    alternate_names: List[str]
    issn: str
    alternate_issns: List[str]
    url: str
    alternate_urls: List[str]


class RawOpenAccessPdf(TypedDict):
    url: str
    status: str
    license: Optional[str]
    disclaimer: str


class RawTLDR(TypedDict):
    model: str
    text: str


class RawJournal(TypedDict):
    name: str
    pages: str
    volume: str


class RawEmbedding(TypedDict):
    model: str
    vector: List[float]


class RawAuthor(TypedDict):
    authorId: str
    url: str
    name: str
    affiliations: List[str]
    homepage: Optional[str]
    hIndex: int


class RawPaper(TypedDict):
    paperId: str
    publicationVenue: Optional[RawPublicationVenue]
    url: str
    title: str
    year: int
    isOpenAccess: bool
    openAccessPdf: RawOpenAccessPdf
    fieldsOfStudy: List[str]
    tldr: Optional[RawTLDR]
    publicationTypes: List[str]
    journal: Optional[RawJournal]
    embedding: RawEmbedding
    authors: List[RawAuthor]
    abstract: Optional[str]
