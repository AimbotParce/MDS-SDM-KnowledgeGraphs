from typing import List, Optional, TypedDict


class RawContextWithIntent(TypedDict):
    context: str
    intents: List[str]


class RawCitedPaper(TypedDict):
    paperId: str


class RawCitingPaper(TypedDict):
    paperId: str


class RawReference(TypedDict):
    isInfluential: bool
    contextsWithIntent: List[RawContextWithIntent]
    citedPaper: RawCitedPaper
    citingPaper: RawCitingPaper
