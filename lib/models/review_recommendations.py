from typing import Dict, List, TypedDict, TypeGuard


class PaperInfo(TypedDict):
    topics: List[str]
    publication_venue: str
    authors: List[str]
    references: List[str]
    citations: List[str]


def is_paper_info(paper_info: Dict) -> TypeGuard[PaperInfo]:
    """
    Check if the given dictionary matches the PaperInfo structure.
    """
    required_keys = {"topics", "publication_venue", "authors", "references", "citations"}
    return (
        isinstance(paper_info, dict)
        and required_keys.issubset(paper_info.keys())
        and all(isinstance(v, list) for k, v in paper_info.items() if k != "publication_venue")
        and isinstance(paper_info["publication_venue"], str)
        and all(isinstance(item, str) and " " not in item and "/" not in item for item in paper_info["topics"])
        and all(isinstance(item, str) and " " not in item and "/" not in item for item in paper_info["authors"])
        and all(isinstance(item, str) and " " not in item and "/" not in item for item in paper_info["references"])
        and all(isinstance(item, str) and " " not in item and "/" not in item for item in paper_info["citations"])
    )
