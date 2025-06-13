import logging
import time
from typing import Any, TypeAlias, Union

import requests

logger = logging.getLogger(__name__)

JSON: TypeAlias = Union[dict, list]


class SemanticScholarAPI(object):
    def __init__(self, api_url: str, api_key: str = None, default_max_retries: int = 1, default_backoff: float = 2):
        self.api_url = api_url
        self.api_key = api_key

        self.default_max_retries = default_max_retries
        self.default_backoff = default_backoff

    def get(self, endpoint: str, params: dict = None, max_retries: int = None, backoff: float = None) -> JSON:
        return self._request("GET", endpoint, params=params, max_retries=max_retries, backoff=backoff)

    def post(
        self, endpoint: str, params: dict = None, json: Any = None, max_retries: int = None, backoff: float = None
    ) -> JSON:
        return self._request("POST", endpoint, params=params, json=json, max_retries=max_retries, backoff=backoff)

    def _request(
        self,
        method: str,
        endpoint: str,
        params: dict = None,
        json: JSON = None,
        max_retries: int = None,
        backoff: float = None,
    ) -> JSON:
        if max_retries is None:
            max_retries = self.default_max_retries
        if backoff is None:
            backoff = self.default_backoff

        for attempt in range(1, max_retries + 1):
            try:
                try:
                    raw_res = requests.request(
                        method=method,
                        url=f"{self.api_url}/{endpoint}",
                        headers={"X-API-KEY": self.api_key},
                        params=params,
                        json=json,
                    )
                    raw_res.raise_for_status()
                except requests.exceptions.HTTPError as e:
                    try:
                        json_res = raw_res.json()
                    except Exception:
                        logger.error(f"Error from {self.api_url}/{endpoint}: {e}")
                    else:
                        logger.error(
                            f"Error from {self.api_url}/{endpoint}: {json_res.get('message', json_res.get('error', 'Unknown error'))}"
                        )
                    raise e
                try:
                    json_res = raw_res.json()
                except Exception as e:
                    logger.error(f"Error decoding response from {self.api_url}/{endpoint}: {e}")
                    raise e

                if "message" in json_res:
                    logger.error(f"Error from {self.api_url}/{endpoint}: {json_res['message']}")
                    raise Exception(json_res["message"])
                return json_res
            except Exception as e:
                if attempt < max_retries:
                    logger.info(f"Retrying {endpoint} after {backoff**attempt} seconds")
                    time.sleep(backoff**attempt)
        raise MaxRetriesException(f"API request failed after {max_retries} retries: {self.api_url}/{endpoint}")


class MaxRetriesException(Exception):
    pass
