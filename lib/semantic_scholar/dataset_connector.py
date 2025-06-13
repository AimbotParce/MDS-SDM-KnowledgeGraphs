import logging
import urllib.request
from pathlib import Path
from typing import TypedDict

from tqdm import tqdm

from .api_connector import SemanticScholarAPI

logger = logging.getLogger(__name__)


class S2DatasetAPI(SemanticScholarAPI):
    def __init__(
        self,
        api_url: str = "http://api.semanticscholar.org/datasets/v1",
        api_key: str = None,
        default_max_retries: int = 1,
        default_backoff: float = 2,
    ):
        super().__init__(api_url, api_key, default_max_retries, default_backoff)
        self._releases: dict[str, Release] = {}

    def getReleaseIDs(self) -> list[str]:
        return self.get("release")  # No cache. They shouldn't change often, but if they do, we'll get the latest info

    def getRelease(self, release_id: str) -> "Release":
        if release_id in self._releases:
            return self._releases[release_id]
        else:
            self._releases[release_id] = Release(self.get(f"release/{release_id}"), self)
            return self._releases[release_id]


class PrunedDatasetData(TypedDict):
    name: str
    description: str
    README: str


class DatasetData(PrunedDatasetData):
    files: list[str]


class ReleaseData(TypedDict):
    release_id: str
    datasets: list[PrunedDatasetData]
    README: str


class Release(object):
    def __init__(self, data: ReleaseData, api: S2DatasetAPI):
        self.data = data
        self.api = api

        self._datasets = {}

    @property
    def release_id(self):
        return self.data["release_id"]

    @property
    def README(self):
        return self.data["README"]

    def getDatasetNames(self):
        return [dataset["name"] for dataset in self.data["datasets"]]

    def getDataset(self, dataset_name: str) -> "Dataset":
        if dataset_name in self._datasets:
            return self._datasets[dataset_name]
        else:
            for dataset in self.data["datasets"]:
                if dataset["name"] == dataset_name:
                    self._datasets[dataset_name] = Dataset(
                        self.api.get(f"release/{self.release_id}/dataset/{dataset_name}"),
                        self,
                    )
                    return self._datasets[dataset_name]

            raise KeyError(f"Dataset {dataset_name} not found in release {self.release_id}")


class Dataset(object):
    def __init__(self, data: DatasetData, release: Release):
        self.data = data
        self.release = release

    @property
    def name(self):
        return self.data["name"]

    @property
    def description(self):
        return self.data["description"]

    @property
    def README(self):
        return self.data["README"]

    @property
    def files(self):
        return self.data["files"]

    def printInfo(self):
        logger.info(f"{self.name.capitalize()} Dataset ({len(self.files)} files):")
        for line in str.splitlines(self.description):
            logger.info("    " + line)

    def downloadFiles(self, output_dir: Path, max_files: int = None, progressbar: bool = True):
        output_dir = Path(output_dir)
        zeros = len(str(len(self.files)))
        for i, file in enumerate(self.files, 1):
            if max_files and i > max_files:
                break
            output_file = output_dir / f"rel-{self.release.release_id}-{self.name}-{i:0{zeros}d}.jsonl.gz"
            if output_file.exists():
                logger.warning(f"File {output_file} already exists. Skipping.")
                continue
            else:
                if progressbar:
                    bar = tqdm(
                        unit="B",
                        unit_scale=True,
                        leave=False,
                        desc=f"Downloading {output_file.name}",
                        ascii=" ▏▎▍▌▋▊▉█",
                    )

                    def report(block_num, block_size, total_size):
                        bar.total = total_size
                        bar.update(block_num * block_size - bar.n)

                else:
                    logger.info(f"Downloading {output_file}")
                    report = None
                try:
                    urllib.request.urlretrieve(file, output_file, reporthook=report)
                except KeyboardInterrupt:
                    logger.warning("Download interrupted. Cleaning up.")
                    output_file.unlink()
                    raise KeyboardInterrupt
                except Exception as e:
                    logger.error(str(e))
                    output_file.unlink()
                    break
                else:
                    logger.info(f"Downloaded {output_file}")
                finally:
                    if progressbar:
                        bar.close()
