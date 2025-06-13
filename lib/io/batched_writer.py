import os
from io import TextIOBase
from typing import List


class BatchedWriter(TextIOBase):
    def __init__(self, file: os.PathLike, batch_size: int, encoding: str = "utf-8"):
        """
        A file writer that writes to multiple files in batches.

        Args:
            file (os.PathLike): The file path with the "{batch}" placeholder
            batch_size (int): The number of lines to write to each file
        """
        self.file = str(file)
        # Check whether file has the "{batch}" placeholder
        if "{batch}" not in self.file:
            raise ValueError("File must have the '{batch}' placeholder")
        self.batch_size = batch_size

        self.batch_number = 1
        self.current_batch_size = 0
        self._is_closed = False
        self._encoding = encoding
        self.output_file = open(self.file.format(batch=self.batch_number), "w", encoding=encoding, newline="")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def write(self, line: str):
        if self._is_closed:
            raise ValueError("I/O operation on closed file")
        if self.current_batch_size >= self.batch_size:
            self.output_file.close()
            self.batch_number += 1
            self.current_batch_size = 0
            self.output_file = open(
                self.file.format(batch=self.batch_number), "w", encoding=self._encoding, newline=""
            )
        res = self.output_file.write(line)
        self.current_batch_size += 1
        return res

    def writelines(self, lines: List[str]):
        for line in lines:
            self.write(line)

    def flush(self):
        self.output_file.flush()

    def close(self):
        if self._is_closed:
            return
        self._is_closed = True
        self.output_file.close()
