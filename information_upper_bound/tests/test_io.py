from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from information_upper_bound.io import iter_jsonl, sha256_file, write_jsonl


class JsonlIoTests(unittest.TestCase):
    def test_gzip_jsonl_is_streamed_and_byte_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.jsonl.gz"
            second = Path(directory) / "second.jsonl.gz"
            rows = ({"id": index, "text": "한글"} for index in range(3))
            write_jsonl(first, rows)
            write_jsonl(second, ({"id": index, "text": "한글"} for index in range(3)))
            self.assertEqual(list(iter_jsonl(first)), list(iter_jsonl(second)))
            self.assertEqual(sha256_file(first), sha256_file(second))


if __name__ == "__main__":
    unittest.main()
