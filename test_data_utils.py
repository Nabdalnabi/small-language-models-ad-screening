"""Synthetic unit tests; contains no research transcripts or identifiers."""
import tempfile
import unittest
from pathlib import Path
from data_utils import check_splits, encode_labels, label_mapping, read_rows


class DataTests(unittest.TestCase):
    def test_mapping_is_shared(self):
        mapping = label_mapping([{"label": "control"}, {"label": "AD"}])
        self.assertEqual(encode_labels([{"label": "control"}], mapping), [mapping["control"]])
        with self.assertRaises(ValueError):
            encode_labels([{"label": "unknown"}], mapping)

    def test_cross_split_duplicates(self):
        with self.assertRaises(ValueError):
            check_splits({"train": [{"text": "Synthetic TEST"}],
                          "val": [{"text": " synthetic  test "}]})

    def test_participant_overlap(self):
        with self.assertRaises(ValueError):
            check_splits({"train": [{"text": "one", "group": "synthetic-1"}],
                          "val": [{"text": "two", "group": "synthetic-1"}]}, grouped=True)

    def test_csv_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.csv"
            path.write_text("input,output\n,control\n")
            with self.assertRaises(ValueError):
                read_rows(path)
            path.write_text("input,output\nsynthetic placeholder,control\n")
            self.assertEqual(len(read_rows(path)), 1)


if __name__ == "__main__":
    unittest.main()
