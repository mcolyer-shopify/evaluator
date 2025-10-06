"""Tests for data source implementations."""

import pytest
import pandas as pd
import yaml
from pathlib import Path

from src.evaluator.sources import CSVSource, YAMLSource


class TestCSVSource:
    """Tests for CSVSource."""

    def test_initialization(self, sample_csv_file):
        """Test CSVSource initialization."""
        source = CSVSource(sample_csv_file)
        assert source.path == Path(sample_csv_file)

    def test_load_csv_data(self, csv_source, sample_data):
        """Test loading data from CSV file."""
        loaded_data = list(csv_source.load())

        assert len(loaded_data) == 3

        # Check that data matches (pandas may convert types)
        for i, item in enumerate(loaded_data):
            assert item["id"] == sample_data[i]["id"]
            assert item["text"] == sample_data[i]["text"]
            assert item["category"] == sample_data[i]["category"]

    def test_load_empty_csv(self, tmp_path):
        """Test loading from empty CSV file."""
        empty_csv = tmp_path / "empty.csv"
        # Create empty CSV with headers only
        pd.DataFrame(columns=["id", "text"]).to_csv(empty_csv, index=False)

        source = CSVSource(str(empty_csv))
        loaded_data = list(source.load())

        assert len(loaded_data) == 0

    def test_load_csv_with_mixed_types(self, tmp_path):
        """Test loading CSV with mixed data types."""
        mixed_data = [
            {"id": 1, "score": 0.85, "valid": True, "note": "good"},
            {"id": 2, "score": 0.72, "valid": False, "note": None},
        ]

        csv_file = tmp_path / "mixed.csv"
        pd.DataFrame(mixed_data).to_csv(csv_file, index=False)

        source = CSVSource(str(csv_file))
        loaded_data = list(source.load())

        assert len(loaded_data) == 2
        assert loaded_data[0]["id"] == 1
        assert loaded_data[0]["score"] == 0.85
        assert loaded_data[0]["valid"]
        assert not loaded_data[1]["valid"]

    def test_load_csv_with_special_characters(self, tmp_path):
        """Test loading CSV with special characters and unicode."""
        special_data = [
            {
                "id": "special-1",
                "text": "Hello, world!",
                "emoji": "🎉",
                "quote": 'He said "hi"',
            },
            {
                "id": "special-2",
                "text": "Line\nbreak",
                "emoji": "🚀",
                "quote": "Simple text",
            },
        ]

        csv_file = tmp_path / "special.csv"
        pd.DataFrame(special_data).to_csv(csv_file, index=False)

        source = CSVSource(str(csv_file))
        loaded_data = list(source.load())

        assert len(loaded_data) == 2
        assert loaded_data[0]["emoji"] == "🎉"
        assert loaded_data[0]["quote"] == 'He said "hi"'
        assert "Line\nbreak" in loaded_data[1]["text"]

    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file raises error."""
        source = CSVSource("nonexistent.csv")

        with pytest.raises(FileNotFoundError):
            list(source.load())

    def test_load_csv_missing_id_field(self, tmp_path):
        """Test loading CSV without id field raises error."""
        data_without_id = [
            {"text": "No ID field", "value": 1},
            {"text": "Another without ID", "value": 2},
        ]

        csv_file = tmp_path / "no_id.csv"
        pd.DataFrame(data_without_id).to_csv(csv_file, index=False)

        source = CSVSource(str(csv_file))
        with pytest.raises(ValueError, match="CSV row missing required 'id' field"):
            list(source.load())

    def test_column_mapping_basic(self, tmp_path):
        """Test basic column mapping functionality."""
        data = [
            {
                "item_id": "1",
                "content": "Hello world",
                "author": "Alice",
                "extra": "ignore",
            },
            {
                "item_id": "2",
                "content": "Goodbye",
                "author": "Bob",
                "extra": "also ignore",
            },
        ]

        csv_file = tmp_path / "mapping_test.csv"
        pd.DataFrame(data).to_csv(csv_file, index=False)

        column_mapping = {
            "item_id": "id",
            "content": "instruction",
            "author": "metadata_author",
        }

        source = CSVSource(str(csv_file), column_mapping=column_mapping)
        loaded_data = list(source.load())

        assert len(loaded_data) == 2

        # Check first row
        assert loaded_data[0]["id"] == 1
        assert loaded_data[0]["instruction"] == "Hello world"
        assert loaded_data[0]["metadata_author"] == "Alice"
        assert "extra" not in loaded_data[0]  # Should be filtered out
        assert len(loaded_data[0]) == 3  # Only mapped columns

        # Check second row
        assert loaded_data[1]["id"] == 2
        assert loaded_data[1]["instruction"] == "Goodbye"
        assert loaded_data[1]["metadata_author"] == "Bob"

    def test_column_mapping_missing_columns(self, tmp_path):
        """Test column mapping with missing CSV columns."""
        data = [
            {"item_id": "1", "content": "Hello world"},
            {"item_id": "2", "content": "Goodbye"},
        ]

        csv_file = tmp_path / "mapping_missing.csv"
        pd.DataFrame(data).to_csv(csv_file, index=False)

        column_mapping = {
            "item_id": "id",
            "content": "instruction",
            "nonexistent": "missing_field",  # This column doesn't exist in CSV
        }

        source = CSVSource(str(csv_file), column_mapping=column_mapping)
        loaded_data = list(source.load())

        assert len(loaded_data) == 2

        # Should only include columns that exist in CSV
        assert loaded_data[0]["id"] == 1
        assert loaded_data[0]["instruction"] == "Hello world"
        assert "missing_field" not in loaded_data[0]
        assert len(loaded_data[0]) == 2

    def test_column_mapping_without_id_raises_error(self, tmp_path):
        """Test that column mapping without mapping to 'id' raises error."""
        data = [
            {"item_id": "1", "content": "Hello world"},
            {"item_id": "2", "content": "Goodbye"},
        ]

        csv_file = tmp_path / "mapping_no_id.csv"
        pd.DataFrame(data).to_csv(csv_file, index=False)

        column_mapping = {
            "content": "instruction",
            # Missing mapping for item_id -> id
        }

        source = CSVSource(str(csv_file), column_mapping=column_mapping)
        with pytest.raises(ValueError, match="CSV row missing required 'id' field"):
            list(source.load())

    def test_column_mapping_none_preserves_existing_behavior(
        self, csv_source, sample_data
    ):
        """Test that column_mapping=None preserves existing behavior."""
        # This test reuses existing fixtures to ensure backward compatibility
        loaded_data = list(csv_source.load())

        assert len(loaded_data) == 3
        for i, item in enumerate(loaded_data):
            assert item["id"] == sample_data[i]["id"]
            assert item["text"] == sample_data[i]["text"]
            assert item["category"] == sample_data[i]["category"]

    def test_column_mapping_empty_dict(self, tmp_path):
        """Test column mapping with empty dict."""
        data = [
            {"id": "1", "content": "Hello world", "author": "Alice"},
            {"id": "2", "content": "Goodbye", "author": "Bob"},
        ]

        csv_file = tmp_path / "mapping_empty.csv"
        pd.DataFrame(data).to_csv(csv_file, index=False)

        source = CSVSource(str(csv_file), column_mapping={})

        # With empty mapping, no columns should be included, so id will be missing
        with pytest.raises(ValueError, match="CSV row missing required 'id' field"):
            list(source.load())

    def test_column_mapping_with_id_column_direct(self, tmp_path):
        """Test column mapping where 'id' column exists directly in CSV."""
        data = [
            {"id": "1", "content": "Hello world", "author": "Alice"},
            {"id": "2", "content": "Goodbye", "author": "Bob"},
        ]

        csv_file = tmp_path / "mapping_direct_id.csv"
        pd.DataFrame(data).to_csv(csv_file, index=False)

        column_mapping = {
            "id": "id",  # Map id to id (explicit)
            "content": "instruction",
        }

        source = CSVSource(str(csv_file), column_mapping=column_mapping)
        loaded_data = list(source.load())

        assert len(loaded_data) == 2
        assert loaded_data[0]["id"] == 1
        assert loaded_data[0]["instruction"] == "Hello world"
        assert "author" not in loaded_data[0]  # Should be filtered out


class TestYAMLSource:
    """Tests for YAMLSource."""

    def test_initialization(self, sample_yaml_file):
        """Test YAMLSource initialization."""
        source = YAMLSource(sample_yaml_file)
        assert source.path == Path(sample_yaml_file)

    def test_load_yaml_list(self, yaml_source, sample_data):
        """Test loading list data from YAML file."""
        loaded_data = list(yaml_source.load())

        assert len(loaded_data) == 3
        assert loaded_data == sample_data

    def test_load_yaml_single_dict(self, tmp_path):
        """Test loading single dictionary from YAML file."""
        single_item = {"id": "single-1", "name": "test", "value": 42, "active": True}

        yaml_file = tmp_path / "single.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(single_item, f)

        source = YAMLSource(str(yaml_file))
        loaded_data = list(source.load())

        assert len(loaded_data) == 1
        assert loaded_data[0] == single_item

    def test_load_yaml_primitive_data(self, tmp_path):
        """Test that primitive data raises error (no longer supported)."""
        primitive_data = "just a string"

        yaml_file = tmp_path / "primitive.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(primitive_data, f)

        source = YAMLSource(str(yaml_file))

        with pytest.raises(ValueError, match="YAML primitive data not supported"):
            list(source.load())

    def test_load_yaml_nested_structure(self, tmp_path):
        """Test loading nested YAML structure."""
        nested_data = {
            "id": "nested-1",
            "config": {"model": "gpt-4", "temperature": 0.7},
            "prompts": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Hello"},
            ],
            "metadata": {"version": "1.0", "tags": ["test", "example"]},
        }

        yaml_file = tmp_path / "nested.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(nested_data, f)

        source = YAMLSource(str(yaml_file))
        loaded_data = list(source.load())

        assert len(loaded_data) == 1
        assert loaded_data[0] == nested_data
        assert loaded_data[0]["config"]["model"] == "gpt-4"
        assert len(loaded_data[0]["prompts"]) == 2

    def test_load_yaml_with_unicode(self, tmp_path):
        """Test loading YAML with unicode characters."""
        unicode_data = [
            {"id": "unicode-1", "text": "Hello 世界", "emoji": "🌍"},
            {"id": "unicode-2", "text": "Café", "special": "ñoño"},
        ]

        yaml_file = tmp_path / "unicode.yaml"
        with open(yaml_file, "w", encoding="utf-8") as f:
            yaml.dump(unicode_data, f, allow_unicode=True)

        source = YAMLSource(str(yaml_file))
        loaded_data = list(source.load())

        assert len(loaded_data) == 2
        assert loaded_data[0]["text"] == "Hello 世界"
        assert loaded_data[0]["emoji"] == "🌍"
        assert loaded_data[1]["special"] == "ñoño"

    def test_load_empty_yaml_list(self, tmp_path):
        """Test loading empty YAML list."""
        yaml_file = tmp_path / "empty_list.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump([], f)

        source = YAMLSource(str(yaml_file))
        loaded_data = list(source.load())

        assert len(loaded_data) == 0

    def test_load_yaml_with_null_values(self, tmp_path):
        """Test loading YAML with null values."""
        data_with_nulls = [
            {"id": 1, "value": None, "text": "valid"},
            {"id": 2, "value": "something", "text": None},
        ]

        yaml_file = tmp_path / "nulls.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(data_with_nulls, f)

        source = YAMLSource(str(yaml_file))
        loaded_data = list(source.load())

        assert len(loaded_data) == 2
        assert loaded_data[0]["value"] is None
        assert loaded_data[1]["text"] is None

    def test_load_nonexistent_yaml_file(self):
        """Test loading from nonexistent YAML file raises error."""
        source = YAMLSource("nonexistent.yaml")

        with pytest.raises(FileNotFoundError):
            list(source.load())

    def test_load_invalid_yaml(self, tmp_path):
        """Test loading invalid YAML raises error."""
        yaml_file = tmp_path / "invalid.yaml"
        with open(yaml_file, "w") as f:
            f.write("{ invalid: yaml: content")

        source = YAMLSource(str(yaml_file))

        with pytest.raises(yaml.YAMLError):
            list(source.load())

    def test_load_yaml_missing_id_field(self, tmp_path):
        """Test loading YAML without id field raises error."""
        data_without_id = [
            {"text": "No ID field", "value": 1},
            {"text": "Another without ID", "value": 2},
        ]

        yaml_file = tmp_path / "no_id.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(data_without_id, f)

        source = YAMLSource(str(yaml_file))
        with pytest.raises(ValueError, match="YAML item missing required 'id' field"):
            list(source.load())


class TestSourceComparison:
    """Test comparing data from different sources."""

    def test_csv_yaml_equivalence(self, tmp_path):
        """Test that same data loads equivalently from CSV and YAML."""
        test_data = [
            {"id": "eq-1", "name": "Alice", "score": 95.5},
            {"id": "eq-2", "name": "Bob", "score": 87.2},
        ]

        # Create CSV file
        csv_file = tmp_path / "test.csv"
        pd.DataFrame(test_data).to_csv(csv_file, index=False)

        # Create YAML file
        yaml_file = tmp_path / "test.yaml"
        with open(yaml_file, "w") as f:
            yaml.dump(test_data, f)

        csv_source = CSVSource(str(csv_file))
        yaml_source = YAMLSource(str(yaml_file))

        csv_data = list(csv_source.load())
        yaml_data = list(yaml_source.load())

        assert len(csv_data) == len(yaml_data) == 2

        # Compare content (accounting for type differences)
        for i in range(len(csv_data)):
            assert csv_data[i]["id"] == yaml_data[i]["id"]
            assert csv_data[i]["name"] == yaml_data[i]["name"]
            assert abs(csv_data[i]["score"] - yaml_data[i]["score"]) < 0.001
