"""Data source implementations."""

from typing import Any, Iterator
import pandas as pd
import yaml
from pathlib import Path

from .base import Source


class CSVSource(Source):
    """Source that loads data from CSV files using pandas."""

    def __init__(self, path: str, column_mapping: dict[str, str] | None = None):
        """Initialize with path to CSV file and optional column mapping.

        Args:
            path: Path to CSV file
            column_mapping: Optional dict mapping CSV column names to output keys.
                           Only mapped columns will be included in output.
                           Example: {"item_id": "id", "content": "instruction"}
        """
        self.path = Path(path)
        self.column_mapping = column_mapping

    def load(self) -> Iterator[dict[str, Any]]:
        """Load CSV data and yield each row as a dictionary."""
        df = pd.read_csv(self.path)
        for _, row in df.iterrows():
            data = row.to_dict()

            # Apply column mapping if provided
            if self.column_mapping is not None:
                mapped_data = {}
                for csv_col, output_key in self.column_mapping.items():
                    if csv_col in data:
                        mapped_data[output_key] = data[csv_col]
                data = mapped_data

            if "id" not in data:
                raise ValueError(f"CSV row missing required 'id' field: {data}")
            yield data


class YAMLSource(Source):
    """Source that loads data from YAML files."""

    def __init__(self, path: str):
        """Initialize with path to YAML file."""
        self.path = Path(path)

    def load(self) -> Iterator[dict[str, Any]]:
        """Load YAML data and yield each item as a dictionary."""
        with open(self.path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        # Handle different YAML structures
        if isinstance(data, list):
            # YAML contains a list of items
            for item in data:
                if not isinstance(item, dict) or "id" not in item:
                    raise ValueError(f"YAML item missing required 'id' field: {item}")
                yield item
        elif isinstance(data, dict):
            # YAML contains a single item or nested structure
            if "id" not in data:
                raise ValueError(f"YAML data missing required 'id' field: {data}")
            yield data
        else:
            # YAML contains primitive data, wrap in dict with generated id
            raise ValueError(
                f"YAML primitive data not supported - must be dict or list of dicts with 'id' fields: {data}"
            )
