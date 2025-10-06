"""Data formatter implementations."""

from typing import Any
import xml.etree.ElementTree as ET
from xml.dom import minidom

from .base import Formatter


class XMLFormatter(Formatter):
    """Formatter that converts data dictionaries to XML strings."""

    def __init__(self, include_keys: list[str] | None = None):
        """Initialize XMLFormatter.

        Args:
            include_keys: Optional list of keys to include in output.
                         If None, all keys are included.
        """
        self.include_keys = include_keys

    def format(self, data: dict[str, Any]) -> str:
        """Format data dictionary as XML string."""
        if self.include_keys:
            filtered_data = {k: v for k, v in data.items() if k in self.include_keys}
        else:
            filtered_data = data
        return self._dict_to_xml(filtered_data)

    def _dict_to_xml(self, data: dict[str, Any], root_name: str = "data") -> str:
        """Convert dictionary to XML string with pretty formatting."""
        root = ET.Element(root_name)
        self._add_dict_to_element(root, data)

        # Pretty print the XML
        rough_string = ET.tostring(root, encoding="unicode")
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ").strip()

    def _add_dict_to_element(self, parent: ET.Element, data: Any) -> None:
        """Recursively add dictionary items to XML element."""
        if isinstance(data, dict):
            for key, value in data.items():
                # Clean key to be valid XML element name
                clean_key = self._clean_xml_key(str(key))
                child = ET.SubElement(parent, clean_key)
                self._add_dict_to_element(child, value)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                child = ET.SubElement(parent, f"item_{i}")
                self._add_dict_to_element(child, item)
        else:
            parent.text = str(data) if data is not None else ""

    def _clean_xml_key(self, key: str) -> str:
        """Clean a string to be a valid XML element name."""
        # Replace invalid characters with underscores
        cleaned = ""
        for char in key:
            if char.isalnum() or char in "_-":
                cleaned += char
            else:
                cleaned += "_"

        # Remove consecutive underscores
        while "__" in cleaned:
            cleaned = cleaned.replace("__", "_")

        # Ensure it starts with a letter or underscore
        if cleaned and not (cleaned[0].isalpha() or cleaned[0] == "_"):
            cleaned = "_" + cleaned

        # Clean up trailing underscores only if there's still content
        cleaned = cleaned.rstrip("_")

        return cleaned or "item"
