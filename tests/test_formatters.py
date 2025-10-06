"""Tests for data formatter implementations."""

import xml.etree.ElementTree as ET

from src.evaluator.formatters import XMLFormatter


class TestXMLFormatter:
    """Tests for XMLFormatter."""

    def test_initialization(self):
        """Test XMLFormatter initialization."""
        formatter = XMLFormatter()
        assert formatter is not None

    def test_format_simple_dict(self, xml_formatter):
        """Test formatting a simple dictionary."""
        data = {"name": "John", "age": 30, "city": "New York"}

        result = xml_formatter.format(data)

        # Parse the XML to verify structure
        root = ET.fromstring(result)
        assert root.tag == "data"

        # Check all elements are present
        children = {child.tag: child.text for child in root}
        assert children["name"] == "John"
        assert children["age"] == "30"
        assert children["city"] == "New York"

    def test_format_nested_dict(self, xml_formatter):
        """Test formatting a nested dictionary."""
        data = {"user": {"name": "Alice", "details": {"age": 25, "location": "Boston"}}}

        result = xml_formatter.format(data)
        root = ET.fromstring(result)

        # Navigate nested structure
        user_elem = root.find("user")
        assert user_elem is not None

        name_elem = user_elem.find("name")
        assert name_elem.text == "Alice"

        details_elem = user_elem.find("details")
        assert details_elem is not None

        age_elem = details_elem.find("age")
        assert age_elem.text == "25"

    def test_format_with_list(self, xml_formatter):
        """Test formatting dictionary with list values."""
        data = {"items": ["apple", "banana", "cherry"], "numbers": [1, 2, 3]}

        result = xml_formatter.format(data)
        root = ET.fromstring(result)

        # Check items list
        items_elem = root.find("items")
        assert items_elem is not None

        item_children = (
            items_elem.findall("item_0")
            + items_elem.findall("item_1")
            + items_elem.findall("item_2")
        )
        assert len(item_children) == 3

        # Check content
        assert items_elem.find("item_0").text == "apple"
        assert items_elem.find("item_1").text == "banana"
        assert items_elem.find("item_2").text == "cherry"

    def test_format_with_mixed_types(self, xml_formatter):
        """Test formatting with mixed data types."""
        data = {
            "string": "hello",
            "integer": 42,
            "float": 3.14159,
            "boolean": True,
            "none_value": None,
            "empty_string": "",
        }

        result = xml_formatter.format(data)
        root = ET.fromstring(result)

        children = {child.tag: child.text or "" for child in root}

        assert children["string"] == "hello"
        assert children["integer"] == "42"
        assert children["float"] == "3.14159"
        assert children["boolean"] == "True"
        assert children["none_value"] == ""  # None becomes empty string
        assert children["empty_string"] == ""

    def test_format_with_special_characters(self, xml_formatter):
        """Test formatting with special characters that need XML escaping."""
        data = {
            "with_ampersand": "Tom & Jerry",
            "with_quotes": 'He said "Hello"',
            "with_brackets": "<tag>content</tag>",
            "unicode": "Hello 世界 🌍",
        }

        result = xml_formatter.format(data)
        root = ET.fromstring(result)

        # XML parser should handle escaping automatically
        children = {child.tag: child.text for child in root}

        assert children["with_ampersand"] == "Tom & Jerry"
        assert children["with_quotes"] == 'He said "Hello"'
        assert children["with_brackets"] == "<tag>content</tag>"
        assert children["unicode"] == "Hello 世界 🌍"

    def test_clean_xml_key_valid_keys(self, xml_formatter):
        """Test XML key cleaning with valid keys."""
        assert xml_formatter._clean_xml_key("valid_key") == "valid_key"
        assert xml_formatter._clean_xml_key("key123") == "key123"
        assert xml_formatter._clean_xml_key("_private") == "_private"
        assert xml_formatter._clean_xml_key("key-with-dashes") == "key-with-dashes"

    def test_clean_xml_key_invalid_keys(self, xml_formatter):
        """Test XML key cleaning with invalid characters."""
        assert xml_formatter._clean_xml_key("key with spaces") == "key_with_spaces"
        assert xml_formatter._clean_xml_key("key@symbol") == "key_symbol"
        assert xml_formatter._clean_xml_key("key.dot") == "key_dot"
        assert xml_formatter._clean_xml_key("key/slash") == "key_slash"
        assert xml_formatter._clean_xml_key("key:colon") == "key_colon"

    def test_clean_xml_key_starts_with_number(self, xml_formatter):
        """Test XML key cleaning when key starts with number."""
        assert xml_formatter._clean_xml_key("123key") == "_123key"
        assert xml_formatter._clean_xml_key("9invalid") == "_9invalid"

    def test_clean_xml_key_empty_or_invalid(self, xml_formatter):
        """Test XML key cleaning with edge cases."""
        assert xml_formatter._clean_xml_key("") == "item"
        assert (
            xml_formatter._clean_xml_key("@#$%") == "item"
        )  # All invalid chars become underscores, then cleaned to "item"

    def test_format_complex_structure(self, xml_formatter):
        """Test formatting a complex nested structure."""
        data = {
            "evaluation": {
                "id": "test-001",
                "input": {
                    "text": "Sample input text",
                    "metadata": {"source": "test", "timestamp": "2024-01-01T00:00:00Z"},
                },
                "judges": [
                    {
                        "name": "quality_judge",
                        "score": 0.85,
                        "annotation": "Good quality",
                    },
                    {"name": "safety_judge", "score": 0.95, "annotation": "Very safe"},
                ],
                "summary": {"total_judges": 2, "avg_score": 0.9},
            }
        }

        result = xml_formatter.format(data)
        root = ET.fromstring(result)

        # Verify structure exists
        eval_elem = root.find("evaluation")
        assert eval_elem is not None
        assert eval_elem.find("id").text == "test-001"

        # Check nested input
        input_elem = eval_elem.find("input")
        assert input_elem.find("text").text == "Sample input text"

        metadata_elem = input_elem.find("metadata")
        assert metadata_elem.find("source").text == "test"

        # Check judges list
        judges_elem = eval_elem.find("judges")
        assert judges_elem is not None

        judge_0 = judges_elem.find("item_0")
        assert judge_0.find("name").text == "quality_judge"
        assert judge_0.find("score").text == "0.85"

        # Check summary
        summary_elem = eval_elem.find("summary")
        assert summary_elem.find("total_judges").text == "2"

    def test_format_empty_dict(self, xml_formatter):
        """Test formatting an empty dictionary."""
        data = {}

        result = xml_formatter.format(data)
        root = ET.fromstring(result)

        assert root.tag == "data"
        assert len(list(root)) == 0  # No children

    def test_format_single_level_with_various_keys(self, xml_formatter):
        """Test formatting with various key types that need cleaning."""
        data = {
            "normal_key": "value1",
            "key with spaces": "value2",
            "123_starts_with_number": "value3",
            "key@with#symbols": "value4",
            "_underscore_key": "value5",
        }

        result = xml_formatter.format(data)
        root = ET.fromstring(result)

        # Check that all keys are present and properly cleaned
        children = {child.tag: child.text for child in root}

        assert "normal_key" in children
        assert "key_with_spaces" in children
        assert "_123_starts_with_number" in children
        assert "key_with_symbols" in children  # @ and # become _
        assert "_underscore_key" in children

        assert children["normal_key"] == "value1"
        assert children["key_with_spaces"] == "value2"

    def test_format_preserves_xml_formatting(self, xml_formatter):
        """Test that the formatted XML is properly indented."""
        data = {"parent": {"child": "value"}}

        result = xml_formatter.format(data)

        # Check that result contains proper indentation
        assert "  " in result  # Should have indentation
        assert result.startswith("<?xml")  # Should have XML declaration

        # Should be parseable
        root = ET.fromstring(result)
        assert root.tag == "data"
