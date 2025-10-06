"""Tests for DuckDB storage implementation."""

import pytest
from pathlib import Path

from src.evaluator.storage import ResultsStorage


class TestResultsStorage:
    """Tests for ResultsStorage."""

    def test_initialization(self, temp_db_path):
        """Test storage initialization."""
        storage = ResultsStorage(temp_db_path, "test_table")

        assert storage.db_path == Path(temp_db_path)
        assert storage.table_name == "test_table"
        assert storage.conn is None

    def test_context_manager(self, temp_db_path):
        """Test storage as context manager."""
        storage = ResultsStorage(temp_db_path, "test_table")

        assert storage.conn is None

        with storage as s:
            assert s.conn is not None
            assert s is storage

        assert storage.conn is None

    def test_manual_connect_close(self, temp_db_path):
        """Test manual connect and close."""
        storage = ResultsStorage(temp_db_path, "test_table")

        storage.connect()
        assert storage.conn is not None

        storage.close()
        assert storage.conn is None

    def test_create_table_basic(self, temp_db_path):
        """Test creating a basic table."""
        columns = {
            "id": "VARCHAR PRIMARY KEY",
            "score": "DOUBLE",
            "annotation": "VARCHAR",
        }

        with ResultsStorage(temp_db_path, "test_table") as storage:
            storage.create_table(columns)

            # Verify table exists and has correct structure
            result = storage.conn.execute("PRAGMA table_info(test_table)").fetchall()

            column_info = {row[1]: row[2] for row in result}  # name: type
            assert "id" in column_info
            assert "score" in column_info
            assert "annotation" in column_info

    def test_create_table_auto_add_id(self, temp_db_path):
        """Test that id column is automatically added if missing."""
        columns = {"score": "DOUBLE", "text": "VARCHAR"}

        with ResultsStorage(temp_db_path, "test_table") as storage:
            storage.create_table(columns)

            result = storage.conn.execute("PRAGMA table_info(test_table)").fetchall()
            column_names = [row[1] for row in result]

            assert "id" in column_names
            assert "score" in column_names
            assert "text" in column_names

    def test_create_table_if_not_exists(self, temp_db_path):
        """Test that creating table multiple times doesn't error."""
        columns = {"id": "VARCHAR PRIMARY KEY", "value": "INTEGER"}

        with ResultsStorage(temp_db_path, "test_table") as storage:
            storage.create_table(columns)
            storage.create_table(columns)  # Should not error

            # Table should still exist
            assert (
                storage.conn.execute("SELECT COUNT(*) FROM test_table").fetchone()[0]
                == 0
            )

    def test_insert_single_result(self, temp_db_path):
        """Test inserting a single result."""
        columns = {
            "id": "VARCHAR PRIMARY KEY",
            "judge_test.score": "DOUBLE",
            "judge_test.annotation": "VARCHAR",
        }

        result_data = {
            "id": "test-001",
            "judge_test.score": 0.85,
            "judge_test.annotation": "Good result",
        }

        with ResultsStorage(temp_db_path, "test_table") as storage:
            storage.create_table(columns)
            storage.insert_result(result_data)

            # Verify data was inserted
            rows = storage.conn.execute("SELECT * FROM test_table").fetchall()
            assert len(rows) == 1

            row = rows[0]
            assert row[0] == "test-001"  # id
            assert row[1] == 0.85  # score
            assert row[2] == "Good result"  # annotation

    def test_insert_multiple_results(self, temp_db_path):
        """Test inserting multiple results."""
        columns = {"id": "VARCHAR PRIMARY KEY", "score": "DOUBLE"}

        with ResultsStorage(temp_db_path, "test_table") as storage:
            storage.create_table(columns)

            # Insert multiple results
            for i in range(3):
                result = {"id": f"test-{i:03d}", "score": 0.5 + (i * 0.2)}
                storage.insert_result(result)

            # Verify all results were inserted
            count = storage.conn.execute("SELECT COUNT(*) FROM test_table").fetchone()[
                0
            ]
            assert count == 3

    def test_result_exists(self, temp_db_path):
        """Test checking if result exists."""
        columns = {"id": "VARCHAR PRIMARY KEY", "value": "INTEGER"}

        with ResultsStorage(temp_db_path, "test_table") as storage:
            storage.create_table(columns)

            # Initially should not exist
            assert not storage.result_exists("test-001")

            # Insert a result
            storage.insert_result({"id": "test-001", "value": 42})

            # Now should exist
            assert storage.result_exists("test-001")
            assert not storage.result_exists("test-002")

    def test_get_all_results(self, temp_db_path):
        """Test getting all results."""
        columns = {"id": "VARCHAR PRIMARY KEY", "score": "DOUBLE", "text": "VARCHAR"}

        test_results = [
            {"id": "test-001", "score": 0.8, "text": "First"},
            {"id": "test-002", "score": 0.6, "text": "Second"},
            {"id": "test-003", "score": 0.9, "text": "Third"},
        ]

        with ResultsStorage(temp_db_path, "test_table") as storage:
            storage.create_table(columns)

            # Insert test data
            for result in test_results:
                storage.insert_result(result)

            # Get all results
            all_results = storage.get_all_results()

            assert len(all_results) == 3

            # Results should be dictionaries
            for result in all_results:
                assert isinstance(result, dict)
                assert "id" in result
                assert "score" in result
                assert "text" in result

            # Check specific values (order may vary)
            ids = {r["id"] for r in all_results}
            assert ids == {"test-001", "test-002", "test-003"}

    def test_get_stats_empty_table(self, temp_db_path):
        """Test getting stats from empty table."""
        with ResultsStorage(temp_db_path, "test_table") as storage:
            # Table doesn't exist yet
            stats = storage.get_stats()
            assert stats["total"] == 0

    def test_get_stats_with_data(self, temp_db_path):
        """Test getting stats from table with data."""
        columns = {
            "id": "VARCHAR PRIMARY KEY",
            "judge_quality.score": "DOUBLE",
            "judge_quality.confidence": "DOUBLE",
            "judge_safety.score": "DOUBLE",
            "judge_quality.annotation": "VARCHAR",  # Non-numeric column
        }

        test_data = [
            {
                "id": "test-001",
                "judge_quality.score": 0.8,
                "judge_quality.confidence": 0.9,
                "judge_safety.score": 0.95,
                "judge_quality.annotation": "Good",
            },
            {
                "id": "test-002",
                "judge_quality.score": 0.6,
                "judge_quality.confidence": 0.7,
                "judge_safety.score": 0.85,
                "judge_quality.annotation": "OK",
            },
            {
                "id": "test-003",
                "judge_quality.score": 0.9,
                "judge_quality.confidence": 0.95,
                "judge_safety.score": 0.9,
                "judge_quality.annotation": "Excellent",
            },
        ]

        with ResultsStorage(temp_db_path, "test_table") as storage:
            storage.create_table(columns)

            for result in test_data:
                storage.insert_result(result)

            stats = storage.get_stats()

            # Check total count
            assert stats["total"] == 3

            # Check numeric column averages
            assert "judge_quality.score_mean" in stats
            assert "judge_quality.confidence_mean" in stats
            assert "judge_safety.score_mean" in stats

            # Verify average calculations
            expected_quality_score_avg = (0.8 + 0.6 + 0.9) / 3
            assert (
                abs(stats["judge_quality.score_mean"] - expected_quality_score_avg)
                < 0.001
            )

            # Non-numeric columns shouldn't have averages
            assert "judge_quality.annotation_mean" not in stats

    def test_get_stats_with_null_values(self, temp_db_path):
        """Test getting stats with null values."""
        columns = {
            "id": "VARCHAR PRIMARY KEY",
            "judge_test.score": "DOUBLE",  # Use judge_ prefix to match stats logic
        }

        with ResultsStorage(temp_db_path, "test_table") as storage:
            storage.create_table(columns)

            # Insert data with some null values
            storage.insert_result({"id": "test-001", "judge_test.score": 0.8})
            storage.insert_result({"id": "test-002"})  # score will be null
            storage.insert_result({"id": "test-003", "judge_test.score": 0.6})

            stats = storage.get_stats()

            assert stats["total"] == 3
            # Average should only consider non-null values
            expected_key = None
            for key in stats.keys():
                if "judge_test.score" in key and "_mean" in key:
                    expected_key = key
                    break

            assert expected_key is not None
            assert abs(stats[expected_key] - 0.7) < 0.001  # (0.8 + 0.6) / 2

    def test_table_exists(self, temp_db_path):
        """Test checking if table exists."""
        with ResultsStorage(temp_db_path, "test_table") as storage:
            # Initially table doesn't exist
            assert not storage.table_exists()

            # Create table
            storage.create_table({"id": "VARCHAR PRIMARY KEY"})

            # Now table should exist
            assert storage.table_exists()

    def test_connection_not_established_errors(self, temp_db_path):
        """Test that methods error when connection not established."""
        storage = ResultsStorage(temp_db_path, "test_table")

        with pytest.raises(RuntimeError, match="Database connection not established"):
            storage.create_table({"id": "VARCHAR"})

        with pytest.raises(RuntimeError, match="Database connection not established"):
            storage.insert_result({"id": "test"})

        with pytest.raises(RuntimeError, match="Database connection not established"):
            storage.result_exists("test")

        with pytest.raises(RuntimeError, match="Database connection not established"):
            storage.get_all_results()

        with pytest.raises(RuntimeError, match="Database connection not established"):
            storage.get_stats()

        with pytest.raises(RuntimeError, match="Database connection not established"):
            storage.table_exists()

    def test_database_persistence(self, temp_db_path):
        """Test that data persists across connections."""
        columns = {"id": "VARCHAR PRIMARY KEY", "value": "INTEGER"}

        # First connection - insert data
        with ResultsStorage(temp_db_path, "test_table") as storage1:
            storage1.create_table(columns)
            storage1.insert_result({"id": "persistent", "value": 42})

        # Second connection - verify data exists
        with ResultsStorage(temp_db_path, "test_table") as storage2:
            assert storage2.result_exists("persistent")
            results = storage2.get_all_results()
            assert len(results) == 1
            assert results[0]["id"] == "persistent"
            assert results[0]["value"] == 42

    def test_concurrent_operations(self, temp_db_path):
        """Test that multiple operations work correctly."""
        columns = {
            "id": "VARCHAR PRIMARY KEY",
            "judge_a.score": "DOUBLE",
            "judge_b.score": "DOUBLE",
        }

        with ResultsStorage(temp_db_path, "test_evaluations") as storage:
            storage.create_table(columns)

            # Simulate evaluation results
            results = []
            for i in range(5):
                result = {
                    "id": f"eval-{i:03d}",
                    "judge_a.score": 0.5 + (i * 0.1),
                    "judge_b.score": 0.8 - (i * 0.05),
                }
                storage.insert_result(result)
                results.append(result)

            # Verify all data
            all_results = storage.get_all_results()
            assert len(all_results) == 5

            # Check stats
            stats = storage.get_stats()
            assert stats["total"] == 5
            assert "judge_a.score_mean" in stats
            assert "judge_b.score_mean" in stats
