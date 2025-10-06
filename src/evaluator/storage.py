"""DuckDB storage integration for evaluation results."""

from typing import Any, Optional
import duckdb
from pathlib import Path


class ResultsStorage:
    """DuckDB-based storage for evaluation results."""

    def __init__(self, db_path: str, table_name: str):
        """Initialize storage with database path and table name."""
        self.db_path = Path(db_path)
        self.table_name = table_name
        self.conn: Optional[duckdb.DuckDBPyConnection] = None

    def connect(self) -> None:
        """Connect to the DuckDB database."""
        self.conn = duckdb.connect(str(self.db_path))

    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def create_table(self, columns: dict[str, str]) -> None:
        """Create results table with dynamic schema.

        Args:
            columns: Dictionary mapping column names to SQL types
        """
        if not self.conn:
            raise RuntimeError("Database connection not established")

        # Always include id column
        if "id" not in columns:
            columns = {"id": "VARCHAR PRIMARY KEY", **columns}

        column_defs = [f'"{name}" {sql_type}' for name, sql_type in columns.items()]
        create_sql = (
            f"CREATE TABLE IF NOT EXISTS {self.table_name} ({', '.join(column_defs)})"
        )

        self.conn.execute(create_sql)

    def insert_result(self, result: dict[str, Any]) -> None:
        """Insert a single evaluation result, skipping if already exists."""
        if not self.conn:
            raise RuntimeError("Database connection not established")

        # Use a transaction to atomically check existence and insert
        self.conn.begin()
        try:
            # Check if result already exists within the transaction
            result_id = result.get("id")
            if not result_id:
                raise ValueError("Result missing required 'id' field")

            exists = (
                self.conn.execute(
                    f'SELECT COUNT(*) FROM {self.table_name} WHERE "id" = ?',
                    [result_id],
                ).fetchone()[0]
                > 0
            )

            if exists:
                # Result already exists, skip insertion
                self.conn.rollback()
                return

            # Prepare insert statement
            columns = list(result.keys())
            quoted_columns = [f'"{col}"' for col in columns]
            placeholders = ["?" for _ in columns]
            values = list(result.values())

            insert_sql = f"INSERT INTO {self.table_name} ({', '.join(quoted_columns)}) VALUES ({', '.join(placeholders)})"
            self.conn.execute(insert_sql, values)
            self.conn.commit()

        except Exception:
            self.conn.rollback()
            raise

    def result_exists(self, result_id: str) -> bool:
        """Check if a result with given ID already exists."""
        if not self.conn:
            raise RuntimeError("Database connection not established")

        result = self.conn.execute(
            f'SELECT COUNT(*) FROM {self.table_name} WHERE "id" = ?', [result_id]
        ).fetchone()

        return result[0] > 0 if result else False

    def get_all_results(self) -> list[dict[str, Any]]:
        """Get all results from the table."""
        if not self.conn:
            raise RuntimeError("Database connection not established")

        results = self.conn.execute(f"SELECT * FROM {self.table_name}").fetchall()

        # Get column names
        columns = [desc[0] for desc in self.conn.description]

        # Convert to list of dictionaries
        return [dict(zip(columns, row)) for row in results]

    def get_stats(self) -> dict[str, Any]:
        """Get summary statistics from the results table."""
        if not self.conn:
            raise RuntimeError("Database connection not established")

        # Check if table exists
        table_exists = (
            self.conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
                [self.table_name],
            ).fetchone()[0]
            > 0
        )

        if not table_exists:
            return {"total": 0}

        # Get total count
        total_count = self.conn.execute(
            f"SELECT COUNT(*) FROM {self.table_name}"
        ).fetchone()[0]

        stats = {"total": total_count}

        # Count errors across all judges
        error_count = 0
        error_columns_info = self.conn.execute(f"""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = '{self.table_name}' 
            AND column_name LIKE 'judge_%.error'
        """).fetchall()

        for col_info in error_columns_info:
            error_column = col_info[0]
            try:
                errors_in_column = self.conn.execute(f"""
                    SELECT COUNT(*) 
                    FROM {self.table_name} 
                    WHERE "{error_column}" IS NOT NULL AND "{error_column}" != ''
                """).fetchone()[0]
                error_count += errors_in_column
            except Exception:
                continue

        stats["total_errors"] = error_count

        # Get column information
        columns_info = self.conn.execute(
            f"PRAGMA table_info({self.table_name})"
        ).fetchall()

        # Calculate stats for numeric columns (judge metrics)
        latency_columns = []
        token_total_columns = []
        token_prompt_columns = []
        token_completion_columns = []
        token_cached_columns = []

        for col_info in columns_info:
            column_name = col_info[1]  # Column name is at index 1
            column_type = col_info[2]  # Column type is at index 2

            # Skip id column and focus on judge metrics
            if column_name == "id" or not column_name.startswith("judge_"):
                continue

            # Only calculate stats for numeric columns
            if "DOUBLE" in column_type.upper() or "FLOAT" in column_type.upper():
                try:
                    mean_result = self.conn.execute(
                        f'SELECT AVG("{column_name}") FROM {self.table_name} WHERE "{column_name}" IS NOT NULL'
                    ).fetchone()

                    if mean_result and mean_result[0] is not None:
                        stats[f"{column_name}_mean"] = mean_result[0]

                    # Collect columns for aggregated performance metrics
                    if column_name.endswith(".meta.latency_ms"):
                        latency_columns.append(column_name)
                    elif column_name.endswith(".meta.tokens_total"):
                        token_total_columns.append(column_name)
                    elif column_name.endswith(".meta.tokens_input"):
                        token_prompt_columns.append(column_name)
                    elif column_name.endswith(".meta.tokens_output"):
                        token_completion_columns.append(column_name)
                    elif column_name.endswith(".meta.tokens_cached"):
                        token_cached_columns.append(column_name)

                except Exception:
                    # Skip columns that can't be averaged
                    continue

        # Calculate aggregated performance statistics
        if total_count > 0:
            # Latency percentiles across all judges
            if latency_columns:
                all_latencies_query = " UNION ALL ".join(
                    [
                        f'SELECT "{col}" as latency FROM {self.table_name} WHERE "{col}" IS NOT NULL'
                        for col in latency_columns
                    ]
                )

                try:
                    percentile_results = self.conn.execute(f"""
                        WITH all_latencies AS ({all_latencies_query})
                        SELECT 
                            MIN(latency) as min_latency,
                            percentile_cont(0.5) WITHIN GROUP (ORDER BY latency) as p50_latency,
                            percentile_cont(0.95) WITHIN GROUP (ORDER BY latency) as p95_latency,
                            MAX(latency) as max_latency,
                            AVG(latency) as mean_latency
                        FROM all_latencies
                    """).fetchone()

                    if percentile_results:
                        stats["performance.latency_min_ms"] = percentile_results[0]
                        stats["performance.latency_p50_ms"] = percentile_results[1]
                        stats["performance.latency_p95_ms"] = percentile_results[2]
                        stats["performance.latency_max_ms"] = percentile_results[3]
                        stats["performance.latency_mean_ms"] = percentile_results[4]
                except Exception:
                    pass

            # Token usage aggregations
            if token_total_columns:
                token_sums_query = " + ".join(
                    [f'COALESCE("{col}", 0)' for col in token_total_columns]
                )
                try:
                    token_stats = self.conn.execute(f"""
                        SELECT 
                            SUM({token_sums_query}) as total_tokens_used,
                            AVG({token_sums_query}) as avg_tokens_per_item
                        FROM {self.table_name}
                    """).fetchone()

                    if token_stats:
                        stats["performance.total_tokens_used"] = token_stats[0] or 0
                        stats["performance.avg_tokens_per_item"] = token_stats[1] or 0
                except Exception:
                    pass

            # Cache efficiency calculation
            if token_cached_columns and token_prompt_columns:
                cached_sums = " + ".join(
                    [f'COALESCE("{col}", 0)' for col in token_cached_columns]
                )
                prompt_sums = " + ".join(
                    [f'COALESCE("{col}", 0)' for col in token_prompt_columns]
                )

                try:
                    cache_stats = self.conn.execute(f"""
                        SELECT 
                            SUM({cached_sums}) as total_cached_tokens,
                            SUM({prompt_sums}) as total_prompt_tokens
                        FROM {self.table_name}
                    """).fetchone()

                    if cache_stats and cache_stats[1] and cache_stats[1] > 0:
                        cache_rate = cache_stats[0] / cache_stats[1] * 100
                        stats["performance.cache_hit_rate_pct"] = cache_rate
                        stats["performance.total_cached_tokens"] = cache_stats[0] or 0
                except Exception:
                    pass

        return stats

    def table_exists(self) -> bool:
        """Check if the results table exists."""
        if not self.conn:
            raise RuntimeError("Database connection not established")

        result = self.conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [self.table_name],
        ).fetchone()

        return result[0] > 0 if result else False
