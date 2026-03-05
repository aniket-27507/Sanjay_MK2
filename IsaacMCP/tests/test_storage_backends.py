"""Tests for storage backends: PostgresStore (import check) and ParquetExporter."""

import pytest

from isaac_mcp.storage.sqlite_store import ExperimentStore


# --- PostgresStore tests ---


class TestPostgresStore:
    def test_import_without_asyncpg(self):
        """PostgresStore should raise ImportError without asyncpg."""
        # asyncpg may or may not be installed; test the import behavior
        try:
            from isaac_mcp.storage.postgres_store import PostgresStore, HAS_ASYNCPG
            if not HAS_ASYNCPG:
                with pytest.raises(ImportError, match="asyncpg"):
                    PostgresStore()
        except ImportError:
            pass  # Module itself failed to import, which is also fine

    def test_has_asyncpg_flag(self):
        """HAS_ASYNCPG should be a boolean."""
        from isaac_mcp.storage.postgres_store import HAS_ASYNCPG
        assert isinstance(HAS_ASYNCPG, bool)


# --- ParquetExporter tests ---


class TestParquetExporter:
    def test_import_without_pyarrow(self):
        """ParquetExporter should raise ImportError without pyarrow."""
        try:
            from isaac_mcp.storage.parquet_exporter import ParquetExporter, HAS_PYARROW
            if not HAS_PYARROW:
                store = ExperimentStore(":memory:")
                with pytest.raises(ImportError, match="pyarrow"):
                    ParquetExporter(store)
        except ImportError:
            pass

    def test_has_pyarrow_flag(self):
        """HAS_PYARROW should be a boolean."""
        from isaac_mcp.storage.parquet_exporter import HAS_PYARROW
        assert isinstance(HAS_PYARROW, bool)

    @pytest.mark.asyncio
    async def test_export_experiments_with_pyarrow(self, tmp_path):
        """Test export if pyarrow is available."""
        from isaac_mcp.storage.parquet_exporter import HAS_PYARROW
        if not HAS_PYARROW:
            pytest.skip("pyarrow not installed")

        from isaac_mcp.storage.parquet_exporter import ParquetExporter

        db_path = str(tmp_path / "test.db")
        store = ExperimentStore(db_path)
        await store.init_db()

        # Add test data
        exp_id = await store.save_experiment("s1", "batch", {"count": 3})
        for i in range(3):
            await store.save_run(exp_id, i, success=(i < 2), duration_s=1.0)

        exporter = ParquetExporter(store, str(tmp_path / "exports"))
        result = await exporter.export_experiments()
        assert result["exported"] == 1
        assert result["path"] != ""
        assert result["size_bytes"] > 0

    @pytest.mark.asyncio
    async def test_export_runs_with_pyarrow(self, tmp_path):
        """Test run export if pyarrow is available."""
        from isaac_mcp.storage.parquet_exporter import HAS_PYARROW
        if not HAS_PYARROW:
            pytest.skip("pyarrow not installed")

        from isaac_mcp.storage.parquet_exporter import ParquetExporter

        db_path = str(tmp_path / "test.db")
        store = ExperimentStore(db_path)
        await store.init_db()

        exp_id = await store.save_experiment("s1", "batch")
        await store.save_run(exp_id, 0, success=True, duration_s=1.5)
        await store.save_run(exp_id, 1, success=False, duration_s=2.0, failure_reason="crash")

        exporter = ParquetExporter(store, str(tmp_path / "exports"))
        result = await exporter.export_runs(exp_id)
        assert result["exported"] == 2

    @pytest.mark.asyncio
    async def test_export_all_with_pyarrow(self, tmp_path):
        """Test full export if pyarrow is available."""
        from isaac_mcp.storage.parquet_exporter import HAS_PYARROW
        if not HAS_PYARROW:
            pytest.skip("pyarrow not installed")

        from isaac_mcp.storage.parquet_exporter import ParquetExporter

        db_path = str(tmp_path / "test.db")
        store = ExperimentStore(db_path)
        await store.init_db()

        exp_id = await store.save_experiment("s1", "batch")
        await store.save_run(exp_id, 0, success=True, duration_s=1.0)

        exporter = ParquetExporter(store, str(tmp_path / "exports"))
        result = await exporter.export_all()
        assert result["experiments_exported"] == 1
        assert result["total_runs_exported"] == 1

    @pytest.mark.asyncio
    async def test_list_exports_with_pyarrow(self, tmp_path):
        """Test listing exports."""
        from isaac_mcp.storage.parquet_exporter import HAS_PYARROW
        if not HAS_PYARROW:
            pytest.skip("pyarrow not installed")

        from isaac_mcp.storage.parquet_exporter import ParquetExporter

        db_path = str(tmp_path / "test.db")
        store = ExperimentStore(db_path)
        await store.init_db()

        exp_id = await store.save_experiment("s1", "batch")
        await store.save_run(exp_id, 0, success=True, duration_s=1.0)

        exporter = ParquetExporter(store, str(tmp_path / "exports"))
        await exporter.export_experiments()
        files = exporter.list_exports()
        assert len(files) >= 1
        assert files[0]["name"] == "experiments.parquet"
