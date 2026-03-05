"""Export experiment data to Parquet format for data lake integration.

Uses PyArrow for efficient columnar storage. Falls back gracefully
when pyarrow is not installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    pa = None  # type: ignore[assignment]
    pq = None  # type: ignore[assignment]
    HAS_PYARROW = False

from isaac_mcp.storage.sqlite_store import ExperimentStore


class ParquetExporter:
    """Export experiment and run data from SQLite to Parquet files.

    Parameters
    ----------
    store:
        The ExperimentStore to read data from.
    output_dir:
        Directory to write Parquet files.
    """

    def __init__(self, store: ExperimentStore, output_dir: str = "data/exports") -> None:
        if not HAS_PYARROW:
            raise ImportError(
                "pyarrow is required for ParquetExporter. "
                "Install it with: pip install pyarrow"
            )
        self._store = store
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    async def export_experiments(self, limit: int = 1000) -> dict[str, Any]:
        """Export experiments table to Parquet."""
        experiments = await self._store.list_experiments(limit=limit)
        if not experiments:
            return {"exported": 0, "path": ""}

        ids = []
        scenario_ids = []
        types = []
        created_ats = []
        success_rates = []
        total_runs_list = []

        for exp in experiments:
            ids.append(exp.get("id", ""))
            scenario_ids.append(exp.get("scenario_id", ""))
            types.append(exp.get("type", ""))
            created_ats.append(exp.get("created_at", ""))
            summary = exp.get("summary", {})
            success_rates.append(summary.get("success_rate", 0.0))
            total_runs_list.append(summary.get("total_runs", 0))

        table = pa.table({
            "id": pa.array(ids, type=pa.string()),
            "scenario_id": pa.array(scenario_ids, type=pa.string()),
            "type": pa.array(types, type=pa.string()),
            "created_at": pa.array(created_ats, type=pa.string()),
            "success_rate": pa.array(success_rates, type=pa.float64()),
            "total_runs": pa.array(total_runs_list, type=pa.int64()),
        })

        path = self._output_dir / "experiments.parquet"
        pq.write_table(table, str(path))

        return {
            "exported": len(experiments),
            "path": str(path),
            "size_bytes": path.stat().st_size,
        }

    async def export_runs(self, experiment_id: str) -> dict[str, Any]:
        """Export runs for a specific experiment to Parquet."""
        exp = await self._store.get_experiment(experiment_id)
        if exp is None:
            return {"exported": 0, "path": "", "error": "experiment_not_found"}

        runs = exp.get("runs", [])
        if not runs:
            return {"exported": 0, "path": ""}

        run_ids = []
        run_indices = []
        successes = []
        durations = []
        failure_reasons = []
        telemetry_jsons = []

        for run in runs:
            run_ids.append(run.get("id", ""))
            run_indices.append(run.get("run_index", 0))
            successes.append(run.get("success", False))
            durations.append(run.get("duration_s", 0.0))
            failure_reasons.append(run.get("failure_reason", ""))
            telemetry_jsons.append(json.dumps(run.get("telemetry", {})))

        table = pa.table({
            "id": pa.array(run_ids, type=pa.string()),
            "experiment_id": pa.array([experiment_id] * len(runs), type=pa.string()),
            "run_index": pa.array(run_indices, type=pa.int64()),
            "success": pa.array(successes, type=pa.bool_()),
            "duration_s": pa.array(durations, type=pa.float64()),
            "failure_reason": pa.array(failure_reasons, type=pa.string()),
            "telemetry_json": pa.array(telemetry_jsons, type=pa.string()),
        })

        path = self._output_dir / f"runs_{experiment_id}.parquet"
        pq.write_table(table, str(path))

        return {
            "exported": len(runs),
            "experiment_id": experiment_id,
            "path": str(path),
            "size_bytes": path.stat().st_size,
        }

    async def export_all(self, limit: int = 1000) -> dict[str, Any]:
        """Export both experiments and all their runs."""
        exp_result = await self.export_experiments(limit=limit)

        experiments = await self._store.list_experiments(limit=limit)
        run_exports: list[dict[str, Any]] = []
        total_runs = 0

        for exp in experiments:
            exp_id = exp.get("id", "")
            if exp_id:
                result = await self.export_runs(exp_id)
                total_runs += result.get("exported", 0)
                run_exports.append(result)

        return {
            "experiments_exported": exp_result.get("exported", 0),
            "experiments_path": exp_result.get("path", ""),
            "total_run_files": len(run_exports),
            "total_runs_exported": total_runs,
            "output_dir": str(self._output_dir),
        }

    def list_exports(self) -> list[dict[str, Any]]:
        """List existing Parquet export files."""
        files: list[dict[str, Any]] = []
        for path in sorted(self._output_dir.glob("*.parquet")):
            files.append({
                "name": path.name,
                "path": str(path),
                "size_bytes": path.stat().st_size,
            })
        return files
