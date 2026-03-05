"""Async SQLite storage for experiments and scenario lab results."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

import aiosqlite


class ExperimentStore:
    """Async SQLite store for experiment and scenario lab data."""

    def __init__(self, db_path: str = "data/isaac_experiments.db"):
        self._db_path = db_path

    @property
    def db_path(self) -> str:
        return self._db_path

    async def init_db(self) -> None:
        """Create tables if they do not exist."""
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    scenario_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    config_json TEXT NOT NULL DEFAULT '{}'
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    run_index INTEGER NOT NULL DEFAULT 0,
                    success INTEGER NOT NULL DEFAULT 0,
                    duration_s REAL NOT NULL DEFAULT 0.0,
                    failure_reason TEXT NOT NULL DEFAULT '',
                    telemetry_json TEXT NOT NULL DEFAULT '{}',
                    logs_json TEXT NOT NULL DEFAULT '[]',
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS scenarios (
                    id TEXT PRIMARY KEY,
                    base_scenario_id TEXT NOT NULL,
                    parameters_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS scenario_results (
                    id TEXT PRIMARY KEY,
                    scenario_id TEXT NOT NULL,
                    success INTEGER NOT NULL DEFAULT 0,
                    failure_type TEXT NOT NULL DEFAULT '',
                    duration_s REAL NOT NULL DEFAULT 0.0,
                    telemetry_json TEXT NOT NULL DEFAULT '{}',
                    FOREIGN KEY (scenario_id) REFERENCES scenarios(id)
                )
            """)
            await db.commit()

    async def save_experiment(self, scenario_id: str, experiment_type: str, config: dict[str, Any] | None = None) -> str:
        """Save a new experiment and return its ID."""
        exp_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO experiments (id, scenario_id, type, created_at, config_json) VALUES (?, ?, ?, ?, ?)",
                (exp_id, scenario_id, experiment_type, now, json.dumps(config or {})),
            )
            await db.commit()
        return exp_id

    async def save_run(
        self,
        experiment_id: str,
        run_index: int,
        success: bool,
        duration_s: float,
        failure_reason: str = "",
        telemetry: dict[str, Any] | None = None,
        logs: list[str] | None = None,
    ) -> str:
        """Save an individual run result and return its ID."""
        run_id = uuid.uuid4().hex[:12]
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO runs (id, experiment_id, run_index, success, duration_s, failure_reason, telemetry_json, logs_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    experiment_id,
                    run_index,
                    1 if success else 0,
                    duration_s,
                    failure_reason,
                    json.dumps(telemetry or {}),
                    json.dumps(logs or []),
                ),
            )
            await db.commit()
        return run_id

    async def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        """Retrieve an experiment with its runs."""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,))
            row = await cursor.fetchone()
            if row is None:
                return None

            experiment = dict(row)
            experiment["config"] = json.loads(experiment.pop("config_json", "{}"))

            cursor = await db.execute(
                "SELECT * FROM runs WHERE experiment_id = ? ORDER BY run_index", (experiment_id,)
            )
            runs = []
            async for run_row in cursor:
                run = dict(run_row)
                run["success"] = bool(run["success"])
                run["telemetry"] = json.loads(run.pop("telemetry_json", "{}"))
                run["logs"] = json.loads(run.pop("logs_json", "[]"))
                runs.append(run)

            experiment["runs"] = runs
            total = len(runs)
            successes = sum(1 for r in runs if r["success"])
            experiment["summary"] = {
                "total_runs": total,
                "successes": successes,
                "failures": total - successes,
                "success_rate": round(successes / total, 4) if total else 0.0,
            }
            return experiment

    async def list_experiments(self, limit: int = 20) -> list[dict[str, Any]]:
        """List recent experiments with summary statistics."""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM experiments ORDER BY created_at DESC LIMIT ?", (limit,)
            )
            experiments = []
            async for row in cursor:
                exp = dict(row)
                exp["config"] = json.loads(exp.pop("config_json", "{}"))

                # Get run summary
                count_cursor = await db.execute(
                    "SELECT COUNT(*) as total, SUM(success) as successes FROM runs WHERE experiment_id = ?",
                    (exp["id"],),
                )
                count_row = await count_cursor.fetchone()
                total = dict(count_row)["total"] or 0
                successes = dict(count_row)["successes"] or 0
                exp["summary"] = {
                    "total_runs": total,
                    "successes": successes,
                    "failures": total - successes,
                    "success_rate": round(successes / total, 4) if total else 0.0,
                }
                experiments.append(exp)
            return experiments

    async def get_sweep_results(self, experiment_id: str) -> dict[str, Any] | None:
        """Retrieve experiment with runs grouped by parameter value for sweep analysis."""
        experiment = await self.get_experiment(experiment_id)
        if experiment is None:
            return None

        config = experiment.get("config", {})
        parameter = config.get("parameter", "unknown")
        runs = experiment.get("runs", [])

        # Group runs by parameter value (stored in telemetry.sweep_value)
        groups: dict[str, list[dict[str, Any]]] = {}
        for run in runs:
            value = str(run.get("telemetry", {}).get("sweep_value", "unknown"))
            groups.setdefault(value, []).append(run)

        sweep_points: list[dict[str, Any]] = []
        for value, group_runs in sorted(groups.items(), key=lambda x: x[0]):
            total = len(group_runs)
            successes = sum(1 for r in group_runs if r["success"])
            sweep_points.append({
                "parameter_value": value,
                "total_runs": total,
                "successes": successes,
                "success_rate": round(successes / total, 4) if total else 0.0,
                "avg_duration_s": round(sum(r["duration_s"] for r in group_runs) / total, 3) if total else 0.0,
            })

        experiment["parameter"] = parameter
        experiment["sweep_points"] = sweep_points
        return experiment

    # --- Scenario Lab tables ---

    async def save_scenario(self, base_scenario_id: str, parameters: dict[str, Any]) -> str:
        """Save a generated scenario and return its ID."""
        scenario_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO scenarios (id, base_scenario_id, parameters_json, created_at) VALUES (?, ?, ?, ?)",
                (scenario_id, base_scenario_id, json.dumps(parameters), now),
            )
            await db.commit()
        return scenario_id

    async def save_scenario_result(
        self,
        scenario_id: str,
        success: bool,
        failure_type: str = "",
        duration_s: float = 0.0,
        telemetry: dict[str, Any] | None = None,
    ) -> str:
        """Save a scenario result and return its ID."""
        result_id = uuid.uuid4().hex[:12]
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "INSERT INTO scenario_results (id, scenario_id, success, failure_type, duration_s, telemetry_json) VALUES (?, ?, ?, ?, ?, ?)",
                (result_id, scenario_id, 1 if success else 0, failure_type, duration_s, json.dumps(telemetry or {})),
            )
            await db.commit()
        return result_id

    async def get_scenario_results(self, scenario_ids: list[str]) -> list[dict[str, Any]]:
        """Retrieve results for a list of scenario IDs."""
        if not scenario_ids:
            return []
        placeholders = ",".join("?" for _ in scenario_ids)
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                f"SELECT * FROM scenario_results WHERE scenario_id IN ({placeholders})",
                scenario_ids,
            )
            results = []
            async for row in cursor:
                r = dict(row)
                r["success"] = bool(r["success"])
                r["telemetry"] = json.loads(r.pop("telemetry_json", "{}"))
                results.append(r)
            return results
