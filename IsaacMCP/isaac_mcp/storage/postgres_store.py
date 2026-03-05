"""PostgreSQL storage backend (drop-in replacement for ExperimentStore).

Uses asyncpg for async PostgreSQL access. Falls back gracefully
when asyncpg is not installed. Schema mirrors the SQLite tables
exactly, so the API is identical.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

try:
    import asyncpg
    HAS_ASYNCPG = True
except ImportError:
    asyncpg = None  # type: ignore[assignment]
    HAS_ASYNCPG = False


class PostgresStore:
    """Async PostgreSQL store for experiments, drop-in compatible with ExperimentStore.

    Parameters
    ----------
    dsn:
        PostgreSQL connection string, e.g. ``postgresql://user:pass@host/dbname``.
    pool_min:
        Minimum connections in pool.
    pool_max:
        Maximum connections in pool.
    """

    def __init__(
        self,
        dsn: str = "postgresql://localhost/isaac_mcp",
        pool_min: int = 2,
        pool_max: int = 10,
    ) -> None:
        if not HAS_ASYNCPG:
            raise ImportError(
                "asyncpg is required for PostgresStore. "
                "Install it with: pip install asyncpg"
            )
        self._dsn = dsn
        self._pool_min = pool_min
        self._pool_max = pool_max
        self._pool: Any = None

    @property
    def dsn(self) -> str:
        return self._dsn

    async def init_db(self) -> None:
        """Create connection pool and tables."""
        self._pool = await asyncpg.create_pool(
            self._dsn,
            min_size=self._pool_min,
            max_size=self._pool_max,
        )
        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    scenario_id TEXT NOT NULL,
                    type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    config_json TEXT NOT NULL DEFAULT '{}'
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL REFERENCES experiments(id),
                    run_index INTEGER NOT NULL DEFAULT 0,
                    success INTEGER NOT NULL DEFAULT 0,
                    duration_s REAL NOT NULL DEFAULT 0.0,
                    failure_reason TEXT NOT NULL DEFAULT '',
                    telemetry_json TEXT NOT NULL DEFAULT '{}',
                    logs_json TEXT NOT NULL DEFAULT '[]'
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS scenarios (
                    id TEXT PRIMARY KEY,
                    base_scenario_id TEXT NOT NULL,
                    parameters_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                )
            """)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS scenario_results (
                    id TEXT PRIMARY KEY,
                    scenario_id TEXT NOT NULL REFERENCES scenarios(id),
                    success INTEGER NOT NULL DEFAULT 0,
                    failure_type TEXT NOT NULL DEFAULT '',
                    duration_s REAL NOT NULL DEFAULT 0.0,
                    telemetry_json TEXT NOT NULL DEFAULT '{}'
                )
            """)
            # Create indexes for common queries
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_runs_experiment ON runs(experiment_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_experiments_created ON experiments(created_at DESC)"
            )

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    async def save_experiment(
        self,
        scenario_id: str,
        experiment_type: str,
        config: dict[str, Any] | None = None,
    ) -> str:
        exp_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO experiments (id, scenario_id, type, created_at, config_json) VALUES ($1, $2, $3, $4, $5)",
                exp_id, scenario_id, experiment_type, now, json.dumps(config or {}),
            )
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
        run_id = uuid.uuid4().hex[:12]
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO runs (id, experiment_id, run_index, success, duration_s, failure_reason, telemetry_json, logs_json) "
                "VALUES ($1, $2, $3, $4, $5, $6, $7, $8)",
                run_id, experiment_id, run_index, 1 if success else 0,
                duration_s, failure_reason,
                json.dumps(telemetry or {}), json.dumps(logs or []),
            )
        return run_id

    async def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM experiments WHERE id = $1", experiment_id
            )
            if row is None:
                return None

            exp = dict(row)
            exp["config"] = json.loads(exp.pop("config_json", "{}"))

            run_rows = await conn.fetch(
                "SELECT * FROM runs WHERE experiment_id = $1 ORDER BY run_index",
                experiment_id,
            )
            runs = []
            for r in run_rows:
                run = dict(r)
                run["success"] = bool(run["success"])
                run["telemetry"] = json.loads(run.pop("telemetry_json", "{}"))
                run["logs"] = json.loads(run.pop("logs_json", "[]"))
                runs.append(run)

            exp["runs"] = runs
            total = len(runs)
            successes = sum(1 for r in runs if r["success"])
            exp["summary"] = {
                "total_runs": total,
                "successes": successes,
                "failures": total - successes,
                "success_rate": round(successes / total, 4) if total else 0.0,
            }
            return exp

    async def list_experiments(self, limit: int = 20) -> list[dict[str, Any]]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT * FROM experiments ORDER BY created_at DESC LIMIT $1",
                limit,
            )
            experiments = []
            for row in rows:
                exp = dict(row)
                exp["config"] = json.loads(exp.pop("config_json", "{}"))

                count_row = await conn.fetchrow(
                    "SELECT COUNT(*) as total, COALESCE(SUM(success), 0) as successes "
                    "FROM runs WHERE experiment_id = $1",
                    exp["id"],
                )
                total = count_row["total"] or 0
                successes = count_row["successes"] or 0
                exp["summary"] = {
                    "total_runs": total,
                    "successes": successes,
                    "failures": total - successes,
                    "success_rate": round(successes / total, 4) if total else 0.0,
                }
                experiments.append(exp)
            return experiments

    async def save_scenario(self, base_scenario_id: str, parameters: dict[str, Any]) -> str:
        scenario_id = uuid.uuid4().hex[:12]
        now = datetime.now(timezone.utc).isoformat()
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO scenarios (id, base_scenario_id, parameters_json, created_at) VALUES ($1, $2, $3, $4)",
                scenario_id, base_scenario_id, json.dumps(parameters), now,
            )
        return scenario_id

    async def save_scenario_result(
        self,
        scenario_id: str,
        success: bool,
        failure_type: str = "",
        duration_s: float = 0.0,
        telemetry: dict[str, Any] | None = None,
    ) -> str:
        result_id = uuid.uuid4().hex[:12]
        async with self._pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO scenario_results (id, scenario_id, success, failure_type, duration_s, telemetry_json) "
                "VALUES ($1, $2, $3, $4, $5, $6)",
                result_id, scenario_id, 1 if success else 0,
                failure_type, duration_s, json.dumps(telemetry or {}),
            )
        return result_id
