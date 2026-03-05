"""SQLite-backed knowledge graph for failure patterns, causes, and fixes.

Replaces the flat JSON knowledge base with a graph structure that tracks
relationships between errors, symptoms, root causes, and fixes.

Node types: ErrorPattern, Fix, Symptom, RootCause
Edge types: CAUSES, FIXES, CO_OCCURS, PRECEDED_BY
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import aiosqlite


@dataclass(slots=True)
class GraphNode:
    id: str
    node_type: str  # error_pattern | fix | symptom | root_cause
    label: str
    category: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""


@dataclass(slots=True)
class GraphEdge:
    id: str
    source_id: str
    target_id: str
    edge_type: str  # CAUSES | FIXES | CO_OCCURS | PRECEDED_BY
    weight: float = 1.0
    confidence: float = 0.5
    total_observations: int = 0
    successful_observations: int = 0
    last_observed: str = ""


@dataclass(slots=True)
class FixRecommendation:
    """A ranked fix recommendation from the knowledge graph."""
    fix_label: str
    fix_id: str
    success_rate: float
    total_attempts: int
    confidence: float
    source: str = "knowledge_graph"
    related_errors: list[str] = field(default_factory=list)


# Temporal decay half-life in days -- older observations contribute less
_DECAY_HALF_LIFE_DAYS = 90.0


def _temporal_weight(last_observed: str) -> float:
    """Compute a decay weight based on how recently the edge was observed."""
    if not last_observed:
        return 0.5
    try:
        observed_dt = datetime.fromisoformat(last_observed)
        now = datetime.now(timezone.utc)
        age_days = (now - observed_dt).total_seconds() / 86400.0
        return math.exp(-0.693 * age_days / _DECAY_HALF_LIFE_DAYS)
    except (ValueError, TypeError):
        return 0.5


class KnowledgeGraph:
    """SQLite-backed graph for error-fix relationships with temporal decay."""

    def __init__(self, db_path: str = "data/knowledge_graph.db"):
        self._db_path = db_path

    @property
    def db_path(self) -> str:
        return self._db_path

    async def init_db(self) -> None:
        """Create tables if they do not exist."""
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    node_type TEXT NOT NULL,
                    label TEXT NOT NULL,
                    category TEXT NOT NULL DEFAULT '',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                )
            """)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS edges (
                    id TEXT PRIMARY KEY,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    edge_type TEXT NOT NULL,
                    weight REAL NOT NULL DEFAULT 1.0,
                    confidence REAL NOT NULL DEFAULT 0.5,
                    total_observations INTEGER NOT NULL DEFAULT 0,
                    successful_observations INTEGER NOT NULL DEFAULT 0,
                    last_observed TEXT NOT NULL DEFAULT '',
                    FOREIGN KEY (source_id) REFERENCES nodes(id),
                    FOREIGN KEY (target_id) REFERENCES nodes(id)
                )
            """)
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(node_type)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_nodes_label ON nodes(label)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_nodes_category ON nodes(category)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target_id)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type)"
            )
            await db.commit()

    # --- Node operations ---

    async def add_node(
        self,
        node_type: str,
        label: str,
        category: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add a node or return existing node ID if label+type already exists."""
        import json

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT id FROM nodes WHERE node_type = ? AND label = ?",
                (node_type, label),
            )
            row = await cursor.fetchone()
            if row is not None:
                return dict(row)["id"]

            node_id = uuid.uuid4().hex[:12]
            now = datetime.now(timezone.utc).isoformat()
            await db.execute(
                "INSERT INTO nodes (id, node_type, label, category, metadata_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (node_id, node_type, label, category, json.dumps(metadata or {}), now),
            )
            await db.commit()
            return node_id

    async def get_node(self, node_id: str) -> GraphNode | None:
        """Retrieve a single node by ID."""
        import json

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM nodes WHERE id = ?", (node_id,))
            row = await cursor.fetchone()
            if row is None:
                return None
            r = dict(row)
            return GraphNode(
                id=r["id"],
                node_type=r["node_type"],
                label=r["label"],
                category=r["category"],
                metadata=json.loads(r.get("metadata_json", "{}")),
                created_at=r["created_at"],
            )

    async def find_nodes(
        self,
        node_type: str = "",
        label_contains: str = "",
        category: str = "",
        limit: int = 50,
    ) -> list[GraphNode]:
        """Search for nodes with optional filters."""
        import json

        conditions: list[str] = []
        params: list[Any] = []

        if node_type:
            conditions.append("node_type = ?")
            params.append(node_type)
        if label_contains:
            conditions.append("label LIKE ?")
            params.append(f"%{label_contains}%")
        if category:
            conditions.append("category = ?")
            params.append(category)

        where = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                f"SELECT * FROM nodes WHERE {where} ORDER BY created_at DESC LIMIT ?",
                params,
            )
            nodes: list[GraphNode] = []
            async for row in cursor:
                r = dict(row)
                nodes.append(GraphNode(
                    id=r["id"],
                    node_type=r["node_type"],
                    label=r["label"],
                    category=r["category"],
                    metadata=json.loads(r.get("metadata_json", "{}")),
                    created_at=r["created_at"],
                ))
            return nodes

    # --- Edge operations ---

    async def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        weight: float = 1.0,
        confidence: float = 0.5,
    ) -> str:
        """Add an edge or return existing edge ID if source+target+type already exists."""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT id FROM edges WHERE source_id = ? AND target_id = ? AND edge_type = ?",
                (source_id, target_id, edge_type),
            )
            row = await cursor.fetchone()
            if row is not None:
                return dict(row)["id"]

            edge_id = uuid.uuid4().hex[:12]
            now = datetime.now(timezone.utc).isoformat()
            await db.execute(
                "INSERT INTO edges (id, source_id, target_id, edge_type, weight, confidence, last_observed) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (edge_id, source_id, target_id, edge_type, weight, confidence, now),
            )
            await db.commit()
            return edge_id

    async def observe_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        success: bool,
    ) -> None:
        """Record an observation on an edge, updating weight and confidence."""
        now = datetime.now(timezone.utc).isoformat()

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM edges WHERE source_id = ? AND target_id = ? AND edge_type = ?",
                (source_id, target_id, edge_type),
            )
            row = await cursor.fetchone()

            if row is None:
                # Auto-create edge on first observation
                edge_id = uuid.uuid4().hex[:12]
                await db.execute(
                    "INSERT INTO edges (id, source_id, target_id, edge_type, weight, confidence, total_observations, successful_observations, last_observed) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (edge_id, source_id, target_id, edge_type, 1.0, 1.0 if success else 0.0, 1, 1 if success else 0, now),
                )
            else:
                r = dict(row)
                total = r["total_observations"] + 1
                successes = r["successful_observations"] + (1 if success else 0)
                confidence = round(successes / total, 4) if total > 0 else 0.0
                weight = confidence * _temporal_weight(now)

                await db.execute(
                    "UPDATE edges SET total_observations = ?, successful_observations = ?, confidence = ?, weight = ?, last_observed = ? WHERE id = ?",
                    (total, successes, confidence, round(weight, 4), now, r["id"]),
                )
            await db.commit()

    async def get_edges_from(
        self, source_id: str, edge_type: str = ""
    ) -> list[GraphEdge]:
        """Get all outgoing edges from a node, optionally filtered by type."""
        conditions = ["source_id = ?"]
        params: list[Any] = [source_id]
        if edge_type:
            conditions.append("edge_type = ?")
            params.append(edge_type)

        where = " AND ".join(conditions)

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                f"SELECT * FROM edges WHERE {where} ORDER BY weight DESC", params
            )
            edges: list[GraphEdge] = []
            async for row in cursor:
                r = dict(row)
                edges.append(GraphEdge(
                    id=r["id"],
                    source_id=r["source_id"],
                    target_id=r["target_id"],
                    edge_type=r["edge_type"],
                    weight=r["weight"],
                    confidence=r["confidence"],
                    total_observations=r["total_observations"],
                    successful_observations=r["successful_observations"],
                    last_observed=r["last_observed"],
                ))
            return edges

    async def get_edges_to(
        self, target_id: str, edge_type: str = ""
    ) -> list[GraphEdge]:
        """Get all incoming edges to a node, optionally filtered by type."""
        conditions = ["target_id = ?"]
        params: list[Any] = [target_id]
        if edge_type:
            conditions.append("edge_type = ?")
            params.append(edge_type)

        where = " AND ".join(conditions)

        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                f"SELECT * FROM edges WHERE {where} ORDER BY weight DESC", params
            )
            edges: list[GraphEdge] = []
            async for row in cursor:
                r = dict(row)
                edges.append(GraphEdge(
                    id=r["id"],
                    source_id=r["source_id"],
                    target_id=r["target_id"],
                    edge_type=r["edge_type"],
                    weight=r["weight"],
                    confidence=r["confidence"],
                    total_observations=r["total_observations"],
                    successful_observations=r["successful_observations"],
                    last_observed=r["last_observed"],
                ))
            return edges

    # --- High-level copilot operations ---

    async def record_fix_outcome(
        self,
        error_type: str,
        cause: str,
        fix_applied: str,
        success: bool,
        category: str = "",
    ) -> None:
        """Record a fix attempt outcome, building graph relationships.

        Creates/updates nodes for the error, cause, and fix, and edges between them.
        This is the primary learning entry point for the copilot.
        """
        # Ensure nodes exist
        error_node_id = await self.add_node("error_pattern", error_type, category=category)
        fix_node_id = await self.add_node("fix", fix_applied, category=category)

        # Record the FIXES relationship
        await self.observe_edge(fix_node_id, error_node_id, "FIXES", success)

        # If a cause is provided, record CAUSES relationship
        if cause:
            cause_node_id = await self.add_node("root_cause", cause, category=category)
            await self.observe_edge(cause_node_id, error_node_id, "CAUSES", True)

    async def record_co_occurrence(
        self, error_type_a: str, error_type_b: str, category: str = ""
    ) -> None:
        """Record that two errors co-occurred in the same diagnosis."""
        node_a = await self.add_node("error_pattern", error_type_a, category=category)
        node_b = await self.add_node("error_pattern", error_type_b, category=category)
        # Bidirectional co-occurrence
        await self.observe_edge(node_a, node_b, "CO_OCCURS", True)
        await self.observe_edge(node_b, node_a, "CO_OCCURS", True)

    async def record_error_sequence(
        self, preceding_error: str, following_error: str, category: str = ""
    ) -> None:
        """Record that one error preceded another in time."""
        node_a = await self.add_node("error_pattern", preceding_error, category=category)
        node_b = await self.add_node("error_pattern", following_error, category=category)
        await self.observe_edge(node_a, node_b, "PRECEDED_BY", True)

    async def query_fixes(
        self,
        error_type: str,
        category: str = "",
        include_related: bool = True,
        limit: int = 10,
    ) -> list[FixRecommendation]:
        """Query the graph for fix recommendations for a given error type.

        Searches for direct FIXES edges and optionally traverses CO_OCCURS
        edges to find fixes for related errors.
        """
        recommendations: list[FixRecommendation] = []
        seen_fix_ids: set[str] = set()

        # Find the error node(s) matching this error type
        error_nodes = await self.find_nodes(
            node_type="error_pattern", label_contains=error_type, category=category
        )

        for error_node in error_nodes:
            # Direct fixes: edges where fix -> error with type FIXES
            fix_edges = await self.get_edges_to(error_node.id, edge_type="FIXES")
            for edge in fix_edges:
                if edge.source_id in seen_fix_ids:
                    continue
                seen_fix_ids.add(edge.source_id)

                fix_node = await self.get_node(edge.source_id)
                if fix_node is None:
                    continue

                total = edge.total_observations
                success_rate = (
                    edge.successful_observations / total if total > 0 else 0.0
                )
                decay = _temporal_weight(edge.last_observed)

                recommendations.append(FixRecommendation(
                    fix_label=fix_node.label,
                    fix_id=fix_node.id,
                    success_rate=round(success_rate, 4),
                    total_attempts=total,
                    confidence=round(edge.confidence * decay, 4),
                    source="direct",
                    related_errors=[error_node.label],
                ))

        # Traverse co-occurrence edges for related error fixes
        if include_related:
            for error_node in error_nodes:
                co_edges = await self.get_edges_from(error_node.id, edge_type="CO_OCCURS")
                for co_edge in co_edges[:5]:  # Limit traversal breadth
                    related_fix_edges = await self.get_edges_to(
                        co_edge.target_id, edge_type="FIXES"
                    )
                    for edge in related_fix_edges:
                        if edge.source_id in seen_fix_ids:
                            continue
                        seen_fix_ids.add(edge.source_id)

                        fix_node = await self.get_node(edge.source_id)
                        if fix_node is None:
                            continue

                        related_node = await self.get_node(co_edge.target_id)
                        total = edge.total_observations
                        success_rate = (
                            edge.successful_observations / total if total > 0 else 0.0
                        )
                        decay = _temporal_weight(edge.last_observed)

                        recommendations.append(FixRecommendation(
                            fix_label=fix_node.label,
                            fix_id=fix_node.id,
                            success_rate=round(success_rate, 4),
                            total_attempts=total,
                            # Discount confidence for indirect (co-occurrence based) fixes
                            confidence=round(edge.confidence * decay * co_edge.confidence * 0.7, 4),
                            source="co_occurrence",
                            related_errors=[
                                related_node.label if related_node else "unknown"
                            ],
                        ))

        # Sort by confidence (accounts for success rate + recency + directness)
        recommendations.sort(key=lambda r: r.confidence, reverse=True)
        return recommendations[:limit]

    async def get_statistics(self) -> dict[str, Any]:
        """Return graph-wide statistics."""
        async with aiosqlite.connect(self._db_path) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM nodes")
            total_nodes = (await cursor.fetchone())[0]

            cursor = await db.execute("SELECT COUNT(*) FROM edges")
            total_edges = (await cursor.fetchone())[0]

            cursor = await db.execute(
                "SELECT node_type, COUNT(*) as cnt FROM nodes GROUP BY node_type"
            )
            nodes_by_type = {row[0]: row[1] async for row in cursor}

            cursor = await db.execute(
                "SELECT edge_type, COUNT(*) as cnt FROM edges GROUP BY edge_type"
            )
            edges_by_type = {row[0]: row[1] async for row in cursor}

            # Fix success statistics
            cursor = await db.execute(
                "SELECT SUM(total_observations), SUM(successful_observations) FROM edges WHERE edge_type = 'FIXES'"
            )
            row = await cursor.fetchone()
            total_fix_attempts = row[0] or 0
            total_fix_successes = row[1] or 0

            return {
                "total_nodes": total_nodes,
                "total_edges": total_edges,
                "nodes_by_type": nodes_by_type,
                "edges_by_type": edges_by_type,
                "fix_statistics": {
                    "total_attempts": total_fix_attempts,
                    "total_successes": total_fix_successes,
                    "overall_success_rate": (
                        round(total_fix_successes / total_fix_attempts, 4)
                        if total_fix_attempts > 0
                        else 0.0
                    ),
                },
            }

    async def get_graph_summary(self) -> dict[str, Any]:
        """Return a compact summary suitable for LLM context / MCP resource."""
        stats = await self.get_statistics()

        # Get top error patterns by fix count
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("""
                SELECT n.label, n.category, COUNT(e.id) as fix_count,
                       SUM(e.successful_observations) as successes,
                       SUM(e.total_observations) as attempts
                FROM nodes n
                JOIN edges e ON e.target_id = n.id AND e.edge_type = 'FIXES'
                WHERE n.node_type = 'error_pattern'
                GROUP BY n.id
                ORDER BY fix_count DESC
                LIMIT 20
            """)
            top_errors: list[dict[str, Any]] = []
            async for row in cursor:
                r = dict(row)
                attempts = r["attempts"] or 0
                successes = r["successes"] or 0
                top_errors.append({
                    "error": r["label"],
                    "category": r["category"],
                    "known_fixes": r["fix_count"],
                    "attempts": attempts,
                    "success_rate": round(successes / attempts, 4) if attempts > 0 else 0.0,
                })

        return {
            "statistics": stats,
            "top_error_patterns": top_errors,
        }

    async def bootstrap_from_error_patterns(
        self, error_patterns: list[dict[str, str]]
    ) -> int:
        """Seed the graph from existing error pattern definitions.

        Creates error_pattern nodes and fix nodes, and FIXES edges between them.
        Returns the number of nodes added.
        """
        added = 0
        for pattern in error_patterns:
            if not isinstance(pattern, dict):
                continue
            error_label = pattern.get("description", "")
            fix_label = pattern.get("fix", "")
            category = pattern.get("category", "")
            if not error_label or not fix_label:
                continue

            error_id = await self.add_node("error_pattern", error_label, category=category)
            fix_id = await self.add_node("fix", fix_label, category=category)
            await self.add_edge(fix_id, error_id, "FIXES", weight=0.5, confidence=0.5)
            added += 1

        return added
