# Archived Design Specs

These documents represent earlier design iterations that have been superseded
by the sensor-adaptive AI architecture (see `docs/ARCHITECTURE.md`).

## Why archived

- **TIDE spec + plans:** Designed a tri-modal always-on fusion network
  (RGB + thermal + LiDAR PointPillars). Replaced by sensor-adaptive
  scheduling with separate per-modal YOLO models and LiDAR as
  navigation-only.

- **SRO-MP spec:** Built around 6 Alpha + 1 Beta fleet model. The
  authoritative architecture is Alpha-only. See `src/response/mission_policy.py`
  for the current implementation.

Retained for historical reference. Do not use for new implementation work.
