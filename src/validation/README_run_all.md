# Unified rig runner

One CLI to drive all six validation rigs + an aggregate Plotly dashboard.

## Quickstart

```bash
# Smoke preset — full suite in ~6s, ~21 trials. CI-friendly.
python -m src.validation.run_all --preset smoke

# Standard — ~60s, ~91 trials. Day-to-day after a code change.
python -m src.validation.run_all --preset standard

# Full — paper-quality, ~1hr, ~500+ trials. Use before tagging a release.
python -m src.validation.run_all --preset full

# Subset of rigs
python -m src.validation.run_all --preset smoke --rigs rig1,rig3,rig6

# Skip per-rig HTML viz (faster, smaller output)
python -m src.validation.run_all --preset smoke --no-viz
```

## What it produces

```
reports/run_<UTC-timestamp>/
    index.html        ← aggregate dashboard (open this first)
    summary.json      ← machine-readable cross-rig summary
    rig1/results.json + viz.html
    rig2/results.json + viz.html
    ...
    rig6/results.json + viz.html
```

`index.html` is fully self-contained (Plotly is inlined) — works offline, can be
emailed, ZIPped, or committed to the `reports/` tree.

## Presets

| Preset    | Wall time | Trials | Use when                                    |
|-----------|-----------|--------|---------------------------------------------|
| smoke     | ~6 s      | ~21    | Pre-commit, CI smoke gate                   |
| standard  | ~60 s     | ~91    | After a planner / cost-function change      |
| full      | ~1 hr     | ~500   | Before tagging a release or paper figure    |

Edit `PRESETS` in `run_all.py` to tune sweep sizes and `gcopter_maxiter` levels.

## Behaviour worth knowing

- **Crash isolation.** A failing rig (bad config, optimizer NaN, missing
  dependency) is captured as `ok=False` with the traceback in `summary.json`
  and an error banner on its card. The other rigs continue.
- **Run order.** Cheap rigs (3, 4, 5) run first so the user sees early
  progress; expensive optimizer-bound rigs (1, 2) trail. Rig 2 always
  dominates wall time at non-smoke presets.
- **Git SHA capture.** The dashboard records the short SHA from `git
  rev-parse --short HEAD`, so a tagged-release `reports/` tree is fully
  reproducible.
- **Each rig still has its own CLI.** This runner imports `run_benchmark`
  directly — it doesn't replace `python -m src.validation.rig1_corridor_benchmark
  --densities 0.05,0.15 --runs 5` for focused debugging.

## Testing

```bash
pytest tests/test_run_all.py -v
```

12 tests in <1s: preset schema validity, adapter shape, crash isolation,
dashboard fragments, summary JSON well-formedness, end-to-end smoke subset.
