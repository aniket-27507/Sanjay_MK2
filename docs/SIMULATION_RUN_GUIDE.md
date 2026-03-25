# Simulation Run Guide

This guide focuses on the current simulation surfaces and tells you which one is authoritative for the implemented architecture.

## Which Path To Use

### Use the scenario framework when you want the current architecture

This is the most aligned path for the implemented Alpha-only police swarm:

- `6` Alpha drones
- deterministic mission-policy gating
- Alpha self-confirmation
- sector backfill behavior
- crowd-overwatch behavior

Primary entry points:

- [src/simulation/scenario_loader.py](/Users/archishmanpaul/Desktop/Sanjay_MK2/src/simulation/scenario_loader.py)
- [src/simulation/scenario_executor.py](/Users/archishmanpaul/Desktop/Sanjay_MK2/src/simulation/scenario_executor.py)
- [config/scenarios](/Users/archishmanpaul/Desktop/Sanjay_MK2/config/scenarios)

### Use Isaac Sim when you want bridge/topic validation

The Isaac path is useful, but it still carries some legacy Beta compatibility. Use it for integration work, not as the sole statement of final deployment architecture.

## Fast Scenario Runs

### Run a baseline patrol scenario

```bash
python3 scripts/run_scenario.py --scenario S10
```

### Run a high-rise threat scenario

```bash
python3 scripts/run_scenario.py --scenario S24
```

### Run targeted tests for the current autonomy path

```bash
python3 -m pytest tests/test_mission_policy.py -q
python3 -m pytest tests/test_scenario_framework.py -q
```

### Useful validation suites

```bash
python3 -m pytest tests/test_config_manager.py tests/test_world_and_sensors.py tests/test_drone_types.py -q
python3 -m pytest tests/test_isaac_sim_bridge.py -q
```

## What The Scenario Path Exercises

The scenario executor currently validates:

- Alpha sector patrol
- change detection and threat creation
- multi-sensor threat scoring
- deterministic policy gating
- Alpha inspection assignment
- zoom EO confirmation
- sector backfill while one Alpha inspects
- crowd-overwatch retasking without descent
- GCS telemetry and threat output

## Isaac Sim Bring-Up

If you need the high-fidelity integration path:

1. follow [docs/ISAAC_SIM_SETUP.md](/Users/archishmanpaul/Desktop/Sanjay_MK2/docs/ISAAC_SIM_SETUP.md)
2. start Isaac Sim and the ROS 2 bridge
3. run the mission tooling or inspect ROS topics directly

Example ROS checks:

```bash
ros2 topic list
ros2 topic hz /alpha_0/odom
ros2 topic echo /alpha_0/rgb/image_raw --once
```

## Expected Current Behavior

### Scenario path

The police/autonomy path should now behave like this:

- all active drones are Alpha drones
- one Alpha may be assigned as inspector
- crowd workflows stay high
- GCS telemetry includes mission state, inspection state, and sector backfill state

### Isaac path

The Isaac path currently behaves like this:

- Alpha topics are aligned around `rgb + thermal + lidar_3d`
- the bridge and scene still retain a legacy `beta_0`
- that Beta should be treated as compatibility scaffolding

## Operational Notes

- do not treat old hardcoded test counts as authoritative; run the current suites instead
- if a doc, scenario comment, or planning note still mentions Beta dispatch, treat the runtime code and canonical docs as authoritative
- the cleanest source of truth for the implemented autonomy path is [scenario_executor.py](/Users/archishmanpaul/Desktop/Sanjay_MK2/src/simulation/scenario_executor.py)

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Scenario tests pass but police scenario tests fail to even collect | The current Homebrew Python 3.14 environment aborts when importing the existing Torch-based crowd-density stack; use the intended Python 3.11 environment for those suites |
| Isaac topics are visible but you still see Beta references | The Isaac path still retains legacy compatibility; this does not mean the authoritative deployment architecture is Beta-based |
| GCS telemetry looks incomplete | Make sure you are running the updated scenario path; it now emits mission, inspection, and backfill state |
