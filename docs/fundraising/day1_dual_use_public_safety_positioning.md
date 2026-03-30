# Dual-Use Public Safety Positioning Brief

## Working thesis

Project Sanjay MK2 is building the autonomy and perception layer for small drone teams that helps operators detect, prioritize, and explain public-safety incidents before they are overloaded.

## Problem

Small teams supervising aerial assets face a real attention bottleneck. Operators are asked to monitor wide areas, notice weak or ambiguous signals, compare multiple incident types, and decide what deserves immediate follow-up. The result is delayed prioritization, inconsistent response, and limited ability to scale surveillance workflows without adding more people.

## Product

Sanjay MK2 is a software stack for operator-supervised aerial detection and response. It combines swarm coordination, scenario-tested mission policy, and edge-AI perception so a small drone team can surface likely incidents, maintain area coverage, and support human decision-making with a clearer operational picture.

## Wedge

The wedge is simulation-validated, operator-supervised, non-kinetic public-safety workflows. The system is not selling autonomous force or unattended deployment. It is designed to improve detection, prioritization, and explanation in bounded aerial overwatch and incident-monitoring use cases.

## Current truth

Today the project is a prototype-stage software platform. The simulation and autonomy backbone are implemented, and the edge-AI training and validation infrastructure is in place. The immediate work now underway is execution of that perception pipeline to produce the first trained checkpoint and validation report.

## Next milestone

The next concrete milestone is:

1. first trained aerial detection checkpoint
2. scenario-based validation against the current heuristic baseline
3. ONNX export for edge deployment direction
4. hardware prototype and hardware-in-the-loop validation

That milestone is the bridge from software prototype to investable physical-system execution.

## Explicit non-claim

This is not a field-ready police deployment system today. It is a prototype-stage autonomy and perception platform that is moving from simulation-grade software into trained perception, edge packaging, and hardware prototype validation.

## Why this framing

The current repo is strongest as a dual-use public-safety autonomy/perception stack. That framing keeps the existing public-safety use case intact while broadening investor accessibility beyond a narrow police-only narrative.
