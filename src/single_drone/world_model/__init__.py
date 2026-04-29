"""
Project Sanjay Mk2 - World Model Submodule
==========================================
Polar-grid LiDAR encoding and predictive occupancy world-model utilities.

The world model is a planner cost bias for APF/HPL — never a motor command.
Raw LiDAR HPL still overrides whenever they disagree. Stale model output
is ignored.

@author: Archishman Paul
"""
