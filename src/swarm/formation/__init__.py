"""
Project Sanjay Mk2 - Formation Submodule
=========================================
Geographic formation maintenance and alignment.

Components:
    - FormationController: Real-time formation keeping with
      multiple geometry patterns (hex, linear, wedge, ring, diamond)

@author: Archishman Paul
"""

from .formation_controller import FormationController, FormationConfig, FormationType

__all__ = [
    "FormationController",
    "FormationConfig",
    "FormationType",
]
