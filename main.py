#!/usr/bin/env python3
"""
Step 1: JSON-Skeleton einlesen & aufbereiten (für späteres Nimble-Fitting).

Erwartetes JSON (wie bei dir):
{
  "joints": ["Hips","Chest","Chest2","Chest3","Chest4","Neck","Head",
             "RightCollar","RightShoulder","RightElbow","RightWrist",
             "LeftCollar","LeftShoulder","LeftElbow","LeftWrist",
             "RightHip","RightKnee","RightAnkle","RightToe",
             "LeftHip","LeftKnee","LeftAnkle","LeftToe"],
  "hierarchy": [[child_idx, parent_idx], ...],
  // optionale Felder für spätere Schritte:
  // "map_to_nimble": { "RightKnee":"knee_r", ... },
  // "unit_scale": 0.01,   # cm -> m (oder 0.001 für mm -> m)
  // "scaleBodies": true
}

Benutzung:
  python step1_read_json.py --json A_test_skeleton.json --write-template
"""

from __future__ import annotations
from pathlib import Path
from pprint import pprint

import numpy as np
from skeleton_parser import SkeletonParser
import nimblephysics as nimble


def main():
    workspace_dir = Path.cwd()
    skeleton_parser = SkeletonParser()
    file_mname="short."
    json_path = f"{workspace_dir}/json_skeleton/{file_mname}_skeleton.json"
    template_path = f"{workspace_dir}/skeleton_map/map_{file_mname}_skeleton.json"
    

    skeleton_spec = skeleton_parser.read_skeleton_json(json_path, write_template=template_path)
    skeleton_parser.print_skeleton_spec()
    
    # Ab hier könntest du direkt in Step 2 (NumPy laden & gegen J validieren) gehen,
    # oder das Mapping-Template öffnen und anpassen.
    # Der nächste Schritt (für Nimble) nutzt:
    #  - spec.joints (Reihenfolge)
    #  - Mapping: map_to_nimble (aus JSON/Template)
    #  - optional: unit_scale, scaleBodies

    # Create a GUI
    gui = nimble.NimbleGUI()
    # Serve the GUI on port 8080
    gui.serve(8080)
    api = gui.nativeAPI()

    api.createSphere("ball", radii=np.array([0.2, 0.2, 0.2]), pos=np.array([0., 0.2, 0.]))
    api.createBox("ground", size=np.array([5., 0.1, 5.]), pos=np.array([0., -0.05, 0.]))


    # Block until the GUI is closed
    gui.blockWhileServing()


if __name__ == "__main__":
    main()
