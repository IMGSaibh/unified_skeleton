from __future__ import annotations
from pathlib import Path
from pprint import pprint
from typing import List
import numpy as np
from numpy_2_nimble_skeleton import build_nimble_skeleton_from_json_npy
from skeleton_parser import SkeletonParser
import nimblephysics as nimble


def main():
    workspace_dir = Path.cwd()
    skeleton_parser = SkeletonParser()
    file_mname="short."
    json_path = f"{workspace_dir}/json_skeleton/{file_mname}_skeleton.json"
    npy_path = f"{workspace_dir}/npy/{file_mname}.npy"
    numpy_file = np.load(npy_path, allow_pickle=True)
    

    skeleton_spec = skeleton_parser.read_skeleton_json(json_path)
    
    source_skeleton: nimble.dynamics.Skeleton = nimble.dynamics.Skeleton()
    rajagopal_opensim: nimble.biomechanics.OpenSimFile = nimble.RajagopalHumanBodyModel()
    target_skeleton: nimble.dynamics.Skeleton = rajagopal_opensim.skeleton
    source_skeleton, _, _, names, _ = build_nimble_skeleton_from_json_npy(json_path, npy_path, unit_scale=0.001)
    print(source_skeleton.getNumJoints())
    print(source_skeleton.getJoint("jRightKnee"))  # → Joint-Objekt oder None

    converter: nimble.biomechanics.SkeletonConverter = nimble.biomechanics.SkeletonConverter(target_skeleton, source_skeleton)
    converter.linkJoints(target_skeleton.getJoint("radius_hand_l"), source_skeleton.getJoint("wrist_l"))



if __name__ == "__main__":
    main()
