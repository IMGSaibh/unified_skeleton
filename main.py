# import os
from pathlib import Path
from pprint import pprint
# from typing import List
# import numpy as np
import numpy as np
from skeleton_parser import SkeletonParser
import nimblephysics as nimble
import nimblephysics_libs as nimble_libs
import json

def main():
  workspace_dir = Path.cwd()
  file_mname="A_test"
  json_path = f"{workspace_dir}/json_skeleton/{file_mname}_skeleton.json"
  npy_path = f"{workspace_dir}/npy/{file_mname}.npy"
  poses = np.load(npy_path, allow_pickle=True)
  
  skeleton_parser = SkeletonParser()
  skeleton_parser.read_skeleton_json(json_path)
  rajagonalHumanModel = nimble.RajagopalHumanBodyModel()
  target_skeleton = rajagonalHumanModel.skeleton
  # nach was wird hier gemappt joints oder Dofs?
  print(target_skeleton.getNumJoints())
  print(target_skeleton.getNumDofs())
  print([target_skeleton.getJoint(i).getName() for i in range(target_skeleton.getNumJoints())])
  pprint(skeleton_parser.skeleton_spec)
  print("================================")


  mapping = skeleton_parser.build_mapping(target_skeleton, strict=False)
  pprint(mapping)
  pprint(mapping.tgt_names)
  # Quell-Joints in welcher Reihenfolge.
  pprint(mapping.src_indices)
  # Nimble-Joint-Handles in derselben Reihenfolge für IK and constraints
  pprint(mapping.tgt_joints)
  print("================================")

  # pts_k = skeleton_parser.map_points_to_nimble_order(frame0_points, mapping)
  
  
  # converter: nimble.biomechanics.SkeletonConverter  = nimble.biomechanics.SkeletonConverter(sourceske, target_skeleton)
  # print(converter)


  






  


if __name__ == "__main__":
    main()
