from pathlib import Path
from pprint import pprint
import numpy as np
from skeleton_parser import SkeletonParser
import nimblephysics as nimble
import json

def main():
  workspace_dir = Path.cwd()
  file_mname="A_test"
  json_path = f"{workspace_dir}/json_skeleton/{file_mname}_skeleton.json"
  npy_path = f"{workspace_dir}/npy/{file_mname}.npy"
  poses = np.load(npy_path, allow_pickle=True)
  
  skeleton_parser = SkeletonParser()
  skeleton_parser.read_skeleton_json(json_path)
  rajagonal_human_model = nimble.RajagopalHumanBodyModel()
  nimble_skeleton = rajagonal_human_model.skeleton
  # nach was wird hier gemappt joints oder Dofs?
  print(f"nimble_skeleton.getNumJoints = {nimble_skeleton.getNumJoints()}")
  print(f"nimble_skeleton.getNumDofs = {nimble_skeleton.getNumDofs()}")
  pprint([nimble_skeleton.getJoint(i).getName() for i in range(nimble_skeleton.getNumJoints())])
  pprint(skeleton_parser.skeleton_spec.joints)
  pprint(skeleton_parser.skeleton_spec.edges)
  pprint(skeleton_parser.skeleton_spec.parents)
  pprint(skeleton_parser.skeleton_spec.roots)
  print("================================")


  mapping = skeleton_parser.build_mapping(nimble_skeleton, strict=False)
  pprint(mapping)
  pprint(mapping.tgt_names)
  # Quell-Joints in welcher Reihenfolge.
  pprint(mapping.src_indices)
  # Nimble-Joint-Handles in derselben Reihenfolge für IK and constraints
  pprint(mapping.tgt_joints)
  print("================================")

  frame0_points = np.asarray(poses[0], dtype=np.float64)  # (J, 3)
  print(f"frame0_points shape = {frame0_points.shape}")
  print(frame0_points)
  points2nimble_order = skeleton_parser.map_points_to_nimble_order(frame0_points, mapping)
  print(f" points2nimble_order = {points2nimble_order.shape}")
  pprint(points2nimble_order)
  print("================================")
  targets_col = skeleton_parser.targets_column_from_points(points2nimble_order)
  print("mapped:", points2nimble_order.shape, "targets:", targets_col.shape)

  # Mapping done?
  idx = np.array(mapping.src_indices)
  print("Index:", idx[:20])

  # zeig dir die ersten 5 Zuordnungen Quelle→Zielnamen
  if skeleton_parser.skeleton_spec is not None and hasattr(skeleton_parser.skeleton_spec, 'joints'):
    for i in range(16):
          print(f"{skeleton_parser.skeleton_spec.joints[idx[i]]}  ->  {mapping.tgt_names[i]}")

  
  
  # bodyJoints in derselben Reihenfolge wie points2nimble_order:
  targetPositions = points2nimble_order.astype(float)               # (K,3) Weltpunkte
  K = len(mapping.tgt_joints)
  assert targets_col.shape == (3*K, 1)
  assert targets_col.dtype == np.float64

  ik_error = nimble_skeleton.fitJointsToWorldPositions(mapping.tgt_joints, targets_col, scaleBodies=True)
  print("IK-Fehler:", ik_error)


  
  # converter: nimble.biomechanics.SkeletonConverter  = nimble.biomechanics.SkeletonConverter(sourceske, target_skeleton)
  # print(converter)


  






  


if __name__ == "__main__":
    main()
