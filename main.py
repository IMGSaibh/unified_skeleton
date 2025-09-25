# import os
from pathlib import Path
# from pprint import pprint
# from typing import List
# import numpy as np
import numpy as np
from skeleton_parser import SkeletonParser
import nimblephysics as nimble

def main():
  workspace_dir = Path.cwd()
  skeleton_parser = SkeletonParser()
  file_mname="A_test"
  json_path = f"{workspace_dir}/json_skeleton/{file_mname}_skeleton.json"
  npy_path = f"{workspace_dir}/npy/{file_mname}.npy"
  poses = np.load(npy_path, allow_pickle=True)
  
  skeleton_parser.read_skeleton_json(json_path)
  human: nimble.biomechanics.OpenSimFile = nimble.RajagopalHumanBodyModel()
  target_skeleton: nimble.dynamics.Skeleton = human.skeleton
  print(target_skeleton.getNumJoints())
  print([target_skeleton.getJoint(i).getName() for i in range(target_skeleton.getNumJoints())])

  skeleton_parser.create_body_joints_list(target_skeleton)
  
  # 3) IK → Ziel-DoFs (Q) im Rajagopal-Zielskelett
  Q = skeleton_parser.build_nimble_body_joints(
      target_skeleton,
      poses,
      unit_scale=1.0,         
      scale_bodies=False,     # True, wenn Segmentlängen mitgeschätzt werden sollen
      damping=1e-2,
      max_steps=100
  )

  # rajagopal_opensim: nimble.biomechanics.OpenSimFile = nimble.RajagopalHumanBodyModel()
  # skeleton: nimble.dynamics.Skeleton = rajagopal_opensim.skeleton
  # right_wrist: nimble.dynamics.Joint = skeleton.getJoint("radius_hand_r")
  # # Set an arbitrary target location
  # target: np.ndarray = np.array([0.5, 0.5, 0.5])

  # # Get the world location of the wrist
  # wrist_pos: np.ndarray = skeleton.getJointWorldPositions([right_wrist])

  # world = nimble.simulation.World()
  # world.addSkeleton(skeleton)

  # # gui = nimble.NimbleGUI(world)
  # gui = nimble.NimbleGUI(world)
  # gui.serve(8080)
  # # gui.nativeAPI().createLine(key="wrist_error", points=[wrist_pos, target], color=[1.0, 0.0, 0.0, 1.0])
  # gui.blockWhileServing()


  


if __name__ == "__main__":
    main()
