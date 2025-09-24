# import os
# from pathlib import Path
# from pprint import pprint
# from typing import List
# import numpy as np
from skeleton_parser import SkeletonParser
import nimblephysics as nimble

def main():
  # workspace_dir = Path.cwd()
  # skeleton_parser = SkeletonParser()
  # file_mname="A_test"
  # json_path = f"{workspace_dir}/json_skeleton/{file_mname}_skeleton.json"
  # npy_path = f"{workspace_dir}/npy/{file_mname}.npy"
  # poses = np.load(npy_path, allow_pickle=True)
  
  # skeleton_spec = skeleton_parser.read_skeleton_json(json_path)
  # src_joint_names = skeleton_spec.joints

  # # source_skeleton: nimble.dynamics.Skeleton = nimble.dynamics.Skeleton()
  # # target_skeleton: nimble.dynamics.Skeleton = rajagopal_opensim.skeleton

  #   # 2) Ziel-Skelett laden (Rajagopal)
  # human: nimble.biomechanics.OpenSimFile = nimble.RajagopalHumanBodyModel()
  # target_skeleton: nimble.dynamics.Skeleton = human.skeleton
  # print(target_skeleton.getNumJoints())
  # print([target_skeleton.getJoint(i).getName() for i in range(target_skeleton.getNumJoints())])

  
  # # 3) IK → Ziel-DoFs (Q) im Rajagopal-Zielskelett
  # Q, mapped_names, mapped_idx = skeleton_parser.build_nimble_body_joints(
  #     src_joint_names,
  #     target_skeleton,
  #     poses,
  #     unit_scale=1.0,         # falls deine Eingabe mm ist: 0.001
  #     scale_bodies=False,     # True, wenn Segmentlängen mitgeschätzt werden sollen
  #     damping=1e-2,
  #     max_steps=100,
  #     verbose=True
  # )
  # np.save("unified/rajagopal_Q.npy", Q)

  opensim = nimble.RajagopalHumanBodyModel()
  world = nimble.simulation.World()
  world.addSkeleton(opensim.skeleton)

  gui = nimble.NimbleGUI(world) 
  gui.serve(8080)
  gui.blockWhileServing()


if __name__ == "__main__":
    main()
