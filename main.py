from __future__ import annotations
from pathlib import Path
from pprint import pprint
from typing import List
import numpy as np
from skeleton_parser import SkeletonParser
import nimblephysics as nimble
import os

import nimblephysics as nimble
import torch

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
  # human: nimble.biomechanics.OpenSimFile = nimble.RajagopalHumanBodyModel()
  # skeleton = human.skeleton

  # # target_skeleton: nimble.dynamics.Skeleton = rajagopal_opensim.skeleton

  # skeleton_parser.build_nimble_body_joints(src_joint_names, skeleton, poses)
  # # converter: nimble.biomechanics.SkeletonConverter = nimble.biomechanics.SkeletonConverter(target_skeleton, source_skeleton)
  # # converter.linkJoints(target_skeleton.getJoint("radius_hand_l"), source_skeleton.getJoint("wrist_l"))


  world = nimble.loadWorld("./half_cheetah.skel")
  initialState = torch.zeros((world.getStateSize()))
  action = torch.zeros((world.getActionSize()))
  state = initialState
  states = []
  for _ in range(300):
    state = nimble.timestep(world, state, action)
    states.append(state)

  # Display our trajectory in a GUI

  gui = nimble.NimbleGUI(world)
  gui.serve(8080) # host the GUI on localhost:8080
  gui.loopStates(states) # tells the GUI to animate our list of states
  gui.blockWhileServing() # block here so we don't exit the program


if __name__ == "__main__":
    main()
