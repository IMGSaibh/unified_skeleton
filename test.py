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
