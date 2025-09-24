from collections import deque
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from typing import List, Tuple
import nimblephysics as nimble

import numpy as np

MAP = {
    # Becken / Root
    "Hips"          : "pelvis",
    
    # Beine rechts
    "RightHip"      : "r_hip",
    "RightKnee"     : "r_knee",
    "RightAnkle"    : "r_ankle",
    "RightToe"      : "r_toe",
    
    # Beine links
    "LeftHip"       : "l_hip",
    "LeftKnee"      : "l_knee",
    "LeftAnkle"     : "l_ankle",
    "LeftToe"       : "l_toe",
    
    # Wirbelsäule
    "Chest4"        : "spine",
    
    # Hals/Kopf
    "Neck"          : "neck",
    "Head"          : "head",
    
    # Schultergürtel rechts
    "RightShoulder" : "r_shoulder",
    "RightElbow"    : "r_elbow",
    "RightWrist"    : "r_wrist",
    
    # Schultergürtel links
    "LeftShoulder"  : "l_shoulder",
    "LeftElbow"     : "l_elbow",
    "LeftWrist"     : "l_wrist",
}



@dataclass
class SkeletonSpec:
    joints: List[str]                    # Reihenfolge der Joints
    edges: List[Tuple[int, int]]         # (child, parent), Indizes in joints
    parents: List[int]                   # parent[i] = Elternindex oder -1 für Root
    children: List[List[int]]            # children[i] = Liste der Kinder
    roots: List[int]                     # i mit parent[i] == -1
    topo_order: List[int]                # topologische Reihenfolge (Root->Blätter)

class SkeletonParser:
    def __init__(self):
        self.spec: Optional[SkeletonSpec] = None


    def read_skeleton_json(self, path: str) -> SkeletonSpec:
        with open(path, "r") as f:
            skeleton_json = json.load(f)


        joints = skeleton_json["joints"]
        edges_raw = [tuple(edge) for edge in skeleton_json["hierarchy"]]

        # Compute parents, children, roots, and topo_order
        parents = [-1] * len(joints)
        children = [[] for _ in joints]
        for child, parent in edges_raw:
            parents[child] = parent
            children[parent].append(child)
        roots = [i for i, p in enumerate(parents) if p == -1]

        # Topological order (simple BFS)
        topo_order = []
        queue = deque(roots)
        while queue:
            node = queue.popleft()
            topo_order.append(node)
            queue.extend(children[node])

        spec = SkeletonSpec(
            joints=joints,
            edges=edges_raw,
            parents=parents,
            children=children,
            roots=roots,
            topo_order=topo_order
        )
        self.spec = spec
        return spec


    def get_joint_handle_by_name(self, skel: nimble.biomechanics.OpenSimFile, name):
        # Suche im Nimble-Skelett nach einem Joint, dessen Name 'name' enthält
        for i in range(skel.getNumJoints()):
            j = skel.getJoint(i)
            if name in j.getName():
                return j
        raise KeyError(f"Nimble-Joint '{name}' nicht gefunden. Passe MAP an (gefunden: {[skel.getJoint(i).getName() for i in range(skel.getNumJoints())]})")



    def build_nimble_body_joints(self, src_joint_names: List[str], skeleton: nimble.dynamics.Skeleton, poses: np.ndarray):
        # Baue die Liste der Ziel-Joints (Nimble) in derselben Reihenfolge wie unsere Quellpunkte
        bodyJoints = []
        src_indices = []
        for i, src_name in enumerate(src_joint_names):
            if src_name in MAP:
                bodyJoints.append(self.get_joint_handle_by_name(skeleton, MAP[src_name]))
                src_indices.append(i)

        bodyJoints = bodyJoints
        src_indices = np.array(src_indices, dtype=int)

        # -----------------------------
        # IK-Fit pro Frame
        # -----------------------------
        T = poses.shape[0]
        fitted_states = []  # hier speichern wir z.B. generalized coordinates q pro Frame

        # Optional: Anfangspose neutral
        skeleton.setPositions(np.zeros(skeleton.getNumDofs()))

        for t in range(T):
            # Zielpositionen in Weltkoordinaten (Liste von 3D-ndarrays)
            targetPositions = [poses[t, j].astype(np.float64) for j in src_indices]
            # IK: passt die Pose des Skeletts an diese Gelenkzentren an
            skeleton.fitJointsToWorldPositions(bodyJoints, targetPositions)  # optional: scaleBodies=True
            fitted_states.append(skeleton.getPositions().copy())

        fitted_states = np.stack(fitted_states, axis=0)  # (T, dofs)
        np.save("fitted_nimble_states.npy", fitted_states)

        print("Fertig: gespeicherte Nimble-DoFs pro Frame in 'fitted_nimble_states.npy'")