from collections import deque
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from typing import List, Tuple
import nimblephysics as nimble
import numpy as np

MAP = {
    # Becken / Root
    "Hips"          : "ground_pelvis",
    
    # Beine rechts
    "RightHip"      : "hip_r",
    "RightKnee"     : "walker_knee_r",
    "RightAnkle"    : "ankle_r",
    "RightToe"      : "mtp_r",
    
    # Beine links
    "LeftHip"       : "hip_l",
    "LeftKnee"      : "walker_knee_l",
    "LeftAnkle"     : "ankle_l",
    "LeftToe"       : "mtp_l",
    
    # Wirbelsäule
    "Chest4"        : "back",
    
    # Hals/Kopf
    "Neck"          : "neck",
    "Head"          : "head",
    
    # Schultergürtel rechts
    "RightShoulder" : "acromial_r",
    "RightElbow"    : "elbow_r",
    "RightWrist"    : "radius_hand_r",
    
    # Schultergürtel links
    "LeftShoulder"  : "acromial_l",
    "LeftElbow"     : "elbow_l",
    "LeftWrist"     : "radius_hand_l",
}

@dataclass
class SkeletonSpec:
    joints: List[str]                    # Reihenfolge der Joints
    edges: List[Tuple[int, int]]         # (child, parent), Indizes in joints
    parents: List[int]                   # parent[i] = Elternindex oder -1 für Root
    children: List[List[int]]            # children[i] = Liste der Kinder
    roots: List[int]                     # i mit parent[i] == -1

class SkeletonParser:
    def __init__(self):
        self.skeleton_spec: Optional[SkeletonSpec] = None
        self.bodyJoints = []
        

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


        spec = SkeletonSpec(
            joints=joints,
            edges=edges_raw,
            parents=parents,
            children=children,
            roots=roots,
        )
        self.skeleton_spec = spec
        return spec
    
    def _targets_column_from_world_points(self, points_3d: np.ndarray) -> np.ndarray:
        """
        points_3d: (N, 3) -> (3N, 1)  float64
        Reihenfolge: [x1, y1, z1, x2, y2, z2, ...] (C-order)
        """
        result_array = np.asarray(points_3d, dtype=np.float64)
        if result_array.ndim != 2 or result_array.shape[1] != 3:
            raise ValueError(f"Erwarte (N,3), bekam {result_array.shape}")
        result_array = np.ascontiguousarray(result_array)
        return result_array.reshape(-1, 1, order="C")      

    def get_joint_handle_by_name(self, skel: nimble.dynamics.Skeleton, name: str):
        # exakte Treffer bevorzugen
        for i in range(skel.getNumJoints()):
            j = skel.getJoint(i)
            if j.getName() == name:
                return j
        # fallback: substring
        for i in range(skel.getNumJoints()):
            j = skel.getJoint(i)
            if name in j.getName():
                return j
        raise KeyError(
            f"Nimble-Joint '{name}' nicht gefunden. "
            f"Passe MAP an (gefunden: {[skel.getJoint(i).getName() for i in range(skel.getNumJoints())]})"
    )

    def create_body_joints_list(self, target_skeleton: nimble.dynamics.Skeleton):
        """       
        - Baut die Joint-Liste auf dem Ziel-Skelett in gleicher Reihenfolge wie Quellpunkte (über MAP).
        """
        src_indices = []
        mapped_names = []
        if self.skeleton_spec is None:
            print("SkeletonSpec ist None. Lade zuerst ein Skelett.")
            return
        for i, src_name in enumerate(self.skeleton_spec.joints):
            if src_name in MAP:
                tgt_name = MAP[src_name]  # MAP: Source -> Target
                self.bodyJoints.append(self.get_joint_handle_by_name(target_skeleton, tgt_name))
                src_indices.append(i)
                mapped_names.append(tgt_name)

        if not self.bodyJoints:
            raise ValueError("Kein einziges Joint-Paar gemappt. Prüfe MAP und Namen.")


    def build_nimble_body_joints(
        self,
        target_skeleton: nimble.dynamics.Skeleton,
        poses: np.ndarray,
        *,
        unit_scale: float = 1.0,           
        scale_bodies: bool = False,
        damping: float = 1e-2,
        max_steps: int = 100,
    ):
        """
        - Führt pro Frame IK durch und speichert DoFs (Q) als (T, dofs).
        - Gibt (Q, gemappte_Namen, gemappte_Indices) zurück.
        """

        if self.skeleton_spec is None:
            return 
        src_indices = np.asarray(self.skeleton_spec.joints, dtype=int)

        # 4) IK pro Frame
        fitted_states = []

        print(target_skeleton.getJoint(1).getName())

        # poses: (T, J, 3)
        for t, frame in enumerate(poses):
            # frame: (J, 3)
            targets_frame = frame[src_indices, :]  # (N, 3) – src_indices: int-Indices der gewünschten Joints
            targets_vec = self._targets_column_from_world_points(targets_frame)  # (3N, 1) float64

            residual = target_skeleton.fitJointsToWorldPositions(
                self.bodyJoints,
                targets_vec,
                scaleBodies=scale_bodies,
                convergenceThreshold=1e-7,
                maxStepCount=max_steps,
                leastSquaresDamping=damping,
                lineSearch=True,
            )

            fitted_states.append(target_skeleton.getPositions().copy())

        Q = np.stack(fitted_states, axis=0)  # (T, dofs)
        np.save("fitted_nimble_states.npy", Q)

        
        return Q
