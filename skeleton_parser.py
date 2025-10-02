import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from typing import List, Tuple
import nimblephysics as nimble
import numpy as np

@dataclass
class Mapping:
    src_indices: List[int]                 # Indizes in source skeleton "joints"
    tgt_names: List[str]                   # Nimble-Joint-Namen
    tgt_joints: List[nimble.dynamics.Joint]

@dataclass
class SkeletonSpec:
    joints: List[str] = field(default_factory=list) # order of Joints
    edges: List[Tuple[int, int]] = field(default_factory=list) # (child, parent), Indizes in joints
    parents: List[int] = field(default_factory=list) # parent[i] = parent index or -1 für Root
    children: List[List[int]] = field(default_factory=list)  # children[i] = list of child indices
    roots: List[int] = field(default_factory=list) # i mit parent[i] == -1

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


class SkeletonParser:
    def __init__(self):
        self.skeleton_spec: Optional[SkeletonSpec] = None
        self.bodyJoints = []
        self.src_indices = []  # Indizes der gemappten Joints in der Quell-Skelettliste
        

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
    
    def _get_joint_handle(self, target_skeleton: nimble.dynamics.Skeleton, name: str) -> nimble.dynamics.Joint:
        # exakte Übereinstimmung
        for i in range(target_skeleton.getNumJoints()):
            joint = target_skeleton.getJoint(i)
            if joint.getName() == name:
                return joint
        # substring-Fallback
        for i in range(target_skeleton.getNumJoints()):
            joint = target_skeleton.getJoint(i)
            if name in joint.getName():
                return joint
        raise KeyError(name)
    
    def build_mapping(
        self,
        target_skeleton: nimble.dynamics.Skeleton,
        name_map: Dict[str, str] = MAP,
        strict: bool = False
    ) -> Mapping:
        """
        mapping Quelle->Nimble.
        - source: self.skeleton_spec.joints (from skeleton.json)
        - target: nimble.dynamics.Skeleton (Rajagopal openSim model)
        """
        if self.skeleton_spec is None:
            raise ValueError("SkeletonSpec ist None. Erst read_skeleton_json() aufrufen.")

        src_indices: List[int] = []
        target_joints_names: List[str] = []
        nimble_joints: List[nimble.dynamics.Joint] = []



        missing_src: List[str] = []
        missing_tgt: List[str] = []

        for i, src_skeleton_joint in enumerate(self.skeleton_spec.joints):
            tgt_name = name_map.get(src_skeleton_joint)
            if not tgt_name:
                missing_src.append(src_skeleton_joint)
                continue
            try:
                jh = self._get_joint_handle(target_skeleton, tgt_name)
            except KeyError:
                missing_tgt.append(tgt_name)
                continue

            src_indices.append(i)
            target_joints_names.append(tgt_name)
            nimble_joints.append(jh)

        if strict and (missing_src or missing_tgt):
            raise KeyError(
                f"Mapping unvollständig. Fehlende Source:{missing_src} Fehlende Target:{missing_tgt}"
            )

        # für spätere Nutzung optional speichern (ersetzt deine bodyJoints/src_indices)
        self.bodyJoints = nimble_joints[:]          # Handles in Zielreihenfolge
        self.src_indices = src_indices[:]        # Source-Indices in derselben Reihenfolge

        return Mapping(src_indices=src_indices, tgt_names=target_joints_names, tgt_joints=nimble_joints)


    def map_points_to_nimble_order(
        self,
        frame_points: np.ndarray,
        mapping: Mapping
    ) -> np.ndarray:
        """
        Nimmt ein (N,3) Frame (Quelle) und extrahiert/ordnet auf (K,3) in Nimble-Reihenfolge um.
        K = len(mapping.src_indices)
        """
        pts = np.asarray(frame_points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"Erwarte (N,3), bekam {pts.shape}")
        return pts[mapping.src_indices, :]       # (K,3)  :contentReference[oaicite:4]{index=4}

    def targets_column_from_points(self, points_3d: np.ndarray) -> np.ndarray:
        """
        (K,3) -> (3K,1)  in C-Order (x1,y1,z1,x2,y2,z2,...)
        """
        arr = np.asarray(points_3d, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"Erwarte (K,3), bekam {arr.shape}")
        return np.ascontiguousarray(arr).reshape(-1, 1, order="C") 

        """
        points_3d: (N, 3) -> (3N, 1)  float64
        Reihenfolge: [x1, y1, z1, x2, y2, z2, ...] (C-order)
        """
        result_array = np.asarray(points_3d, dtype=np.float64)
        if result_array.ndim != 2 or result_array.shape[1] != 3:
            raise ValueError(f"Erwarte (N,3), bekam {result_array.shape}")
        result_array = np.ascontiguousarray(result_array)
        return result_array.reshape(-1, 1, order="C")      

