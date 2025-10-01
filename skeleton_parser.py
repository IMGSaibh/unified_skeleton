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
class Mapping:
    src_indices: List[int]                 # Indizes in source "joints"
    tgt_names: List[str]                   # Nimble-Joint-Namen
    tgt_joints: List[nimble.dynamics.Joint]

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
    

    def build_mapping(
        self,
        target_skeleton: nimble.dynamics.Skeleton,
        name_map: Dict[str, str] = MAP,
        strict: bool = False
    ) -> Mapping:
        """
        Liefert ein kompaktes Mapping Quelle->Nimble.
        - Quelle: self.skeleton_spec.joints (aus skeleton.json)
        - Ziel:   nimble.dynamics.Skeleton (z.B. Rajagopal)
        """
        if self.skeleton_spec is None:
            raise ValueError("SkeletonSpec ist None. Erst read_skeleton_json() aufrufen.")

        src_indices: List[int] = []
        tgt_names: List[str] = []
        tgt_joints: List[nimble.dynamics.Joint] = []

        # Hilfsfunktion: exakte Namen bevorzugen, sonst Substring-Fallback (deine bestehende Logik)
        def _get_joint_handle(name: str) -> nimble.dynamics.Joint:
            # exakter Treffer
            for i in range(target_skeleton.getNumJoints()):
                j = target_skeleton.getJoint(i)
                if j.getName() == name:
                    return j
            # Substring-Fallback
            for i in range(target_skeleton.getNumJoints()):
                j = target_skeleton.getJoint(i)
                if name in j.getName():
                    return j
            raise KeyError(name)

        missing_src: List[str] = []
        missing_tgt: List[str] = []

        for i, src_name in enumerate(self.skeleton_spec.joints):
            tgt_name = name_map.get(src_name)
            if not tgt_name:
                missing_src.append(src_name)
                continue
            try:
                jh = _get_joint_handle(tgt_name)
            except KeyError:
                missing_tgt.append(tgt_name)
                continue

            src_indices.append(i)
            tgt_names.append(tgt_name)
            tgt_joints.append(jh)

        if strict and (missing_src or missing_tgt):
            raise KeyError(
                f"Mapping unvollständig. Fehlende Source:{missing_src} Fehlende Target:{missing_tgt}"
            )

        # für spätere Nutzung optional speichern (ersetzt deine bodyJoints/src_indices)
        self.bodyJoints = tgt_joints[:]          # Handles in Zielreihenfolge
        self.src_indices = src_indices[:]        # Source-Indices in derselben Reihenfolge

        return Mapping(src_indices=src_indices, tgt_names=tgt_names, tgt_joints=tgt_joints)  # :contentReference[oaicite:3]{index=3}


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
            f"\nPasse MAP an (gefunden: {[skel.getJoint(i).getName() for i in range(skel.getNumJoints())]})"
    )

    def create_body_joints_list(self, target_skeleton: nimble.dynamics.Skeleton):
        """       
        - Baut die Joint-Liste auf dem Ziel-Skelett in gleicher Reihenfolge wie Quellpunkte (über MAP).
        """
        mapped_names = []
        if self.skeleton_spec is None:
            print("SkeletonSpec ist None. Lade zuerst ein Skelett.")
            return
        for i, src_name in enumerate(self.skeleton_spec.joints):
            if src_name in MAP:
                tgt_name = MAP[src_name]  # MAP: Source -> Target
                target_skeleton.getJoint(tgt_name)
                try:
                    self.bodyJoints.append(self.get_joint_handle_by_name(target_skeleton, tgt_name))
                except KeyError as e:
                    print(e)
                    print("===============================")
                    continue
                self.src_indices.append(i)
                mapped_names.append(tgt_name)

        if not self.bodyJoints:
            raise ValueError("Kein einziges Joint-Paar gemappt. Prüfe MAP und Namen.")
