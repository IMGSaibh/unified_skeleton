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


    def _targets_column_from_world_points(self, points_3d: np.ndarray) -> np.ndarray:
        """
        points_3d: (N,3) float -> (3N,1) float64
        """
        arr = np.asarray(points_3d, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"Erwarte (N,3), bekam {arr.shape}")
        return arr.reshape(-1, 1)

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


    def build_nimble_body_joints(
        self,
        src_joint_names: List[str],
        skeleton: nimble.dynamics.Skeleton,
        poses: np.ndarray,
        *,
        unit_scale: float = 1.0,           
        axis_map = ("x","y","z"),
        scale_bodies: bool = False,
        damping: float = 1e-2,
        max_steps: int = 100,
        verbose: bool = False
    ):
        """
        - Baut die Joint-Liste auf dem Ziel-Skelett in gleicher Reihenfolge wie Quellpunkte (über MAP).
        - Führt pro Frame IK durch und speichert DoFs (Q) als (T, dofs).
        - Gibt (Q, gemappte_Namen, gemappte_Indices) zurück.
        """
        # 1) Joint-Paare (Ziel-Namen) gemäß MAP für vorhandene Source-Namen aufbauen
        bodyJoints = []
        src_indices = []
        mapped_names = []
        for i, src_name in enumerate(src_joint_names):
            if src_name in MAP:
                tgt_name = MAP[src_name]  # MAP: Source -> Target
                try:
                    bodyJoints.append(self.get_joint_handle_by_name(skeleton, tgt_name))
                    src_indices.append(i)
                    mapped_names.append(tgt_name)
                except KeyError as e:
                    if verbose:
                        print("Warnung:", e)
                    continue

        if not bodyJoints:
            raise ValueError("Kein einziges Joint-Paar gemappt. Prüfe MAP und Namen.")

        src_indices = np.asarray(src_indices, dtype=int)

        # 2) Poses in erwartete Form bringen
        arr = np.asarray(poses)
        if arr.ndim == 2 and arr.shape[-1] == 3:
            arr = arr[None, ...]
        if not (arr.ndim == 3 and arr.shape[-1] == 3):
            raise ValueError(f"poses erwartet (T,J,3) oder (J,3); bekam {arr.shape}")
        T, J, _ = arr.shape

        # 3) Einheiten/Koordinaten anpassen (nur einfache Einheit hier; Achsen optional später)
        arr = (arr * unit_scale).astype(np.float64, copy=False)

        # 4) IK pro Frame
        fitted_states = []
        # skeleton.setPositions(np.zeros(skeleton.getNumDofs()))  # neutrale Startpose

        for t in range(T):
            targets_frame = arr[t, src_indices, :]              # (N,3)
            targets_vec = self._targets_column_from_world_points(targets_frame)  # (3N,1)

            residual = skeleton.fitJointsToWorldPositions(
                bodyJoints,
                targets_vec,
                # scaleBodies=scale_bodies,
                # convergenceThreshold=1e-7,
                # maxStepCount=max_steps,
                # leastSquaresDamping=damping,
                # lineSearch=True,
                # logOutput=verbose
            )
            if verbose and (t % 50 == 0):
                print(f"[{t+1}/{T}] residual={residual:.6f}")

            fitted_states.append(skeleton.getPositions().copy())

        Q = np.stack(fitted_states, axis=0)  # (T, dofs)
        np.save("fitted_nimble_states.npy", Q)
        if verbose:
            print("Fertig: Nimble-DoFs gespeichert -> fitted_nimble_states.npy")
        return Q, mapped_names, src_indices
