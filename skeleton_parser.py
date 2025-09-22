from collections import deque
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from typing import List, Tuple


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

    def _validate_and_normalize(self, joints: List[str], edges_raw: List[Tuple[int, int]]) -> SkeletonSpec:
        joints_count = len(joints)
        
        # Doppelte Namen checken
        check_doubles = [n for n in set(joints) if joints.count(n) > 1]
        if check_doubles:
            raise ValueError(f"Doppelte Joint-Namen: {check_doubles}")

        # Kanten-Indices & Root-Self-Parents normalisieren
        edges: List[Tuple[int, int]] = []
        for child, parent in edges_raw:
            if not (0 <= child < joints_count) or not (0 <= parent < joints_count):
                raise ValueError(f"Ungültiger Index in hierarchy: {(child, parent)} (0..{joints_count-1} erlaubt)")
            # Konvention: self-parent (z.B. [0,0]) bedeutet Root
            if child == parent:
                edges.append((child, -1))
            else:
                edges.append((child, parent))

        # Elternliste: -1 initial
        parents = [-1] * joints_count
        for child, parent in edges:
            if parents[child] != -1 and parent != -1:
                raise ValueError(f"Joint {child} ('{joints[child]}') hat mehrere Eltern: {parents[child]} und {parent}")
            parents[child] = parent

        # Falls einige Joints in edges nicht vorkommen => bleiben Root (-1)
        roots = [i for i in range(joints_count) if parents[i] == -1]
        if not roots:
            raise ValueError("Kein Root gefunden (mindestens ein Joint muss parent=-1 haben).")

        # Kinderliste
        children: List[List[int]] = [[] for _ in range(joints_count)]
        for child, parent in edges:
            if parent != -1:
                children[parent].append(child)

        # Zyklus-Check + Topologische Order (Kahn)
        indeg = [0] * joints_count
        for child, parent in edges:
            if parent != -1:
                indeg[child] += 1
        q = deque([i for i in range(joints_count) if indeg[i] == 0])
        topo: List[int] = []
        while q:
            u = q.popleft()
            topo.append(u)
            for v in children[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        if len(topo) != joints_count:
            raise ValueError("Die Hierarchie enthält einen Zyklus oder inkonsistente Kanten.")

        return SkeletonSpec(
            joints=joints,
            edges=edges,
            parents=parents,
            children=children,
            roots=roots,
            topo_order=topo,
        )


    def _guess_map_template(self, joints: List[str]) -> Dict[str, str]:
        """
        Erzeugt ein sinnvolles Template für map_to_nimble (Rajagopal-ähnliche Namen).
        Nicht-bindend; du kannst es später überschreiben/feintunen.
        """
        base = {
            # Achse
            "Hips":   "",        # Root-Center normalerweise nicht direkt fitten
            "Chest":  "spine",   # wenn im Modell vorhanden, sonst leer lassen
            "Chest2": "",
            "Chest3": "",
            "Chest4": "",
            "Neck":   "neck",
            "Head":   "head",

            # Rechts (Arm)
            "RightCollar":   "",
            "RightShoulder": "shoulder_r",
            "RightElbow":    "elbow_r",
            "RightWrist":    "wrist_r",

            # Links (Arm)
            "LeftCollar":    "",
            "LeftShoulder":  "shoulder_l",
            "LeftElbow":     "elbow_l",
            "LeftWrist":     "wrist_l",

            # Rechts (Bein)
            "RightHip":      "hip_r",
            "RightKnee":     "knee_r",
            "RightAnkle":    "ankle_r",
            "RightToe":      "toe_r",

            # Links (Bein)
            "LeftHip":       "hip_l",
            "LeftKnee":      "knee_l",
            "LeftAnkle":     "ankle_l",
            "LeftToe":       "toe_l",
        }
        # Fülle nur für vorhandene Namen; andere bleiben weg
        return {name: base.get(name, "") for name in joints}

    def read_skeleton_json(self, path: str, write_template: Optional[str] = None) -> SkeletonSpec:
        with open(path, "r") as f:
            data = json.load(f)

        if "joints" not in data or "hierarchy" not in data:
            raise ValueError("JSON benötigt Felder 'joints' (List[str]) und 'hierarchy' (List[[child,parent],...]).")

        joints = data["joints"]
        edges_raw = [tuple(edge) for edge in data["hierarchy"]]

        self.spec = self._validate_and_normalize(joints, edges_raw)

        if write_template:
            tmpl = {
                "map_to_nimble": self._guess_map_template(self.spec.joints),
                "unit_scale": 1.0,
                "scaleBodies": True
            }
            with open(write_template, "w") as f:
                json.dump(tmpl, f, indent=2)
            print(f"[Info] Mapping-Template geschrieben: {write_template}")

        return self.spec
    
    def print_skeleton_spec(self):
        if self.spec is None:
            print("No skeleton spec loaded.")
            return
        print("=== Skeleton Specification ===")
        print(f"Joints count: {len(self.spec.joints)}")
        print(f"Joints: {self.spec.joints}")
        print(f"Edges: {self.spec.edges}")
        print(f"Parents: {self.spec.parents}")
        print(f"Children: {self.spec.children}")
        print(f"Roots: {self.spec.roots}")
        print(f"Topological Order: {self.spec.topo_order}")

