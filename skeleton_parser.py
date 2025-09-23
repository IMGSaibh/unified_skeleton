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


