import json
import numpy as np
import nimblephysics as nimble

def _axis_matrix(axis_map=("x","y","z")):
    basis = {
        "x": np.array([1.,0.,0.]),
        "y": np.array([0.,1.,0.]),
        "z": np.array([0.,0.,1.]),
    }
    M = np.zeros((3,3))
    for r, sym in enumerate(axis_map):
        sgn = -1.0 if sym.startswith("-") else 1.0
        ax = sym.lstrip("-")
        M[r,:] = sgn * basis[ax]
    return M

def create_free_joint(skel, parent=None, name="root"):
    # Versuche versionsabhängige Helfer – sonst generische Factory
    if hasattr(skel, "createFreeJointAndBodyNodePair"):
        joint, body = skel.createFreeJointAndBodyNodePair(parent)
    elif hasattr(skel, "createFreeJointAndBodyNode"):
        joint, body = skel.createFreeJointAndBodyNode(parent)          # einige ältere Wheels
    else:
        # Generische Factory: Joint-Typ als Klasse übergeben (robust)
        joint, body = skel.createJointAndBodyNodePair(nimble.dynamics.FreeJoint, parent)
    joint.setName(name)
    body.setName(f"{name}_body")
    return joint, body

def create_ball_joint(skel, parent_body, name, local_offset=None):
    if hasattr(skel, "createBallJointAndBodyNodePair"):
        joint, body = skel.createBallJointAndBodyNodePair(parent_body)
    elif hasattr(skel, "createBallJointAndBodyNode"):
        joint, body = skel.createBallJointAndBodyNode(parent_body)
    else:
        joint, body = skel.createJointAndBodyNodePair(nimble.dynamics.BallJoint, parent_body)
    joint.setName(name)
    body.setName(f"{name}_body")
    if local_offset is not None:
        T = nimble.math.Isometry3(); T.set_identity(); T.set_translation(local_offset)
        joint.setTransformFromParentBodyNode(T)
    return joint, body

def build_nimble_skeleton_from_json_npy(
    json_path: str,
    npy_path: str,
    unit_scale: float = 1.0,          # z.B. 0.001 bei mm→m
    axis_map = ("x","y","z")          # Quelle -> Nimble Weltachse
):
    """
    Baut ein nimble.dynamics.Skeleton aus:
      - skeleton.json  (joints, hierarchy mit self-parent für Roots)
      - poses.npy      (frames, joints, 3)
    - Root = FreeJoint
    - Kinder = BallJoint
    - Joint-Lokaloffset aus Frame 0 (nach Einheit + Achsen umgerechnet)
    """
    with open(json_path, "r") as f:
        js = json.load(f)
    joint_names = js["joints"]
    hierarchy = js["hierarchy"]      # [[child, parent], ...], Root: [i,i]

    poses = np.load(npy_path)        # (T, J, 3) oder (J,3)
    if poses.ndim == 2 and poses.shape[-1] == 3:
        poses = poses[None, ...]
    assert poses.ndim == 3 and poses.shape[-1] == 3, f"Unexpected npy shape: {poses.shape}"

    T, J, _ = poses.shape
    assert J == len(joint_names), "JSON und NPY haben unterschiedliche Joint-Anzahl"

    # In Meter + in Zielkoordinatenrahmen bringen (nur Frame 0 für Offsets)
    M = _axis_matrix(axis_map)
    P0 = (poses[0] * unit_scale) @ M.T  # (J,3)

    # Elternliste (-1 für Root)
    parents = []
    for c,p in hierarchy:
        parents.append(-1 if c == p else p)

    skel = nimble.dynamics.Skeleton()

    # Wir brauchen einen BodyNode-Handle je Joint, um Kinder anzuhängen
    body_nodes = [None]*J
    joints_obj = [None]*J

    # Einen Root wählen: ersten mit parent==-1
    roots = [i for i,p in enumerate(parents) if p == -1]
    if not roots:
        raise ValueError("Keine Root in der Hierarchie gefunden (erwarte [i,i] Einträge).")
    root_idx = roots[0]

    # 1) Root anlegen (FreeJoint)
    root_joint, root_body = skel.createFreeJointAndBodyNode()
    root_joint.setName(joint_names[root_idx])
    root_body.setName(joint_names[root_idx] + "_body")
    body_nodes[root_idx] = root_body
    joints_obj[root_idx] = root_joint

    # Optional: initiale Weltpose des Root aus P0 setzen (Translation; Rotation bleibt 0)
    q = skel.getPositions().copy()
    # FreeJoint-DoFs: [tx, ty, tz, rx, ry, rz] (Achtung: Nimble verwendet eine bestimmte Parametrisierung)
    q[:3] = P0[root_idx]  # nur Translation initialisieren
    skel.setPositions(q)

    # 2) Kinder rekursiv hinzufügen (Top-Down)
    children_of = {i: [] for i in range(J)}
    for c,p in enumerate(parents):
        if p >= 0:
            children_of[p].append(c)

    def add_subtree(parent_idx):
        parent_body = body_nodes[parent_idx]
        for c in children_of[parent_idx]:
            # BallJoint + BodyNode
            joint, body = skel.createBallJointAndBodyNode(parent_body)
            joint.setName(joint_names[c])
            body.setName(joint_names[c] + "_body")
            body_nodes[c] = body
            joints_obj[c] = joint

            # Lokaler Transform = Offset von Parent → Child aus P0
            offset = P0[c] - P0[parent_idx]
            T_pc = nimble.math.Isometry3()
            T_pc.set_identity()
            T_pc.set_translation(offset.astype(np.float64))
            joint.setTransformFromParentBodyNode(T_pc)

            # Rekursiv tiefer
            add_subtree(c)

    add_subtree(root_idx)

    # 3) Falls es mehrere Roots gibt, hängen wir sie mit Weld am Root zusammen
    for extra_root in roots[1:]:
        joint, body = skel.createWeldJointAndBodyNode(root_body)
        joint.setName(joint_names[extra_root] + "_weldToRoot")
        body.setName(joint_names[extra_root] + "_body")
        body_nodes[extra_root] = body
        joints_obj[extra_root] = joint
        # Position relativ zum Root aus P0
        offset = P0[extra_root] - P0[root_idx]
        T_pc = nimble.math.Isometry3()
        T_pc.set_identity()
        T_pc.set_translation(offset.astype(np.float64))
        joint.setTransformFromParentBodyNode(T_pc)

    return skel, joints_obj, body_nodes, joint_names, parents
