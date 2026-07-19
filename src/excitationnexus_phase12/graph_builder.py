from __future__ import annotations

import json
from pathlib import Path

import torch
from torch_geometric.data import Data

ROLE_TO_INDEX = {"donor": 0, "acceptor": 1, "unknown": 2}


def _directed_radius_graph(pos: torch.Tensor, cutoff: float, max_neighbors: int) -> torch.Tensor:
    dist = torch.cdist(pos, pos)
    edges = []
    n = pos.shape[0]
    for dst in range(n):
        candidates = [(float(dist[src, dst]), src) for src in range(n)
                      if src != dst and float(dist[src, dst]) <= cutoff]
        for _, src in sorted(candidates, key=lambda x: (x[0], x[1]))[:max_neighbors]:
            edges.append((src, dst))
    if not edges:
        raise ValueError("graph has no cutoff edges")
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def build_graph_from_json(structure_path: str | Path, sidecar_path: str | Path,
                          *, cutoff: float = 5.0, max_neighbors: int = 32) -> Data:
    structure = json.loads(Path(structure_path).read_text())
    sidecar = json.loads(Path(sidecar_path).read_text())
    atoms = sorted(structure["atoms"], key=lambda a: int(a["index"]))
    expected = list(range(1, len(atoms) + 1))
    if [int(a["index"]) for a in atoms] != expected:
        raise ValueError("non-contiguous atom indices")
    roles = [str(a.get("type", "")).lower() for a in atoms]
    origins = [str(x).lower() for x in sidecar.get("atom_origins", [])]
    if len(origins) != len(atoms) or roles != origins:
        raise ValueError("atom role missing or inconsistent with sidecar")
    if any(r not in ROLE_TO_INDEX for r in roles):
        raise ValueError("unknown atom-role token")
    z = []
    periodic = torch.tensor([0])  # marker to keep mapping local and dependency-free
    del periodic
    symbols = {"H": 1, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Si": 14,
               "P": 15, "S": 16, "Cl": 17, "Br": 35, "Se": 34, "I": 53}
    for atom in atoms:
        value = atom.get("atomic_num")
        if value is None:
            value = symbols.get(str(atom["element"]).capitalize())
        if value is None or int(value) <= 0:
            raise ValueError(f"unsupported element: {atom.get('element')}")
        z.append(int(value))
    pos = torch.tensor([a["coords"] for a in atoms], dtype=torch.float32)
    if pos.shape != (len(atoms), 3) or not torch.isfinite(pos).all():
        raise ValueError("invalid coordinates")
    role = torch.tensor([ROLE_TO_INDEX[r] for r in roles], dtype=torch.long)
    return Data(z=torch.tensor(z, dtype=torch.long), pos=pos, role=role,
                donor_mask=role.eq(0), acceptor_mask=role.eq(1), unknown_mask=role.eq(2),
                edge_index=_directed_radius_graph(pos, cutoff, max_neighbors),
                num_nodes=len(atoms))


def raw_paths(raw_root: str | Path, molecule_id: str, view: str) -> tuple[Path, Path]:
    root = Path(raw_root)
    if view == "tier1_pm6_3d":
        folder, suffix = "pm6", "pm6"
    elif view == "tier2_dft_3d":
        folder, suffix = "dft", "dft"
    else:
        raise ValueError(f"no 3D raw path for {view}")
    base = root / folder / "results" / molecule_id
    return base / f"{molecule_id}_{suffix}.json", base / f"{molecule_id}_sidecar.json"
