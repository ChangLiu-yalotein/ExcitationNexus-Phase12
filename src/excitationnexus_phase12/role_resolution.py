from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import networkx as nx
from rdkit import Chem


ALLOWED_ROLES = {"donor", "acceptor", "unknown"}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_structure_fields(path: str | Path) -> dict:
    """Read only target-free structural fields from a PM6/DFT JSON."""
    raw = json.loads(Path(path).read_text())
    atoms = sorted(raw["atoms"], key=lambda atom: int(atom["index"]))
    bonds = sorted(
        raw.get("bonds", []),
        key=lambda bond: (min(int(bond["atom1_index"]), int(bond["atom2_index"])),
                          max(int(bond["atom1_index"]), int(bond["atom2_index"]))),
    )
    return {
        "atoms": [{
            "index": int(atom["index"]), "element": str(atom["element"]),
            "coords": tuple(float(value) for value in atom["coords"]),
            "role": str(atom.get("type", "")).lower(),
        } for atom in atoms],
        "bonds": [{
            "a": int(bond["atom1_index"]), "b": int(bond["atom2_index"]),
            "type": str(bond.get("type", "single")).lower(),
        } for bond in bonds],
    }


def load_sidecar_roles(path: str | Path) -> list[str]:
    raw = json.loads(Path(path).read_text())
    return [str(value).lower() for value in raw.get("atom_origins", [])]


def graph_from_fields(fields: dict, atom_indices: set[int] | None = None, *, heavy_only: bool = True) -> nx.Graph:
    graph = nx.Graph()
    for atom in fields["atoms"]:
        if atom_indices is not None and atom["index"] not in atom_indices:
            continue
        if heavy_only and atom["element"] == "H":
            continue
        graph.add_node(atom["index"], element=atom["element"], formal_charge=0)
    for bond in fields["bonds"]:
        if bond["a"] in graph and bond["b"] in graph:
            graph.add_edge(bond["a"], bond["b"], bond_type=bond["type"])
    return graph


def component_nonplaceholder_formula(smiles: str) -> tuple[Counter, bool]:
    mol = Chem.MolFromSmiles(str(smiles).replace("[A]", "[*]"), sanitize=False)
    if mol is None:
        return Counter(), False
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
    except Exception:
        return Counter(), False
    atoms = [atom for atom in mol.GetAtoms() if atom.GetAtomicNum() > 0]
    neutral = all(atom.GetFormalCharge() == 0 for atom in atoms)
    return Counter(atom.GetSymbol() for atom in atoms), neutral


def resolve_empty_donor(fields: dict, donor_smiles: str, *, max_mapping_count: int = 100_000) -> dict:
    unknown = {atom["index"] for atom in fields["atoms"] if atom["role"] == "unknown"}
    unknown_heavy = {atom["index"] for atom in fields["atoms"]
                     if atom["role"] == "unknown" and atom["element"] != "H"}
    query = graph_from_fields(fields, unknown_heavy)
    full = graph_from_fields(fields)
    component_formula, neutral = component_nonplaceholder_formula(donor_smiles)
    query_formula = Counter(data["element"] for _, data in query.nodes(data=True))
    if not unknown or not query.nodes:
        return {"status": "UNRESOLVED_INSUFFICIENT_EVIDENCE", "mapping_multiplicity": 0,
                "distinct_role_sets": 0, "resolved_donor_indices": [],
                "reason": "no explicit unknown candidate graph"}
    if not neutral:
        return {"status": "UNRESOLVED_INSUFFICIENT_EVIDENCE", "mapping_multiplicity": 0,
                "distinct_role_sets": 0, "resolved_donor_indices": [],
                "reason": "charged donor cannot be checked because DFT atom JSON has no formal-charge field"}
    if component_formula != query_formula or not nx.is_connected(query):
        return {"status": "UNRESOLVED_INCONSISTENT", "mapping_multiplicity": 0,
                "distinct_role_sets": 0, "resolved_donor_indices": [],
                "reason": f"component/query formula or connectivity mismatch: {dict(component_formula)} vs {dict(query_formula)}"}
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        full, query,
        node_match=lambda left, right: left["element"] == right["element"] and left["formal_charge"] == right["formal_charge"],
        edge_match=lambda left, right: left["bond_type"] == right["bond_type"],
    )
    sets: set[tuple[int, ...]] = set()
    multiplicity = 0
    for mapping in matcher.subgraph_isomorphisms_iter():
        multiplicity += 1
        sets.add(tuple(sorted(mapping.keys())))
        if len(sets) > 1:
            return {"status": "UNRESOLVED_AMBIGUOUS", "mapping_multiplicity": multiplicity,
                    "distinct_role_sets": len(sets), "resolved_donor_indices": [],
                    "reason": "multiple element/bond-equivalent atom-role sets"}
        if multiplicity >= max_mapping_count:
            return {"status": "UNRESOLVED_AMBIGUOUS", "mapping_multiplicity": multiplicity,
                    "distinct_role_sets": len(sets), "resolved_donor_indices": [],
                    "reason": "mapping enumeration safety limit reached"}
    if not sets:
        return {"status": "UNRESOLVED_INCONSISTENT", "mapping_multiplicity": 0,
                "distinct_role_sets": 0, "resolved_donor_indices": [],
                "reason": "no element/bond/neutral-charge-aware full-graph match"}
    heavy = set(next(iter(sets)))
    resolved = set(heavy)
    # Hydrogens inherit donor only when bonded exclusively to a resolved donor heavy atom.
    adjacency = {atom["index"]: set() for atom in fields["atoms"]}
    for bond in fields["bonds"]:
        adjacency[bond["a"]].add(bond["b"]); adjacency[bond["b"]].add(bond["a"])
    elements = {atom["index"]: atom["element"] for atom in fields["atoms"]}
    for index, element in elements.items():
        if element == "H" and adjacency[index] and adjacency[index].issubset(heavy):
            resolved.add(index)
    return {"status": "RESOLVED_UNIQUE_OR_SYMMETRY_EQUIVALENT", "mapping_multiplicity": multiplicity,
            "distinct_role_sets": 1, "resolved_donor_indices": sorted(resolved),
            "reason": "all complete graph mappings yield one atom-role set"}
