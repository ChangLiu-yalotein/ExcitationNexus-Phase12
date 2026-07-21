from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return sha256_text(payload)


@dataclass(frozen=True)
class ComponentSite:
    role: str
    structure_identity: str
    source_smiles: str
    clean_smiles: str
    anchor_index: int
    placeholder_degree: int
    source_sha256: str

    def mol(self) -> Chem.Mol:
        mol = Chem.MolFromSmiles(self.clean_smiles)
        if mol is None:
            raise ValueError("frozen clean component no longer parses")
        return mol

    def marked_attachment_smiles(self) -> str:
        mol = self.mol()
        mol.GetAtomWithIdx(self.anchor_index).SetAtomMapNum(1)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def parse_component_site(raw_smiles: str, role: str, structure_identity: str) -> ComponentSite:
    if not isinstance(raw_smiles, str) or raw_smiles.count("[A]") != 1:
        raise ValueError("component must contain exactly one literal [A] marker")
    marker_mol = Chem.MolFromSmiles(raw_smiles.replace("[A]", "[*]"), sanitize=False)
    if marker_mol is None:
        raise ValueError("component marker graph does not parse")
    dummy_atoms = [atom for atom in marker_mol.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummy_atoms) != 1 or dummy_atoms[0].GetDegree() < 1:
        raise ValueError("component must expose exactly one connected dummy marker")
    dummy = dummy_atoms[0]
    source_anchor = list(dummy.GetNeighbors())[0].GetIdx()
    anchor_index = source_anchor - int(source_anchor > dummy.GetIdx())
    clean_mol = Chem.MolFromSmiles(raw_smiles.replace("[A]", ""))
    if clean_mol is None or not 0 <= anchor_index < clean_mol.GetNumAtoms():
        raise ValueError("clean component or attachment index is invalid")
    clean_smiles = Chem.MolToSmiles(clean_mol, canonical=False, isomericSmiles=True)
    return ComponentSite(role, structure_identity, raw_smiles, clean_smiles, anchor_index,
                         dummy.GetDegree(), sha256_text(raw_smiles))


def assemble_components(donor: ComponentSite, acceptor: ComponentSite) -> Chem.Mol:
    donor_mol, acceptor_mol = donor.mol(), acceptor.mol()
    combined = Chem.RWMol(Chem.CombineMols(donor_mol, acceptor_mol))
    donor_anchor = donor.anchor_index
    acceptor_anchor = donor_mol.GetNumAtoms() + acceptor.anchor_index
    for index in (donor_anchor, acceptor_anchor):
        atom = combined.GetAtomWithIdx(index)
        if atom.GetNumExplicitHs() > 0:
            atom.SetNumExplicitHs(atom.GetNumExplicitHs() - 1)
    combined.AddBond(donor_anchor, acceptor_anchor, Chem.BondType.SINGLE)
    product = combined.GetMol()
    product.ClearComputedProps()
    Chem.SanitizeMol(product)
    return product


def canonical_isomeric(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def canonical_graph(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)


def graph_canonical_from_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError("full structure does not parse")
    return canonical_graph(mol)


def molecular_summary(mol: Chem.Mol) -> dict:
    elements = sorted({atom.GetSymbol() for atom in mol.GetAtoms()})
    return {
        "molecular_weight": float(Descriptors.MolWt(mol)),
        "heavy_atom_count": int(mol.GetNumHeavyAtoms()),
        "heteroatom_count": int(rdMolDescriptors.CalcNumHeteroatoms(mol)),
        "formal_charge": int(sum(atom.GetFormalCharge() for atom in mol.GetAtoms())),
        "element_inventory": ";".join(elements),
        "murcko_scaffold": MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False),
    }


def morgan_fingerprint(mol: Chem.Mol, radius: int = 2, bits: int = 2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=bits, useChirality=False)


def deterministic_pairwise_similarity(fingerprints: Iterable, max_items: int = 1024) -> dict:
    fps = list(fingerprints)[:max_items]
    if len(fps) < 2:
        return {"sample_size": len(fps), "pair_count": 0, "quantiles": {}}
    values: list[float] = []
    for index in range(1, len(fps)):
        values.extend(DataStructs.BulkTanimotoSimilarity(fps[index], fps[:index]))
    array = np.asarray(values, dtype=np.float64)
    return {
        "sample_size": len(fps),
        "pair_count": int(array.size),
        "quantiles": {str(q): float(np.quantile(array, q)) for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)},
    }
