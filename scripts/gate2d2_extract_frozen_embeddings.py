#!/usr/bin/env python3
"""Embedding extraction entry point; fail closed while the frozen PCA contract is blocked."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 status=json.loads((ROOT/"data_registry/gate2d2_pca_registry.json").read_text())["status"]
 if status=="BLOCKED_PREREGISTERED_PCA_INFEASIBLE":
  raise RuntimeError("embedding extraction forbidden: preregistered donor PCA-256 is infeasible")
 raise RuntimeError("Gate 2-D2 embedding execution requires an explicitly authorized unblocked preregistration")
if __name__=="__main__": main()
