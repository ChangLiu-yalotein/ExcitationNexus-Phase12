#!/usr/bin/env python3
"""Validation-only training entry point; fail closed before features are admitted."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 registry=json.loads((ROOT/"data_registry/gate2d2_model_registry.json").read_text())
 if registry["status"]!="GATE2D2_FEATURES_FROZEN":
  raise RuntimeError("training forbidden: Gate 2-D2 features are not frozen")
if __name__=="__main__": main()
