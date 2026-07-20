#!/usr/bin/env python3
"""Validation analysis entry point; fail closed when no frozen predictions exist."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 registry=json.loads((ROOT/"data_registry/gate2d2_model_registry.json").read_text())
 if registry.get("validation_predictions_created",0)!=12:
  raise RuntimeError("analysis forbidden: twelve frozen validation predictions do not exist")
if __name__=="__main__": main()
