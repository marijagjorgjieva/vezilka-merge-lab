#!/usr/bin/env python3
"""Compatibility launcher for mergekit-yaml with Pydantic 2.10.x."""

from __future__ import annotations

import torch

from mergekit.architecture.base import (
    ConfiguredModelArchitecture,
    ConfiguredModuleArchitecture,
    ModelArchitecture,
    ModuleDefinition,
    WeightInfo,
)
from mergekit.architecture.json_definitions import (
    JsonLayerTemplates,
    JsonModularArchitectureDefinition,
    JsonModuleArchDef,
    JsonModuleArchitecture,
    JsonModuleDefinition,
)
from mergekit.scripts.run_yaml import main


def rebuild_pydantic_models() -> None:
    namespace = {"torch": torch}
    for model in [
        WeightInfo,
        ConfiguredModuleArchitecture,
        ModuleDefinition,
        ModelArchitecture,
        ConfiguredModelArchitecture,
        JsonLayerTemplates,
        JsonModuleArchDef,
        JsonModuleArchitecture,
        JsonModuleDefinition,
        JsonModularArchitectureDefinition,
    ]:
        model.model_rebuild(_types_namespace=namespace)


if __name__ == "__main__":
    rebuild_pydantic_models()
    main()
