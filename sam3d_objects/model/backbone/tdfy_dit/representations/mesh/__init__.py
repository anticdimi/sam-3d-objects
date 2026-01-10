# Copyright (c) Meta Platforms, Inc. and affiliates.

try:
    from .cube2mesh import SparseFeatures2Mesh, MeshExtractResult
except ModuleNotFoundError:  # pragma: no cover
    SparseFeatures2Mesh = None
    MeshExtractResult = None
