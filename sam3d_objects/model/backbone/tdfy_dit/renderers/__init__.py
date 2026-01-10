# Copyright (c) Meta Platforms, Inc. and affiliates.
from .octree_renderer import OctreeRenderer

try:
    from .gaussian_render import GaussianRenderer
except ModuleNotFoundError:  # pragma: no cover
    GaussianRenderer = None

# handle case when nvdiffrast is not present on the machine
try:
    from .mesh_renderer import MeshRenderer
except ImportError:
    MeshRenderer = None
