"""SAM3D-Objects optional initialization.

The top-level package imports this module unless `LIDRA_SKIP_INIT` is set.
Keep this file lightweight and free of hard optional-dependency imports so that
`import sam3d_objects` succeeds in minimal installations.

If the upstream project needs additional initialization hooks (e.g. registering
OmegaConf resolvers), add them here behind local/optional imports.
"""

from __future__ import annotations


def init() -> None:
    """Run optional initialization.

    Intentionally a no-op by default.
    """


# Run on import for backward-compat with existing callers.
init()
