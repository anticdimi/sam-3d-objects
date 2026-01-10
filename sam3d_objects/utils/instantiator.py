from __future__ import annotations

from functools import partial
from importlib import import_module
from typing import Any

from omegaconf import DictConfig, ListConfig, OmegaConf


def _locate(path: str) -> Any:
    parts = path.split('.')
    if len(parts) < 2:
        raise ImportError(f'Invalid target: {path!r}')

    last_error: Exception | None = None
    for i in range(len(parts), 0, -1):
        module_name = '.'.join(parts[:i])
        try:
            obj: Any = import_module(module_name)
        except ModuleNotFoundError as e:
            last_error = e
            continue
        for attr in parts[i:]:
            obj = getattr(obj, attr)
        return obj

    raise ModuleNotFoundError(
        f'Could not import any module prefix for _target_: {path}'
    ) from last_error


def _to_plain(obj: Any) -> Any:
    if isinstance(obj, (DictConfig, ListConfig)):
        return OmegaConf.to_container(obj, resolve=True)
    return obj


def instantiate(cfg: Any, *args: Any, **kwargs: Any) -> Any:
    """Hydra-like instantiate() for OmegaConf-style configs.

    Supports dict/DictConfig objects with:
    - `_target_`: import path to class/callable
    - `_args_`: optional positional args list
    - `_partial_`: if True, returns functools.partial

    Falls back to recursively instantiating nested dict/list configs.
    """

    if cfg is None:
        return None

    if isinstance(cfg, (ListConfig, list, tuple)):
        items = list(cfg)
        out = [instantiate(v) for v in items]
        return tuple(out) if isinstance(cfg, tuple) else out

    if isinstance(cfg, (DictConfig, dict)):
        cfg_dict = dict(_to_plain(cfg))
        if '_target_' not in cfg_dict:
            return {k: instantiate(v) for k, v in cfg_dict.items()}

        make_partial = bool(cfg_dict.pop('_partial_', False))
        cfg_args = cfg_dict.pop('_args_', [])

        target = cfg_dict.pop('_target_')
        cls_or_fn = _locate(str(target))

        params = {k: instantiate(v) for k, v in cfg_dict.items() if not k.startswith('_')}
        params.update({k: instantiate(v) for k, v in kwargs.items()})

        call_args = [instantiate(a) for a in cfg_args] + [instantiate(a) for a in args]
        if make_partial:
            return partial(cls_or_fn, *call_args, **params)

        return cls_or_fn(*call_args, **params)

    return cfg
